# core/llm/chat_manager.py
import os
from typing import Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from operator import itemgetter
from huggingface_hub import hf_hub_download
from django.conf import settings

class ChatManager:
    def __init__(self):
        """Initialize the RAG system."""
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        print("Initializing local model...")
        self._setup_local_llm()
        
        # Initialize vector store path
        self.vector_store_path = os.path.join(settings.VECTOR_DB_PATH, "chroma")
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Setup prompt template - simplified for chat
        self.prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context:

Context: {context}

Question: {question}

Answer:"""
        )

    def _setup_local_llm(self):
        """Set up the local language model."""
        # Setup callback manager for streaming responses
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        # Download small quantized model
        print("Downloading model...")
        model_path = hf_hub_download(
            repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        )
        
        print("Initializing LlamaCpp...")
        self.llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=-1,  # Attempt to offload all layers to GPU
            n_batch=512,
            n_ctx=4096,       # Reduced context window
            f16_kv=True,
            callbacks=callback_manager,
            verbose=False,
            temperature=0.7,
            top_p=0.95,
            max_tokens=512,
        )

    def get_or_create_vector_store(self, collection_id):
        """Get or create a vector store for a specific collection."""
        persist_directory = os.path.join(self.vector_store_path, str(collection_id))
        os.makedirs(persist_directory, exist_ok=True)
        
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

    def process_documents(self, collection):
        """Process documents for a collection and store in vector store."""
        try:
            print(f"Processing documents for collection {collection.id}")
            vector_store = self.get_or_create_vector_store(collection.id)
            
            for document in collection.documents.all():
                print(f"Processing document: {document.title}")
                if not document.content:
                    continue
                    
                # Add document content directly
                vector_store.add_texts(
                    texts=[document.content],
                    metadatas=[{"source": f"Document: {document.title}"}]
                )
            
            return vector_store
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            return None

    def get_response(self, message: str, session) -> Optional[str]:
        """Get a response for a chat message using the RAG system."""
        try:
            print(f"Processing message: {message} for session {session.id}")
            
            # Handle direct responses if no documents
            if not session.collection.documents.exists():
                print("No documents found, generating direct response")
                return self.llm.invoke(message)

            # Process documents and create vector store
            vector_store = self.process_documents(session.collection)
            if not vector_store:
                return "I apologize, but I couldn't process the documents. Please try again."

            # Create retriever
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            
            # Create chain
            chain = (
                {
                    "context": itemgetter("question") | retriever,
                    "question": itemgetter("question")
                }
                | self.prompt
                | self.llm
            )
            
            # Get response
            print("Generating response...")
            response = chain.invoke({"question": message})
            print("Response generated successfully")
            
            return response
            
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return f"I encountered an error: {str(e)}"

# Singleton instance
chat_manager = None

def get_chat_manager():
    global chat_manager
    if chat_manager is None:
        print("Initializing chat manager...")
        chat_manager = ChatManager()
    return chat_manager