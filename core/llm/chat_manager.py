# core/llm/chat_manager.py
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from operator import itemgetter
from huggingface_hub import hf_hub_download
from django.conf import settings

class ChatManager:
    """
    Manages chat interactions with different LLM models, including document retrieval,
    conversation memory, and model management.
    """
    
    # Define available models
    AVAILABLE_MODELS = {
        "tinyllama": {
            "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "display_name": "TinyLlama 1.1B"
        },
        "capybara": {
            "repo_id": "TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF",
            "filename": "capybarahermes-2.5-mistral-7b.Q4_K_M.gguf",
            "display_name": "Capybara Hermes"
        },
        "rollama": {
            "repo_id": "RichardErkhov/Llama-2-7b-chat-hf-gguf",
            "filename": "Llama-2-7b-chat-hf.IQ3_XS.gguf",
            "display_name": "RichardErkov LLama 2-7-b"
        }
    }

    def __init__(self):
        """Initialize the RAG system with necessary components."""
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Initialize models directory
        self.models_dir = Path(settings.LLM_MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        print(f"Models directory: {self.models_dir}")
        
        # Initialize models dictionary and set default model
        self.models: Dict[str, LlamaCpp] = {}
        self.current_model_key = "tinyllama"
        
        # Initialize vector store path
        self.vector_store_path = os.path.join(settings.VECTOR_DB_PATH, "chroma")
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Setup chat prompt template
        self.prompt = ChatPromptTemplate.from_template(
            """Answer the question based on the following context and chat history.
            If the context doesn't contain relevant information, use your general knowledge
            while maintaining consistency with previous responses.

            Context: {context}
            
            Chat History: {chat_history}
            
            Current Question: {question}
            
            Answer:"""
        )

        # Setup system prompt for direct interactions
        self.system_prompt = """You are a helpful AI assistant. Provide clear, 
        accurate, and relevant responses while maintaining context from our previous 
        conversation. If you're unsure about something, acknowledge it and explain 
        what you do know."""

    def _get_model_path(self, model_key: str) -> Path:
        """
        Get the path where the model should be stored.
        
        Args:
            model_key (str): Key identifying the model in AVAILABLE_MODELS
            
        Returns:
            Path: Path where the model should be stored
            
        Raises:
            ValueError: If the model_key is not recognized
        """
        if model_key not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_key}")
        return self.models_dir / self.AVAILABLE_MODELS[model_key]["filename"]

    def _setup_model(self, model_key: str) -> LlamaCpp:
        """
        Set up a specific model, downloading if necessary and initializing it.
        
        Args:
            model_key (str): Key identifying the model to setup
            
        Returns:
            LlamaCpp: Initialized model instance
            
        Raises:
            Exception: If model setup fails
        """
        try:
            model_config = self.AVAILABLE_MODELS[model_key]
            model_path = self._get_model_path(model_key)
            
            print(f"Setting up {model_config['display_name']}...")
            print(f"Checking for model at: {model_path}")
            
            # Download model if it doesn't exist
            if not model_path.exists():
                print(f"Downloading {model_config['display_name']}...")
                downloaded_path = hf_hub_download(
                    repo_id=model_config["repo_id"],
                    filename=model_config["filename"],
                    local_dir=self.models_dir,
                    local_dir_use_symlinks=False
                )
                print(f"Model downloaded to: {downloaded_path}")
                
                if downloaded_path != model_path:
                    os.replace(downloaded_path, model_path)
                    print(f"Model moved to final location: {model_path}")
            else:
                print(f"Using existing model at: {model_path}")

            print(f"Initializing {model_config['display_name']}...")
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            model = LlamaCpp(
                model_path=str(model_path),
                n_gpu_layers=-1,  # Use all available GPU layers
                n_batch=512,
                n_ctx=4096,
                f16_kv=True,
                callbacks=callback_manager,
                verbose=False,
                temperature=0.7,
                top_p=0.95,
                max_tokens=512,
            )
            print(f"Successfully initialized model: {model_config['display_name']}")
            return model
            
        except Exception as e:
            print(f"Error in _setup_model: {str(e)}")
            raise

    def get_model(self, model_key: str = None) -> LlamaCpp:
        """
        Get a specific model, initializing it if necessary.
        
        Args:
            model_key (str, optional): Key identifying the model. Defaults to current model.
            
        Returns:
            LlamaCpp: Initialized model instance
        """
        model_key = model_key or self.current_model_key
        
        if model_key not in self.models:
            self.models[model_key] = self._setup_model(model_key)
        
        return self.models[model_key]

    def set_current_model(self, model_key: str):
        """
        Set the current model to use.
        
        Args:
            model_key (str): Key identifying the model to set as current
            
        Raises:
            ValueError: If the model_key is not recognized
        """
        if model_key not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_key}")
        self.current_model_key = model_key

    def get_or_create_vector_store(self, collection_id: str) -> Chroma:
        """
        Get or create a vector store for a specific collection.
        
        Args:
            collection_id (str): Identifier for the collection
            
        Returns:
            Chroma: Vector store instance
        """
        persist_directory = os.path.join(self.vector_store_path, str(collection_id))
        os.makedirs(persist_directory, exist_ok=True)
        
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

    def process_documents(self, collection) -> Optional[Chroma]:
        """
        Process documents for a collection and store in vector store.
        
        Args:
            collection: Collection containing documents to process
            
        Returns:
            Optional[Chroma]: Vector store instance if successful, None if failed
        """
        try:
            print(f"Processing documents for collection {collection.id}")
            vector_store = self.get_or_create_vector_store(collection.id)
            
            for document in collection.documents.all():
                print(f"Processing document: {document.title}")
                if not document.content:
                    continue
                    
                # Add document content with metadata
                vector_store.add_texts(
                    texts=[document.content],
                    metadatas=[{
                        "source": f"Document: {document.title}",
                        "document_id": str(document.id),
                        "collection_id": str(collection.id)
                    }]
                )
            
            return vector_store
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            return None
        
    def get_chat_history(self, session) -> List[Tuple[str, str]]:
        """
        Get formatted chat history from session.
        
        Args:
            session: Session containing chat messages
            
        Returns:
            List[Tuple[str, str]]: List of (human_message, ai_message) pairs
        """
        messages = session.messages.order_by('created_at')
        history = []
        
        # Pair user messages with assistant responses
        user_messages = messages.filter(is_user=True)
        assistant_messages = messages.filter(is_user=False)
        
        for user_msg, asst_msg in zip(user_messages, assistant_messages):
            history.append((user_msg.content, asst_msg.content))
            
        return history

    # [Previous code remains the same until the get_response method]

    def get_response(self, message: str, session, model_key: str = None) -> Optional[str]:
        """
        Get a response using the specified or current model with memory handling.
        
        Args:
            message (str): User's input message
            session: Session containing conversation context
            model_key (str, optional): Key identifying the model to use
            
        Returns:
            Optional[str]: Model's response or error message
        """
        try:
            print(f"Processing message: {message} for session {session.id}")
            model = self.get_model(model_key)
            
            # Initialize memory with chat history
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="question",
                output_key="answer",
                return_messages=True
            )
            
            # Load chat history into memory
            chat_history = self.get_chat_history(session)
            for human_msg, ai_msg in chat_history:
                memory.save_context(
                    {"question": human_msg},
                    {"answer": ai_msg}
                )
            
            # Handle direct responses if no documents
            if not session.collection.documents.exists():
                print("No documents found, generating direct response")
                # Combine system prompt, history, and current message
                full_prompt = f"{self.system_prompt}\n\nChat History:\n"
                for human_msg, ai_msg in chat_history:
                    full_prompt += f"Human: {human_msg}\nAssistant: {ai_msg}\n"
                full_prompt += f"Human: {message}\nAssistant:"
                
                response = model.invoke(full_prompt)
                memory.save_context({"question": message}, {"answer": response})
                return response

            # Process documents and create vector store
            vector_store = self.process_documents(session.collection)
            if not vector_store:
                return "I apologize, but I couldn't process the documents. Please try again."

            # Get number of documents in collection
            doc_count = session.collection.documents.count()
            print(f"Number of documents in collection: {doc_count}")

            # Create retriever with adaptive k parameter
            k = min(3, doc_count)  # Use either 3 or total doc count, whichever is smaller
            print(f"Setting retriever to fetch {k} documents")
            
            retriever = vector_store.as_retriever(
                search_kwargs={
                    "k": k
                }
            )
            
            # Create conversational chain with memory
            chain = ConversationalRetrievalChain.from_llm(
                llm=model,
                retriever=retriever,
                memory=memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={
                    "prompt": self.prompt
                },
                verbose=True
            )
            
            # Get response using the chain
            print("Generating response...")
            response = chain({
                "question": message,
                "chat_history": memory.chat_memory.messages
            })
            
            print("Response generated successfully")
            return response['answer']
            
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return f"I encountered an error: {str(e)}"


    def cleanup(self):
        """Clean up resources when shutting down."""
        for model in self.models.values():
            try:
                del model
            except Exception as e:
                print(f"Error cleaning up model: {e}")
        self.models.clear()

def test_model_setup(model_key: str) -> bool:
    """
    Test function to verify model setup and provide detailed feedback.
    
    Args:
        model_key (str): Key identifying the model to test
        
    Returns:
        bool: True if test successful, False otherwise
    """
    try:
        manager = get_chat_manager()
        model_config = manager.AVAILABLE_MODELS[model_key]
        model_path = manager._get_model_path(model_key)
        
        print("\nTesting model setup:")
        print(f"Model key: {model_key}")
        print(f"Display name: {model_config['display_name']}")
        print(f"Expected path: {model_path}")
        print(f"Path exists: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            print(f"File size: {os.path.getsize(model_path) / (1024*1024*1024):.2f} GB")
        
        print("\nTrying to initialize model...")
        model = manager.get_model(model_key)
        print("Model initialized successfully!")
        
        print("\nTesting inference...")
        response = model.invoke("Hello, can you hear me?")
        print(f"Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        return False

# Singleton instance with cleanup handling
_instance = None

def get_chat_manager() -> ChatManager:
    """
    Get or create the singleton ChatManager instance.
    
    Returns:
        ChatManager: Singleton instance of ChatManager
    """
    global _instance
    if _instance is None:
        print("Initializing chat manager...")
        _instance = ChatManager()
    return _instance

# Register cleanup on exit
import atexit

def cleanup_manager():
    """Clean up the ChatManager instance on program exit."""
    global _instance
    if _instance is not None:
        print("Cleaning up chat manager...")
        _instance.cleanup()
        _instance = None

atexit.register(cleanup_manager)