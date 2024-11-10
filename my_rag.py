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
from termcolor import colored
import sys

class SimpleRAG:
    def __init__(self, vectorstore_path: str = "vectorstore/chroma/7", use_openai: bool = False):
        """Initialize the RAG system with existing vectorstore."""
        self.vectorstore_path = vectorstore_path
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Initialize LLM based on choice
        if use_openai:
            from langchain_openai import ChatOpenAI
            print("Initializing OpenAI model...")
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
            )
        else:
            print("Initializing local model...")
            self._setup_local_llm()
        
        # Setup vector store
        self.vectorstore = self._setup_vectorstore()
        
        # Setup prompt template
        self.prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context:

Context: {context}

Question: {question}

Answer:"""
        )

    def _setup_local_llm(self):
        """Set up the local language model."""
        # Custom callback handler for colored output
        class CustomCallbackHandler(StreamingStdOutCallbackHandler):
            def on_llm_new_token(self, token: str, **kwargs) -> None:
                colored_token = colored(token, 'blue')
                sys.stdout.write(colored_token)
                sys.stdout.flush()

        # Setup callback manager
        callback_manager = CallbackManager([CustomCallbackHandler()])

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
            n_ctx=2048,       # Reduced context window
            f16_kv=True,
            callbacks=callback_manager,
            verbose=False,
            temperature=0.7,
            top_p=0.95,
            max_tokens=512,
        )

    def _setup_vectorstore(self) -> Chroma:
        """Load the existing vector store."""
        if not os.path.exists(self.vectorstore_path):
            raise ValueError(f"Vectorstore not found at {self.vectorstore_path}")
            
        return Chroma(
            persist_directory=self.vectorstore_path,
            embedding_function=self.embeddings
        )

    def ask(self, question: str) -> Optional[str]:
        """Ask a question and get an answer based on the loaded documents."""
        try:
            # Create retriever
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            
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
            response = chain.invoke({"question": question})
            return response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return None

def main():
    os.environ["OPENAI_API_KEY"] = "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"

    # Get model choice
    while True:
        choice = input("Choose model (0 for OpenAI, 1 for local model): ")
        if choice in ['0', '1']:
            break
        print("Invalid choice. Please enter 0 or 1.")
    
    use_openai = (choice == '0')
    
    # Initialize RAG system with chosen model
    print(f"\nInitializing system with {'OpenAI' if use_openai else 'local'} model...")
    rag = SimpleRAG(use_openai=use_openai)
    
    # Main interaction loop
    print("\nReady to answer questions! (Type 'quit' to exit)")
    
    while True:
        question = input("\nQuestion: ")
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
            
        print("\nGenerating answer...")
        answer = rag.ask(question)
        
        if answer:
            print("\nAnswer:", answer)
        else:
            print("\nSorry, I couldn't generate an answer.")

if __name__ == "__main__":
    main()