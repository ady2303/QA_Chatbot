import os
from typing import Optional, Dict, List
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
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
import shutil

class SimpleRAG:
    # Define available local models
    LOCAL_MODELS = {
        "tinyllama": {
            "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "display_name": "TinyLlama 1.1B",
            "color": "blue"
        },
        "capybara": {
            "repo_id": "TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF",
            "filename": "capybarahermes-2.5-mistral-7b.Q4_K_M.gguf",
            "display_name": "Capybara Hermes",
            "color": "green"
        }
    }

    def __init__(self, vectorstore_path: str = "vectorstore/chroma/22", use_openai: bool = False, 
                 local_model_keys: List[str] = None, models_dir: str = None):
        """Initialize the RAG system with existing vectorstore."""
        self.vectorstore_path = vectorstore_path
        self.use_openai = use_openai
        self.local_model_keys = local_model_keys or []
        self.models = {}  # Dictionary to store multiple models
        
        # Set up models directory on external drive
        self.models_dir = self._setup_models_directory(models_dir)
        
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
            for model_key in self.local_model_keys:
                print(f"Initializing {self.LOCAL_MODELS[model_key]['display_name']}...")
                self.models[model_key] = self._setup_local_llm(model_key)
        
        # Setup vector store
        self.vectorstore = self._setup_vectorstore()
        
        # Setup prompt template
        self.prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context:

Context: {context}

Question: {question}

Answer:"""
        )

    def _setup_models_directory(self, base_dir: Optional[str] = None) -> Path:
        """Set up the models directory on the external drive."""
        if base_dir is None:
            base_dir = "/Volumes/Crucial X9/huggingface_llms"
        
        models_dir = Path(base_dir)
        
        try:
            # Create directory if it doesn't exist
            models_dir.mkdir(parents=True, exist_ok=True)
            print(f"Models directory set up at: {models_dir}")
        except Exception as e:
            print(f"Error setting up models directory: {e}")
            print("Please ensure your external drive is connected and mounted.")
            sys.exit(1)
        
        return models_dir

    def _setup_local_llm(self, model_key: str):
        """Set up a local language model with storage on external drive."""
        # Custom callback handler for colored output
        class CustomCallbackHandler(StreamingStdOutCallbackHandler):
            def __init__(self, color):
                self.color = color
                
            def on_llm_new_token(self, token: str, **kwargs) -> None:
                colored_token = colored(token, self.color)
                sys.stdout.write(colored_token)
                sys.stdout.flush()

        # Setup callback manager with model-specific color
        callback_manager = CallbackManager([
            CustomCallbackHandler(self.LOCAL_MODELS[model_key]['color'])
        ])

        # Get model configuration
        model_config = self.LOCAL_MODELS[model_key]
        
        # Define model path on external drive
        model_path = self.models_dir / model_config["filename"]
        
        # Download model if it doesn't exist
        if not model_path.exists():
            print(f"Downloading {model_config['display_name']} to external drive...")
            temp_path = hf_hub_download(
                repo_id=model_config["repo_id"],
                filename=model_config["filename"]
            )
            # Copy to external drive
            shutil.copy2(temp_path, model_path)
            print(f"Model saved to {model_path}")
        else:
            print(f"Using existing model at {model_path}")
        
        print(f"Initializing {model_config['display_name']}...")
        return LlamaCpp(
            model_path=str(model_path),  # Convert Path to string
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
        """Ask a question and get answer(s) based on the loaded documents."""
        try:
            # Create retriever
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            
            if self.use_openai:
                # Single model case (OpenAI)
                chain = (
                    {
                        "context": itemgetter("question") | retriever,
                        "question": itemgetter("question")
                    }
                    | self.prompt
                    | self.llm
                )
                return chain.invoke({"question": question})
            else:
                # Multiple models case
                responses = {}
                for model_key, model in self.models.items():
                    print(f"\n{self.LOCAL_MODELS[model_key]['display_name']} response:")
                    chain = (
                        {
                            "context": itemgetter("question") | retriever,
                            "question": itemgetter("question")
                        }
                        | self.prompt
                        | model
                    )
                    responses[model_key] = chain.invoke({"question": question})
                return responses
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return None

def choose_models() -> tuple[bool, List[str]]:
    """Helper function to handle model selection."""
    while True:
        print("\nAvailable options:")
        print("0. Use all local models")
        print("1. TinyLlama 1.1B")
        print("2. Capybara Hermes")
        print("3. OpenAI GPT-3.5")
        
        choice = input("\nChoose model (0-3): ")
        
        if choice == '0':
            return False, list(SimpleRAG.LOCAL_MODELS.keys())
        elif choice == '1':
            return False, ["tinyllama"]
        elif choice == '2':
            return False, ["capybara"]
        elif choice == '3':
            return True, []
        
        print("Invalid choice. Please try again.")

def verify_external_drive():
    """Verify that the external drive is connected and mounted."""
    drive_path = Path("/Volumes/Crucial X9")
    if not drive_path.exists():
        print("Error: Crucial X9 drive not found!")
        print("Please ensure your external drive is connected and mounted.")
        return False
    return True

def main():
    # Check for external drive
    if not verify_external_drive():
        sys.exit(1)

    os.environ["OPENAI_API_KEY"] = "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"

    # Get model choice
    use_openai, model_keys = choose_models()
    
    # Specify the external drive path
    external_drive_path = "/Volumes/Crucial X9/huggingface_llms"
    
    # Initialize RAG system with chosen model(s) and external drive path
    if use_openai:
        print("\nInitializing system with OpenAI...")
    elif len(model_keys) > 1:
        print("\nInitializing system with all local models...")
    else:
        print(f"\nInitializing system with {SimpleRAG.LOCAL_MODELS[model_keys[0]]['display_name']}...")
    
    try:
        rag = SimpleRAG(
            use_openai=use_openai, 
            local_model_keys=model_keys,
            models_dir=external_drive_path
        )
        
        # Main interaction loop
        print("\nReady to answer questions! (Type 'quit' to exit)")
        
        while True:
            question = input("\nQuestion: ")
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
                
            print("\nGenerating answer...")
            answer = rag.ask(question)
            
            if answer:
                if not isinstance(answer, dict):
                    print("\nAnswer:", answer)
                # For multiple models, responses are already printed during generation
            else:
                print("\nSorry, I couldn't generate an answer.")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please ensure your external drive is connected and has sufficient space.")

if __name__ == "__main__":
    main()