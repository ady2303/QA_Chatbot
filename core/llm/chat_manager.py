# core/llm/chat_manager.py
import os
import shutil
from pathlib import Path
from typing import Optional, Dict
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
        "optimus": {
            "repo_id": "RichardErkhov/Q-bert-Optimus-7B-gguf",
            "filename": "Optimus-7B.Q4_K_M.gguf",
            "display_name": "Optimus 7B"
        },
        "rollama": {
            "repo_id": "RichardErkhov/OpenLLM-Ro-RoLlama2-7b-Chat-gguf",
            "filename": "RoLlama2-7b-Chat.IQ3_M.gguf",
            "display_name": "RoLlama2 7B"
        }
    }

    def __init__(self):
        """Initialize the RAG system."""
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Initialize models directory
        self.models_dir = Path(settings.LLM_MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        print(f"Models directory: {self.models_dir}")
        
        # Initialize models dictionary
        self.models: Dict[str, LlamaCpp] = {}
        self.current_model_key = "tinyllama"  # Default model
        
        # Initialize vector store path
        self.vector_store_path = os.path.join(settings.VECTOR_DB_PATH, "chroma")
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Setup prompt template
        self.prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context:

Context: {context}

Question: {question}

Answer:"""
        )

    def _get_model_path(self, model_key: str) -> Path:
        """Get the path where the model should be stored."""
        if model_key not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_key}")
        return self.models_dir / self.AVAILABLE_MODELS[model_key]["filename"]

    def _setup_model(self, model_key: str) -> LlamaCpp:
        """Set up a specific model."""
        try:
            model_config = self.AVAILABLE_MODELS[model_key]
            model_path = self._get_model_path(model_key)
            
            print(f"Setting up {model_config['display_name']}...")
            print(f"Checking for model at: {model_path}")
            
            # Check if model exists in our directory
            if not model_path.exists():
                print(f"Downloading {model_config['display_name']}...")
                # Download directly to our models directory
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
                n_gpu_layers=-1,
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
        """Get a specific model, initializing it if necessary."""
        model_key = model_key or self.current_model_key
        
        if model_key not in self.models:
            self.models[model_key] = self._setup_model(model_key)
        
        return self.models[model_key]

    def set_current_model(self, model_key: str):
        """Set the current model to use."""
        if model_key not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_key}")
        self.current_model_key = model_key

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

    def get_response(self, message: str, session, model_key: str = None) -> Optional[str]:
        """Get a response using the specified or current model."""
        try:
            print(f"Processing message: {message} for session {session.id}")
            print(f"Using model: {model_key if model_key else self.current_model_key}")
            
            # Get the appropriate model
            model = self.get_model(model_key)
            
            # Handle direct responses if no documents
            if not session.collection.documents.exists():
                print("No documents found, generating direct response")
                return model.invoke(message)

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
                | model
            )
            
            # Get response
            print("Generating response...")
            response = chain.invoke({"question": message})
            print("Response generated successfully")
            
            return response
            
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

def test_model_setup(model_key):
    """Test function to verify model setup and provide detailed feedback."""
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

def get_chat_manager():
    global _instance
    if _instance is None:
        print("Initializing chat manager...")
        _instance = ChatManager()
    return _instance

# Register cleanup on exit
import atexit

def cleanup_manager():
    global _instance
    if _instance is not None:
        print("Cleaning up chat manager...")
        _instance.cleanup()
        _instance = None

atexit.register(cleanup_manager)