from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class LLMManager:
    def __init__(self):
        # Initialize model and tokenizer
        model_name = "TheBloke/zephyr-7B-alpha-GGUF"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        
        # Create pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        # Create LangChain LLM
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

    def get_response(self, question, chat_history, context):
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        for q, a in chat_history:
            memory.save_context({"human_input": q}, {"output": a})
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=context,
            memory=memory,
            return_source_documents=True,
        )
        
        result = qa_chain({"question": question})
        return result["answer"]