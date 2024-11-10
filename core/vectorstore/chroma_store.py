from chromadb.config import Settings
from langchain.vectorstores import Chroma
from django.conf import settings
import os

class ChromaDBManager:
    def __init__(self, embedding_function):
        self.embeddings = embedding_function
        self.persist_directory = settings.VECTOR_DB_PATH
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.vectordb = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def add_documents(self, texts, metadatas=None):
        return self.vectordb.add_texts(texts=texts, metadatas=metadatas)

    def get_retriever(self):
        return self.vectordb.as_retriever(
            search_kwargs={"k": 3}
        )

    def similarity_search(self, query, k=3):
        return self.vectordb.similarity_search(query, k=k)