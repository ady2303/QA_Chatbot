from core.llm.base import LLMManager
from core.vectorstore.chroma_store import ChromaDBManager
from apps.chat.models import ChatMessage

class ChatService:
    def __init__(self):
        self.llm_manager = LLMManager()
        self.vector_store = ChromaDBManager(self.llm_manager.embeddings)
    
    def get_chat_history(self, chat_session):
        messages = chat_session.messages.order_by('created_at')
        return [(msg.content, msg.response) for msg in messages if msg.is_user]
    
    def process_message(self, chat_session, message_content):
        # Get chat history
        chat_history = self.get_chat_history(chat_session)
        
        # Get relevant documents from the collection
        collection_docs = []
        for doc in chat_session.collection.documents.all():
            collection_docs.append(doc.content)
        
        # Add documents to vector store if not already added
        if collection_docs:
            self.vector_store.add_documents(collection_docs)
        
        # Get response from LLM
        retriever = self.vector_store.get_retriever()
        response = self.llm_manager.get_response(
            question=message_content,
            chat_history=chat_history,
            context=retriever
        )
        
        # Save message and response
        user_message = ChatMessage.objects.create(
            session=chat_session,
            content=message_content,
            is_user=True
        )
        
        bot_message = ChatMessage.objects.create(
            session=chat_session,
            content=response,
            is_user=False
        )
        
        return bot_message.content