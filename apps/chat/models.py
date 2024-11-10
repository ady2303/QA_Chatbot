# apps/chat/models.py
from django.db import models
from django.contrib.auth.models import User
from apps.document_collections.models import Collection

class ChatSession(models.Model):
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE, related_name='chat_sessions')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Chat session {self.id} - {self.collection.name}"

class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    content = models.TextField()
    is_user = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['created_at']
    
    def __str__(self):
        return f"{'User' if self.is_user else 'Bot'}: {self.content[:50]}"