from django.contrib import admin
from .models import ChatSession, ChatMessage

@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ('id', 'collection', 'user', 'created_at')
    list_filter = ('created_at', 'user')

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('session', 'is_user', 'content', 'created_at')
    list_filter = ('is_user', 'created_at', 'session')