from django.contrib import admin
from .models import Document

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'collection', 'file_type', 'created_at')
    search_fields = ('title', 'content')
    list_filter = ('file_type', 'collection', 'created_at')