# apps/documents/models.py
from django.db import models
from django.conf import settings
import os
from apps.document_collections.models import Collection

class Document(models.Model):
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE, related_name='documents')
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='documents/')
    content = models.TextField(blank=True)  # Extracted text content
    file_type = models.CharField(max_length=50)  # e.g., 'txt', 'pdf', 'html'
    vector_embedding_id = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        # Set file type based on extension
        if self.file:
            self.file_type = os.path.splitext(self.file.name)[1][1:].lower()
            
            # If it's a text file, read its content
            if self.file_type == 'txt':
                self.content = self.file.read().decode('utf-8')
                self.file.seek(0)  # Reset file pointer after reading
                
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        # Delete the file when document is deleted
        if self.file:
            if os.path.isfile(self.file.path):
                os.remove(self.file.path)
        super().delete(*args, **kwargs)

    class Meta:
        ordering = ['-created_at']