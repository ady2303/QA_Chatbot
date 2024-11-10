# apps/document_collections/models.py
from django.db import models
from django.contrib.auth.models import User
import shutil
import os
from django.conf import settings

class Collection(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.name

    def delete(self, *args, **kwargs):
        # Delete associated vector store
        vector_store_path = os.path.join(settings.VECTOR_DB_PATH, "chroma", str(self.id))
        if os.path.exists(vector_store_path):
            shutil.rmtree(vector_store_path)
            
        # Delete document files
        for document in self.documents.all():
            if document.file:
                if os.path.isfile(document.file.path):
                    os.remove(document.file.path)
        
        # Super will delete all related objects due to CASCADE
        super().delete(*args, **kwargs)

    class Meta:
        ordering = ['-created_at']