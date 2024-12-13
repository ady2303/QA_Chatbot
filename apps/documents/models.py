from django.db import models  # Import the models module
from apps.document_collections.models import Collection  # Ensure Collection is imported
import os  # For file path operations
import zipfile  # For processing zip files
import tempfile  # For creating temporary directories
from bs4 import BeautifulSoup  # For parsing HTML files

class Document(models.Model):
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE, related_name='documents')
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='documents/')
    content = models.TextField(blank=True)  # Extracted text content
    file_type = models.CharField(max_length=50)  # e.g., 'txt', 'html', 'zip'
    vector_embedding_id = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        if self.file:
            self.file_type = os.path.splitext(self.file.name)[1][1:].lower()

            if self.file_type == 'txt':
                self.content = self.file.read().decode('utf-8')
                self.file.seek(0)
            elif self.file_type == 'html':
                html_content = self.file.read().decode('utf-8')
                soup = BeautifulSoup(html_content, 'html.parser')
                self.content = soup.get_text()
                self.file.seek(0)
            elif self.file_type == 'zip':
                # Process zip files
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(self.file, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)

                        # Iterate through extracted files
                        extracted_content = []
                        for root, dirs, files in os.walk(temp_dir):
                            for file_name in files:
                                file_path = os.path.join(root, file_name)
                                file_extension = os.path.splitext(file_name)[1][1:].lower()
                                if file_extension == 'txt':
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        extracted_content.append(f.read())
                                elif file_extension == 'html':
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        html_content = f.read()
                                        soup = BeautifulSoup(html_content, 'html.parser')
                                        extracted_content.append(soup.get_text())

                        # Combine all extracted content into the content field
                        self.content = "\n\n".join(extracted_content)

        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        if self.file:
            if os.path.isfile(self.file.path):
                os.remove(self.file.path)
        super().delete(*args, **kwargs)

    class Meta:
        ordering = ['-created_at']
