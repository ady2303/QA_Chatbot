from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponseForbidden
from .models import Document
from apps.document_collections.models import Collection
import magic  # For file type detection
import tempfile  # Import tempfile module
import zipfile  # For handling zip files
from bs4 import BeautifulSoup  # For parsing HTML files
import os


@login_required
def document_upload(request):
    collection_id = request.GET.get('collection')
    collection = get_object_or_404(Collection, pk=collection_id, created_by=request.user)
    
    if request.method == 'POST':
        title = request.POST.get('title')
        file = request.FILES.get('file')
        
        if title and file:
            # Validate file type (allow text, HTML, and zip files)
            mime = magic.Magic(mime=True)
            file_type = mime.from_buffer(file.read(1024))
            file.seek(0)  # Reset file pointer after reading
            
            if file_type.startswith('text/') or file_type == 'text/html':
                document = Document.objects.create(
                    collection=collection,
                    title=title,
                    file=file
                )
                messages.success(request, 'Document uploaded successfully!')

            elif file_type == 'application/zip':
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(file, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)

                        for root, dirs, files in os.walk(temp_dir):
                            for file_name in files:
                                file_path = os.path.join(root, file_name)
                                file_extension = os.path.splitext(file_name)[1][1:].lower()

                                if file_extension in ['txt', 'html']:
                                    with open(file_path, 'r', encoding='utf-8') as extracted_file:
                                        content = extracted_file.read()
                                        if file_extension == 'html':
                                            soup = BeautifulSoup(content, 'html.parser')
                                            content = soup.get_text()

                                        Document.objects.create(
                                            collection=collection,
                                            title=f"{title} - {file_name}",
                                            file=None,
                                            content=content,
                                            file_type=file_extension
                                        )
                messages.success(request, 'Zip file processed and documents created successfully!')
            else:
                messages.error(request, 'Unsupported file type. Only text, HTML, and zip files are allowed.')
                return render(request, 'documents/upload.html', {'collection': collection})
            
            return redirect('document_collections:detail', pk=collection.pk)
            
    return render(request, 'documents/upload.html', {'collection': collection})




@login_required
def document_detail(request, pk):
    document = get_object_or_404(Document, pk=pk)
    # Check if user has access to this document
    if document.collection.created_by != request.user:
        return HttpResponseForbidden("You don't have permission to view this document.")
    
    return render(request, 'documents/detail.html', {'document': document})


@login_required
def document_delete(request, pk):
    document = get_object_or_404(Document, pk=pk)
    # Check if user has access to this document
    if document.collection.created_by != request.user:
        return HttpResponseForbidden("You don't have permission to delete this document.")
    
    collection_id = document.collection.pk
    if request.method == 'POST':
        document.delete()
        messages.success(request, 'Document deleted successfully!')
        return redirect('document_collections:detail', pk=collection_id)
        
    return render(request, 'documents/delete.html', {'document': document})
