# apps/documents/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponseForbidden
from .models import Document
from apps.document_collections.models import Collection
import magic  # For file type detection

@login_required
def document_upload(request):
    collection_id = request.GET.get('collection')
    collection = get_object_or_404(Collection, pk=collection_id, created_by=request.user)
    
    if request.method == 'POST':
        title = request.POST.get('title')
        file = request.FILES.get('file')
        
        if title and file:
            # Validate file type (allow only text files for now)
            mime = magic.Magic(mime=True)
            file_type = mime.from_buffer(file.read(1024))
            file.seek(0)  # Reset file pointer after reading
            
            if not file_type.startswith('text/'):
                messages.error(request, 'Only text files are supported at this time.')
                return render(request, 'documents/upload.html', {'collection': collection})
            
            document = Document.objects.create(
                collection=collection,
                title=title,
                file=file
            )
            
            messages.success(request, 'Document uploaded successfully!')
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