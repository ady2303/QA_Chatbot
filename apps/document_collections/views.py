# apps/document_collections/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Count
from .models import Collection
from django.urls import reverse
from django.http import HttpResponseForbidden

@login_required
def collection_list(request):
    collections = Collection.objects.filter(created_by=request.user).annotate(
        document_count=Count('documents'),
        chat_count=Count('chat_sessions')
    )
    return render(request, 'document_collections/list.html', {'collections': collections})

@login_required
def collection_create(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description')
        if name:
            Collection.objects.create(
                name=name,
                description=description,
                created_by=request.user
            )
            messages.success(request, 'Collection created successfully!')
            return redirect('document_collections:list')
    return render(request, 'document_collections/create.html')

@login_required
def collection_detail(request, pk):
    collection = get_object_or_404(Collection, pk=pk)
    if collection.created_by != request.user:
        return HttpResponseForbidden("You don't have permission to view this collection.")
    return render(request, 'document_collections/detail.html', {'collection': collection})

@login_required
def collection_delete(request, pk):
    collection = get_object_or_404(Collection, pk=pk)
    
    # Check if user owns the collection
    if collection.created_by != request.user:
        return HttpResponseForbidden("You don't have permission to delete this collection.")
    
    if request.method == 'POST':
        try:
            collection.delete()
            messages.success(request, f'Collection "{collection.name}" has been deleted.')
            return redirect('document_collections:list')
        except Exception as e:
            messages.error(request, f'Error deleting collection: {str(e)}')
            return redirect('document_collections:detail', pk=pk)
    
    return render(request, 'document_collections/delete.html', {'collection': collection})