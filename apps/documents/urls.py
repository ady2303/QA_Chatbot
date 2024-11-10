from django.urls import path
from . import views

app_name = 'documents'

urlpatterns = [
    path('upload/', views.document_upload, name='upload'),
    path('<int:pk>/', views.document_detail, name='detail'),
    path('<int:pk>/delete/', views.document_delete, name='delete'),
]