from django.urls import path
from . import views

app_name = 'document_collections'

urlpatterns = [
    path('', views.collection_list, name='list'),
    path('create/', views.collection_create, name='create'),
    path('<int:pk>/', views.collection_detail, name='detail'),
    # path('<int:pk>/update/', views.collection_update, name='update'),
    path('<int:pk>/delete/', views.collection_delete, name='delete'),
    path('signup/', views.signup, name='signup'), 
]