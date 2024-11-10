from django.urls import path
from . import views

app_name = 'chat'

urlpatterns = [
    path('session/<int:collection_id>/', views.chat_session, name='session'),
    path('message/send/', views.send_message, name='send_message'),
    path('history/<int:session_id>/', views.chat_history, name='history'),
]