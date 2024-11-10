from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import ensure_csrf_cookie
from .models import ChatSession, ChatMessage
from apps.document_collections.models import Collection
from core.llm.chat_manager import get_chat_manager
import logging

logger = logging.getLogger(__name__)

@login_required
@ensure_csrf_cookie
def chat_session(request, collection_id):
    collection = get_object_or_404(Collection, pk=collection_id, created_by=request.user)
    session, created = ChatSession.objects.get_or_create(
        user=request.user,
        collection=collection
    )
    return render(request, 'chat/session.html', {
        'session': session,
        'collection': collection
    })



@login_required
@require_http_methods(["POST"])
def send_message(request):
    try:
        message = request.POST.get('message')
        session_id = request.POST.get('session_id')
        
        logger.info(f"Received message request - Message: {message}, Session ID: {session_id}")
        
        if not message or not session_id:
            return JsonResponse({
                'status': 'error',
                'error': f'Missing parameters. Message: {message}, Session ID: {session_id}'
            }, status=400)
        
        session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        logger.info(f"Found session {session_id} for user {request.user.username}")
        
        # Create user message
        user_message = ChatMessage.objects.create(
            session=session,
            content=message,
            is_user=True
        )
        logger.info(f"Created user message with ID {user_message.id}")
        
        try:
            # Get chat manager instance
            logger.info("Getting chat manager instance")
            manager = get_chat_manager()
            
            # Get response from LLM
            logger.info("Requesting response from LLM")
            response_text = manager.get_response(message, session)
            
            logger.info(f"Received response: {response_text[:100]}...")
            
            if not response_text:
                response_text = "I apologize, but I couldn't generate a response. Please try again."
                logger.warning("Empty response received from LLM")
            
            # Create bot message
            bot_message = ChatMessage.objects.create(
                session=session,
                content=response_text,
                is_user=False
            )
            logger.info(f"Created bot message with ID {bot_message.id}")
            
            return JsonResponse({
                'status': 'success',
                'response': response_text
            })
            
        except Exception as e:
            import traceback
            logger.error(f"Error getting response from LLM: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return JsonResponse({
                'status': 'error',
                'error': f'LLM Error: {str(e)}'
            }, status=500)
        
    except Exception as e:
        import traceback
        logger.error(f"Error in send_message: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)
        
  


@login_required
def chat_history(request, session_id):
    session = get_object_or_404(ChatSession, pk=session_id, user=request.user)
    messages = session.messages.all().order_by('created_at')
    return render(request, 'chat/history.html', {'messages': messages})