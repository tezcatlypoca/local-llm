from typing import List, Dict


def format_qwen_messages(messages: List[Dict[str, str]]) -> str:
    """
    Formate une liste de messages pour le modèle Qwen.
    
    Le format Qwen utilise le format ChatML avec les tokens :
    - <|im_start|>role\ncontent<|im_end|>
    
    Args:
        messages: Liste de dictionnaires avec les clés 'role' et 'content'.
                 Les rôles peuvent être 'system', 'user', ou 'assistant'.
    
    Returns:
        Chaîne formatée selon le template Qwen.
    
    Example:
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi! How can I help?'}
        ]
        format_qwen_messages(messages)
        # Retourne:
        # <|im_start|>system
        # You are a helpful assistant.<|im_end|>
        # <|im_start|>user
        # Hello<|im_end|>
        # <|im_start|>assistant
        # Hi! How can I help?<|im_end|>
    """
    formatted_parts = []
    
    for message in messages:
        role = message.get('role', 'user')
        content = message.get('content', '')
        formatted_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    
    return "\n".join(formatted_parts)

