from typing import List, Dict


def format_mistral_messages(messages: List[Dict[str, str]]) -> str:
    """
    Formate une liste de messages pour le modèle Mistral.
    
    Le format Mistral utilise :
    - <s> au début de la conversation
    - [INST] message [/INST] pour les messages utilisateur
    - message</s> pour les réponses de l'assistant
    
    Args:
        messages: Liste de dictionnaires avec les clés 'role' et 'content'.
                 Les rôles peuvent être 'system', 'user', ou 'assistant'.
                 Les messages system sont généralement ignorés ou intégrés dans le premier [INST].
    
    Returns:
        Chaîne formatée selon le template Mistral.
    
    Example:
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi! How can I help?'}
        ]
        format_mistral_messages(messages)
        # Retourne:
        # <s> [INST] You are a helpful assistant.\n\nHello [/INST] Hi! How can I help?</s>
    """
    formatted = "<s>"
    system_prompt = None
    
    # Extraire le message system s'il existe
    for message in messages:
        if message.get('role') == 'system':
            system_prompt = message.get('content', '')
            break
    
    # Traiter les messages user et assistant
    i = 0
    while i < len(messages):
        message = messages[i]
        role = message.get('role', 'user')
        content = message.get('content', '')
        
        if role == 'system':
            # Le system prompt sera intégré dans le premier [INST]
            i += 1
            continue
        elif role == 'user':
            # Construire le message [INST]
            inst_content = content
            if system_prompt and i == 0:
                # Intégrer le system prompt dans le premier [INST]
                inst_content = f"{system_prompt}\n\n{content}"
                system_prompt = None  # Ne l'utiliser qu'une fois
            
            formatted += f" [INST] {inst_content} [/INST]"
            
            # Vérifier si le message suivant est une réponse de l'assistant
            if i + 1 < len(messages) and messages[i + 1].get('role') == 'assistant':
                assistant_content = messages[i + 1].get('content', '')
                formatted += f" {assistant_content}</s>"
                i += 2  # Passer les deux messages
            else:
                i += 1
        elif role == 'assistant':
            # Réponse de l'assistant (normalement déjà traité avec le user précédent)
            formatted += f" {content}</s>"
            i += 1
        else:
            i += 1
    
    return formatted

