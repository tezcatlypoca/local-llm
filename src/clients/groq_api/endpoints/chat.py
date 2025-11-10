from typing import List, Dict, Optional

def send_messages(
    groq_client, 
    messages: List[Dict[str, str]],  # ✅ Liste de dictionnaires : Dict[str, str] avec crochets
    model_name: str, 
    temperature: Optional[float] = None,  # Valeur par défaut None
    is_stream: Optional[bool] = None  # Valeur par défaut None
):
    """
    Envoie une liste de messages à l'API Groq.
    
    Args:
        groq_client: Instance du client Groq
        messages: Liste de dictionnaires avec les clés "role" et "content" (tous des strings)
        model_name: Nom du modèle à utiliser
        temperature: Température pour la génération (défaut: 0.7)
        is_stream: Activer le streaming (défaut: False)
    
    Returns:
        Contenu de la réponse de l'assistant
    """
    # Préparer les paramètres avec valeurs par défaut
    params = {
        "messages": messages,
        "model": model_name,
        "temperature": temperature or 0.7,  # ✅ Utiliser 'or' ou condition ternaire
        "stream": is_stream or False  # ✅ Utiliser 'or' ou condition ternaire
    }
    
    res = groq_client.chat.completions.create(**params)
    
    return res.choices[0].message.content
