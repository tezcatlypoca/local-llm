"""
Exemple d'utilisation du module de gestion de conversations.

Ce script montre comment utiliser ConversationManager pour :
- Créer une conversation
- Envoyer des messages avec gestion automatique du contexte
- Sauvegarder et reprendre des conversations
"""
from src.client import LLMClient
from src.conversation import ConversationManager


def exemple_basique():
    """Exemple basique d'utilisation."""
    
    # Initialiser le client API
    client = LLMClient(base_url="http://localhost:5000")
    
    # Initialiser le gestionnaire de conversations
    manager = ConversationManager(client, auto_save=True)
    
    # Charger un modèle sur un GPU (exemple avec TinyLlama)
    print("Chargement du modèle...")
    result = client.models.load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    gpu_id = result["gpu_id"]
    model_name = result["model_name"]
    print(f"Modèle chargé sur GPU {gpu_id}")
    
    # Créer une nouvelle conversation
    conv_id = manager.create_conversation(
        model_name=model_name,
        gpu_id=gpu_id,
        system_prompt="Tu es un assistant IA utile et bienveillant.",
        max_messages=10  # Garder les 10 derniers messages
    )
    print(f"Conversation créée: {conv_id}")
    
    # Envoyer des messages
    print("\n--- Envoi de messages ---")
    
    response1 = manager.send_message(
        conversation_id=conv_id,
        user_message="Bonjour !"
    )
    print(f"User: Bonjour !")
    print(f"Assistant: {response1['response']}")
    
    response2 = manager.send_message(
        conversation_id=conv_id,
        user_message="Qu'est-ce que l'intelligence artificielle ?"
    )
    print(f"\nUser: Qu'est-ce que l'intelligence artificielle ?")
    print(f"Assistant: {response2['response']}")
    
    # Récupérer le contexte
    context = manager.get_conversation(conv_id)
    print(f"\nNombre de messages dans le contexte: {context.get_message_count()}")
    
    # Nettoyer
    client.models.unload_model(gpu_id=gpu_id, access_token=result["access_token"])


def exemple_multiple_modeles():
    """Exemple avec différents modèles."""
    
    client = LLMClient(base_url="http://localhost:5000")
    manager = ConversationManager(client)
    
    # Exemple avec TinyLlama
    print("\n=== TinyLlama ===")
    result_tiny = client.models.load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    conv_tiny = manager.create_conversation(
        model_name=result_tiny["model_name"],
        gpu_id=result_tiny["gpu_id"],
        max_messages=20
    )
    
    response = manager.send_message(conv_tiny, "Explique-moi Python en une phrase.")
    print(f"TinyLlama: {response['response']}")
    
    # Exemple avec Qwen2.5
    print("\n=== Qwen2.5 ===")
    result_qwen = client.models.load_model("Qwen/Qwen2.5-7B-Instruct")
    conv_qwen = manager.create_conversation(
        model_name=result_qwen["model_name"],
        gpu_id=result_qwen["gpu_id"],
        max_messages=20
    )
    
    response = manager.send_message(conv_qwen, "Explique-moi Python en une phrase.")
    print(f"Qwen2.5: {response['response']}")
    
    # Nettoyer
    client.models.unload_model(gpu_id=result_tiny["gpu_id"], access_token=result_tiny["access_token"])
    client.models.unload_model(gpu_id=result_qwen["gpu_id"], access_token=result_qwen["access_token"])


def exemple_reprise_conversation():
    """Exemple de reprise d'une conversation sauvegardée."""
    
    client = LLMClient(base_url="http://localhost:5000")
    manager = ConversationManager(client, auto_save=True)
    
    # Lister les conversations existantes
    conversations = manager.list_conversations()
    print(f"Conversations sauvegardées: {len(conversations)}")
    
    if conversations:
        # Reprendre la dernière conversation
        conv_id = conversations[0]
        print(f"\nReprise de la conversation: {conv_id}")
        
        context = manager.get_conversation(conv_id)
        print(f"Modèle: {context.model_name}")
        print(f"Nombre de messages: {context.get_message_count()}")
        
        # Continuer la conversation
        response = manager.send_message(
            conversation_id=conv_id,
            user_message="Peux-tu résumer notre conversation précédente ?"
        )
        print(f"\nAssistant: {response['response']}")


if __name__ == "__main__":
    # Décommenter l'exemple souhaité
    # exemple_basique()
    # exemple_multiple_modeles()
    # exemple_reprise_conversation()
    pass

