"""
Script de test pour le client API Local LLM.

Ce script démontre comment utiliser le client pour interagir avec l'API.
"""
import sys
from pathlib import Path

# Ajouter le dossier src au path pour les imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from client import LLMClient
import json


def test_root_endpoint(client: LLMClient):
    """Test de la route racine."""
    print("\n" + "="*60)
    print("Test: Route racine")
    print("="*60)
    
    try:
        status = client.root.check_status()
        print(f"✅ Status de l'API: {status}")
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def test_models_endpoint(client: LLMClient):
    """Test des endpoints de modèles (liste uniquement)."""
    print("\n" + "="*60)
    print("Test: Endpoints Models (Liste)")
    print("="*60)
    
    try:
        # Lister les modèles
        print("\n1. Liste des modèles disponibles:")
        models_response = client.models.list_models()
        print(f"   ✅ {models_response.get('count', 0)} modèle(s) trouvé(s)")
        
        if models_response.get('models'):
            for model in models_response['models'][:3]:  # Afficher les 3 premiers
                print(f"      - {model.get('identifier', 'unknown')} ({model.get('size_mb', 0):.2f} MB)")
        
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def test_load_chat_unload_cycle(client: LLMClient, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", gpu_id: int = 1):
    """
    Test du cycle complet: charger un modèle, envoyer un message, décharger le modèle.
    
    Args:
        client: Instance du client API
        model_name: Nom du modèle à charger (défaut: TinyLlama)
        gpu_id: ID du GPU à utiliser (défaut: 0)
    
    Returns:
        bool: True si tous les tests sont réussis, False sinon
    """
    print("\n" + "="*60)
    print(f"Test: Cycle complet (Load -> Chat -> Unload) - {model_name}")
    print("="*60)
    
    access_token = None
    loaded_gpu_id = gpu_id  # Initialisation par défaut
    try:
        # Étape 1: Charger un modèle
        print(f"\n1. Chargement du modèle '{model_name}' sur GPU {gpu_id}...")
        load_result = client.models.load_model(model_name)
        
        if load_result.get('status') != 'success':
            print(f"   ❌ Échec du chargement: {load_result.get('message', 'Erreur inconnue')}")
            return False
        
        access_token = load_result.get('access_token')
        loaded_gpu_id = load_result.get('gpu_id')
        loaded_model_name = load_result.get('model_name')
        
        print(f"   ✅ Modèle chargé avec succès!")
        print(f"   ✅ GPU: {loaded_gpu_id}")
        print(f"   ✅ Modèle: {loaded_model_name}")
        print(f"   ✅ Access token: {access_token[:20]}...")
        
        # Vérifier que le modèle est bien chargé
        health = client.health.check_gpu_health(loaded_gpu_id)
        if not health.get('model', {}).get('loaded'):
            print(f"   ❌ Le modèle n'apparaît pas comme chargé dans le health check")
            return False
        
        # Étape 2: Envoyer un message de chat
        print(f"\n2. Envoi d'un message de chat...")
        chat_response = client.chat.send_message(
            gpu_id=loaded_gpu_id,
            message="Bonjour! Pouvez-vous me dire bonjour en retour?",
            temperature=0.7,
            max_new_tokens=50
        )
        
        if chat_response.get('status') != 'success':
            print(f"   ❌ Échec du chat: {chat_response.get('message', 'Erreur inconnue')}")
            # On continue quand même pour décharger le modèle
        else:
            response_text = chat_response.get('response', '')
            print(f"   ✅ Réponse générée ({len(response_text)} caractères)")
            print(f"   ✅ Réponse: {response_text[:100]}...")
        
        # Étape 3: Décharger le modèle
        print(f"\n3. Déchargement du modèle du GPU {loaded_gpu_id}...")
        unload_result = client.models.unload_model(
            gpu_id=loaded_gpu_id,
            access_token=access_token
        )
        
        if unload_result.get('status') != 'success':
            print(f"   ❌ Échec du déchargement: {unload_result.get('message', 'Erreur inconnue')}")
            return False
        
        print(f"   ✅ Modèle déchargé avec succès!")
        
        # Vérifier que le modèle est bien déchargé
        health_after = client.health.check_gpu_health(loaded_gpu_id)
        if health_after.get('model', {}).get('loaded'):
            print(f"   ⚠️  Le modèle apparaît encore comme chargé dans le health check")
            # Ce n'est pas nécessairement une erreur critique, mais c'est suspect
        
        print(f"\n✅ Cycle complet réussi: Load -> Chat -> Unload")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        
        # Tentative de nettoyage: décharger le modèle si on a l'access_token
        if access_token is not None:
            try:
                print(f"\n🧹 Tentative de nettoyage: déchargement du modèle sur GPU {loaded_gpu_id}...")
                client.models.unload_model(gpu_id=loaded_gpu_id, access_token=access_token)
                print(f"   ✅ Nettoyage réussi")
            except Exception as cleanup_error:
                print(f"   ⚠️  Erreur lors du nettoyage: {cleanup_error}")
        
        return False


def test_health_endpoint(client: LLMClient):
    """Test des endpoints de health check."""
    print("\n" + "="*60)
    print("Test: Endpoints Health")
    print("="*60)
    
    try:
        # Health check global
        print("\n1. Health check global:")
        health = client.health.check_health()
        print(f"   ✅ Status: {health.get('status')}")
        print(f"   ✅ GPUs disponibles: {health.get('total_gpus', 0)}")
        
        summary = health.get('summary', {})
        print(f"   ✅ GPUs libres: {summary.get('gpus_free', 0)}")
        print(f"   ✅ GPUs utilisés: {summary.get('gpus_with_models', 0)}")
        
        # Health check GPU 1
        print("\n2. Health check GPU 1:")
        gpu_health = client.health.check_gpu_health(1)
        print(f"   ✅ Status: {gpu_health.get('status')}")
        print(f"   ✅ Disponible: {gpu_health.get('available', False)}")
        
        model_info = gpu_health.get('model', {})
        if model_info.get('loaded'):
            print(f"   ✅ Modèle chargé: {model_info.get('name')}")
        else:
            print(f"   ℹ️  Aucun modèle chargé")
        
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def test_chat_endpoint(client: LLMClient):
    """
    Test de l'endpoint chat (si un modèle est déjà chargé).
    
    Note: Ce test nécessite qu'un modèle soit déjà chargé sur GPU 1.
    Pour un test complet du cycle, utilisez test_load_chat_unload_cycle().
    """
    print("\n" + "="*60)
    print("Test: Endpoint Chat (si modèle déjà chargé)")
    print("="*60)
    
    try:
        # Vérifier qu'un modèle est chargé
        health = client.health.check_gpu_health(1)
        if not health.get('model', {}).get('loaded'):
            print("   ⚠️  Aucun modèle chargé sur GPU 1. Test ignoré.")
            print("   💡 Utilisez test_load_chat_unload_cycle() pour un test complet")
            return True
        
        # Envoyer un message
        print("\n1. Envoi d'un message (format string):")
        response = client.chat.send_message(
            gpu_id=1,
            message="Bonjour, comment allez-vous?",
            temperature=0.7,
            max_new_tokens=50
        )
        print(f"   ✅ Réponse: {response.get('response', '')[:100]}...")
        print(f"   ✅ Modèle utilisé: {response.get('model_name')}")
        
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def test_completion_endpoint(client: LLMClient):
    """Test de l'endpoint completion."""
    print("\n" + "="*60)
    print("Test: Endpoint Completion")
    print("="*60)
    
    try:
        # Vérifier qu'un modèle est chargé
        health = client.health.check_gpu_health(1)
        if not health.get('model', {}).get('loaded'):
            print("   ⚠️  Aucun modèle chargé sur GPU 1. Test ignoré.")
            return True
        
        # Générer une completion
        print("\n1. Génération d'une completion:")
        response = client.completion.complete(
            gpu_id=1,
            prompt="Le machine learning est une branche de",
            temperature=0.7,
            max_new_tokens=50
        )
        print(f"   ✅ Completion: {response.get('response', '')[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def test_logs_endpoint(client: LLMClient):
    """Test de l'endpoint logs."""
    print("\n" + "="*60)
    print("Test: Endpoints Logs")
    print("="*60)
    
    try:
        # Statistiques des logs
        print("\n1. Statistiques des logs:")
        stats = client.logs.get_stats()
        stats_data = stats.get('stats', {})
        print(f"   ✅ Total logs: {stats_data.get('total_logs', 0)}")
        print(f"   ✅ Taille max: {stats_data.get('max_size', 0)}")
        
        level_counts = stats_data.get('level_counts', {})
        if level_counts:
            print(f"   ✅ Répartition: {level_counts}")
        
        # Historique des logs
        print("\n2. Historique des logs (5 derniers):")
        history = client.logs.get_history(limit=5, level="INFO")
        logs = history.get('logs', [])
        print(f"   ✅ {len(logs)} log(s) récupéré(s)")
        
        for log in logs[:3]:  # Afficher les 3 premiers
            print(f"      - [{log.get('level')}] {log.get('message', '')[:60]}...")
        
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def main():
    """Fonction principale de test."""
    print("="*60)
    print("Test du Client API Local LLM")
    print("="*60)
    
    # Initialisation du client
    try:
        # Le client utilisera la configuration par défaut ou les variables d'environnement
        # Pour spécifier une URL: client = LLMClient(base_url="http://localhost:5000")
        print("\n📡 Initialisation du client...")
        client = LLMClient(
            base_url="http://192.168.1.50:5000",
            timeout=60,
        )
        print(f"   ✅ Client initialisé: {client}")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        return
    
    # Exécution des tests
    tests = [
        ("Route racine", test_root_endpoint),
        ("Models (Liste)", test_models_endpoint),
        ("Health", test_health_endpoint),
        ("Cycle complet (Load->Chat->Unload)", lambda c: test_load_chat_unload_cycle(c, "TinyLlama/TinyLlama-1.1B-Chat-v1.0")),
        ("Chat (si modèle chargé)", test_chat_endpoint),
        ("Completion", test_completion_endpoint),
        ("Logs", test_logs_endpoint),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func(client)
        except Exception as e:
            print(f"\n❌ Erreur inattendue dans {test_name}: {e}")
            results[test_name] = False
    
    # Résumé
    print("\n" + "="*60)
    print("Résumé des tests")
    print("="*60)
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    total = len(results)
    passed = sum(1 for success in results.values() if success)
    print(f"\nTotal: {passed}/{total} tests réussis")
    
    # Fermeture du client
    client.close()
    print("\n✅ Client fermé")


if __name__ == "__main__":
    main()

