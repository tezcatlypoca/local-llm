# Réponses aux questions sur l'implémentation

## 1. Gestion du gpu_id et access_token lors du chargement de modèle

**Problème actuel** : Actuellement, quand on charge un modèle via `/models/load/<model_name>`, on retourne le résultat de l'API de base (qui contient `gpu_id` et `access_token`), mais ces informations ne sont pas stockées côté surcouche.

**Solution proposée** : 
- Option A : Stocker les modèles chargés dans un dictionnaire en mémoire (simple mais perdu au redémarrage)
- Option B : Demander à l'utilisateur de fournir le `gpu_id` lors de la création de conversation (il doit gérer le chargement lui-même)
- Option C : Créer un système de tracking des modèles chargés avec persistance

**Recommandation** : Option B pour l'instant (plus simple), mais on peut ajouter Option A pour faciliter l'utilisation.

**Actuellement** : La route `/conversations` POST attend un `gpu_id` dans le body. L'utilisateur doit :
1. Charger le modèle via `/models/load/<model_name>` → récupère `gpu_id` et `access_token`
2. Créer la conversation en fournissant le `gpu_id`
3. Conserver l'`access_token` pour décharger le modèle plus tard

## 2. Templates automatiques

**Oui, les templates sont automatiquement utilisés !**

Dans `ConversationManager.send_message()` (ligne 138-147) :
```python
# 3. Obtenir le template
template = TemplateRegistry.get_template(context.model_name)

# 4. Obtenir les messages formatés
messages_to_format = context.get_messages_for_formatting()

# 5. Formater le contexte selon le template
formatted_prompt = template.format_conversation(
    messages_to_format,
    context.system_prompt
)
```

Le template est automatiquement sélectionné selon le `model_name` de la conversation, et le formatage est appliqué avant l'envoi à l'API de base.

## 3. Persistance JSON

**À implémenter** : Créer un `JSONStorage` qui implémente `StorageBackend` et stocke les données dans `src/data/` au format JSON.

