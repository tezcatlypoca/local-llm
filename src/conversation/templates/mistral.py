"""
Template pour Mistral 7B Instruct (et variantes quantisées).
"""
from typing import List, Optional
from .base import Template
from ..context import Message, MessageRole


class MistralInstructTemplate(Template):
    """
    Template pour Mistral 7B Instruct (quantisé ou non).
    
    Format Mistral Instruct :
    <s>[INST] System: {system_prompt}
    User: {user_message} [/INST] Assistant: {assistant_response} </s>
    
    Ou format simplifié sans système :
    <s>[INST] {user_message} [/INST] {assistant_response} </s>
    
    Pour les conversations multiples :
    <s>[INST] {user_message} [/INST] {assistant_response} </s>[INST] {user_message} [/INST]
    """
    
    def get_system_token(self) -> Optional[str]:
        return "[INST] System:"
    
    def get_user_token(self) -> str:
        return "[INST]"
    
    def get_assistant_token(self) -> str:
        return "[/INST] Assistant:"
    
    def get_end_token(self) -> Optional[str]:
        return "</s>"
    
    def format_message(self, role: MessageRole, content: str) -> str:
        """Formate un message selon son rôle."""
        if role == MessageRole.SYSTEM:
            # Le système est intégré dans le premier [INST]
            return f"{self.system_token} {content}\n"
        elif role == MessageRole.USER:
            return f"{self.user_token} {content} "
        elif role == MessageRole.ASSISTANT:
            return f"{self.assistant_token} {content} {self.end_token}"
        return content
    
    def format_conversation(
        self, 
        messages: List[Message], 
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Formate une conversation complète selon le format Mistral Instruct.
        
        Format: <s>[INST] {user_message} [/INST] {assistant_message} </s>[INST] {user_message} [/INST]
        """
        parts = []
        
        # Récupérer le prompt système
        system_content = system_prompt
        if not system_content:
            # Chercher un message système dans la liste
            for msg in messages:
                if msg.role == MessageRole.SYSTEM:
                    system_content = msg.content
                    break
        
        # Filtrer les messages système de la liste (on les traite séparément)
        conversation_messages = [msg for msg in messages if msg.role != MessageRole.SYSTEM]
        
        # Si pas de messages, retourner vide
        if not conversation_messages:
            return ""
        
        # Commencer avec <s> pour la première paire user/assistant
        first_pair = True
        
        i = 0
        while i < len(conversation_messages):
            msg = conversation_messages[i]
            
            if msg.role == MessageRole.USER:
                # Construire le message utilisateur
                if first_pair:
                    # Premier message : ajouter <s> et le système si présent
                    if system_content:
                        user_text = f"<s>[INST] {system_content}\n\n{msg.content} [/INST]"
                    else:
                        user_text = f"<s>[INST] {msg.content} [/INST]"
                    first_pair = False
                else:
                    # Messages suivants : nouvelle instruction
                    user_text = f"</s>[INST] {msg.content} [/INST]"
                
                parts.append(user_text)
                
                # Chercher la réponse assistant suivante
                if i + 1 < len(conversation_messages) and conversation_messages[i + 1].role == MessageRole.ASSISTANT:
                    assistant_msg = conversation_messages[i + 1]
                    parts.append(f" {assistant_msg.content}")
                    i += 2  # Passer user et assistant
                else:
                    # Pas de réponse assistant, on laisse ouvert pour génération
                    i += 1
            else:
                # Si on a un assistant sans user avant (ne devrait pas arriver)
                i += 1
        
        # Si le dernier message était un user (pas de réponse), on ne ferme pas avec </s>
        # Sinon, on ferme avec </s> si nécessaire
        result = "".join(parts)
        
        # Si le résultat se termine par [/INST] sans réponse, on laisse tel quel
        # Sinon, si on a une réponse assistant, on peut fermer avec </s> (mais pas obligatoire pour génération)
        
        return result

