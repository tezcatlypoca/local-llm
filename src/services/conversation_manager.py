from typing import Optional
from utils.conversation_model import ConversationModel

class ConversationManager:

    def __init__(self):
        self.conversations = []
    
    # POST conversations/create
    def create_conversation(self, id: Optional[int] = 0, name: Optional[str] = "Conversation", temperature: Optional[float] = 0.7, message_max: Optional[int] = 10):
        conv = ConversationModel(id=id, name=name, temperature=temperature, message_max=message_max)
        self.conversations.append(conv)
        return conv
    
    # GET conversations/
    def get_conversations(self):
        return self.conversations
    
    # GET conversations/{id}
    def get_conversation(self, id: int):
        return self.conversations[id]
    
    # POST conversations/delete/{id}
    def delete_conversation(self, id: int):
        self.conversations.pop(id)

    # POST conversations/{id}/message
    def post_message(self, id: int, message: str):
        self.conversations[id].messages.append(message)