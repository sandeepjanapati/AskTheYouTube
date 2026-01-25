import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import List, Dict
from services.vertex_client import VertexClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMService:
    """
    Service to construct RAG prompts and handle LLM interaction logic.
    """

    def __init__(self, vertex_client: VertexClient):
        """
        Inject the unified VertexClient.
        """
        self.client = vertex_client
        
        # System Instruction Definition
        self.SYSTEM_INSTRUCTION = (
            "You are a helpful and intelligent AI assistant tailored for YouTube video Q&A. "
            "Your goal is to answer the user's question accurately using ONLY the provided Video Transcript Context. "
            "If the answer is not in the context, politely state that the information is not mentioned in the video. "
            "Do not hallucinate or use outside knowledge."
        )

    def _format_history(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Converts the list of chat objects into a string representation.
        Input: [{'role': 'user', 'content': 'hi'}, {'role': 'model', 'content': 'hello'}]
        Output: 
        User: hi
        AI: hello
        """
        if not chat_history:
            return "No previous conversation."
        
        formatted_history = ""
        for msg in chat_history:
            role = "User" if msg.get('role') == 'user' else "AI"
            content = msg.get('content', '')
            formatted_history += f"{role}: {content}\n"
        
        return formatted_history

    def generate_answer(self, query: str, context: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Constructs the prompt and calls the LLM.
        
        Args:
            query: The current user question.
            context: The retrieved text chunks from Pinecone.
            chat_history: List of previous chat messages.
        """
        logger.info("Constructing LLM Prompt...")

        # 1. Format Chat History
        history_str = self._format_history(chat_history)

        # 2. Construct the Final Prompt
        # We use a structured format to clearly delineate sections for the model.
        full_prompt = (
            f"{self.SYSTEM_INSTRUCTION}\n\n"
            f"--- START OF CONTEXT ---\n"
            f"{context}\n"
            f"--- END OF CONTEXT ---\n\n"
            f"--- CONVERSATION HISTORY ---\n"
            f"{history_str}\n"
            f"--- END OF HISTORY ---\n\n"
            f"User Question: {query}\n"
            f"Answer:"
        )

        try:
            # 3. Call Gemini
            # We don't pass system_instruction separately here because we baked it into the prompt 
            # for stronger adherence in RAG scenarios.
            response_text = self.client.generate_content(full_prompt)
            
            logger.info("LLM Response generated successfully.")
            return response_text

        except Exception as e:
            logger.error(f"Error in LLM Service: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."

# --- Usage Example (Integration in main.py) ---
'''
# 1. Initialization (Once)
# vertex_client = VertexClient() # Initializes both Embed & LLM
# llm_service = LLMService(vertex_client)

# 2. Execution Loop
# answer = llm_service.generate_answer(
#     query="What is the summary?", 
#     context="...retrieved text...", 
#     chat_history=[{'role': 'user', 'content': 'Hi'}]
# )
'''