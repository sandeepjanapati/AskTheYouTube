import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from enum import Enum
from vertexai.generative_models import GenerativeModel, GenerationConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Classification of user query intent."""
    FULL_VIDEO_SUMMARY = "full_video_summary"  # User wants summary of entire video
    SPECIFIC_QUERY = "specific_query"          # User has a specific question/topic


class IntentClassifier:
    """
    Uses a lightweight LLM (Gemini Flash) to classify user intent.
    Determines if user wants a full video summary or has a specific question.
    """
    
    # Using Gemini 2.0 Flash-Lite - fastest and cheapest for simple classification
    MODEL_NAME = "gemini-2.0-flash-lite"
    
    CLASSIFICATION_PROMPT = """You are an intent classifier for a YouTube video Q&A application.

Your task: Determine if the user wants a FULL SUMMARY of the entire video, or if they have a SPECIFIC question about a topic in the video.

FULL_VIDEO_SUMMARY - User wants:
- A complete overview of the video
- Summary of all main topics covered
- General "what is this video about" questions
- No specific topic mentioned

SPECIFIC_QUERY - User wants:
- Information about a specific topic, concept, or section
- Summary of a PARTICULAR part/topic (not the whole video)
- Answers to specific questions
- Details about something specific mentioned

Examples:
- "Give me a summary" → FULL_VIDEO_SUMMARY
- "Summarize the whole video" → FULL_VIDEO_SUMMARY  
- "What are the main points?" → FULL_VIDEO_SUMMARY
- "What is this video about?" → FULL_VIDEO_SUMMARY
- "Summarize the part about machine learning" → SPECIFIC_QUERY
- "What does he say about Python?" → SPECIFIC_QUERY
- "Give me a summary of the authentication section" → SPECIFIC_QUERY
- "How does the speaker explain databases?" → SPECIFIC_QUERY
- "What are the key points about security?" → SPECIFIC_QUERY

User Query: {query}

Respond with ONLY one word: either FULL_VIDEO_SUMMARY or SPECIFIC_QUERY"""

    def __init__(self):
        """
        Initialize the lightweight classification model.
        NOTE: Vertex AI must already be initialized (via VertexClient) before creating this.
        """
        try:
            logger.info(f"Initializing Intent Classifier with model: {self.MODEL_NAME}")
            self._model = GenerativeModel(self.MODEL_NAME)
            
            # Fast, deterministic config for classification
            self._config = GenerationConfig(
                temperature=0.0,  # Deterministic
                max_output_tokens=20,  # Only need one word
                top_p=1.0,
                top_k=1
            )
            logger.info("Intent Classifier initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Intent Classifier: {e}")
            raise e
    
    def classify(self, query: str) -> QueryIntent:
        """
        Classify user query intent using LLM.
        
        Args:
            query: The user's question
            
        Returns:
            QueryIntent enum indicating full summary or specific query
        """
        if not query or not query.strip():
            logger.warning("Empty query received, defaulting to SPECIFIC_QUERY")
            return QueryIntent.SPECIFIC_QUERY
        
        try:
            prompt = self.CLASSIFICATION_PROMPT.format(query=query)
            
            response = self._model.generate_content(
                prompt,
                generation_config=self._config
            )
            
            result = response.text.strip().upper()
            logger.info(f"Intent classification for '{query[:50]}...': {result}")
            
            if "FULL_VIDEO_SUMMARY" in result:
                return QueryIntent.FULL_VIDEO_SUMMARY
            else:
                # Default to specific query for safety
                return QueryIntent.SPECIFIC_QUERY
                
        except Exception as e:
            logger.error(f"Intent classification failed: {e}. Defaulting to SPECIFIC_QUERY")
            # Fail-safe: default to specific query (normal RAG flow)
            return QueryIntent.SPECIFIC_QUERY

