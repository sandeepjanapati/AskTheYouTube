import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import logging
from typing import List, Optional
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel, GenerationConfig, SafetySetting
from utils.secrets import SecretManager


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VertexClient:
    """
    Wrapper for Google Vertex AI Embedding and LLM Model.
    Initializes the model once and provides a clean interface for embedding generation and Text Generation.
    """
    
    def __init__(self, embedding_model_name: str = "text-multilingual-embedding-002", llm_model_name: str = "gemini-2.5-flash"):
        self.project_id = SecretManager.get_project_id()
        self.location = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
       
        # Initialize Vertex AI SDK
        vertexai.init(project=self.project_id, location=self.location)

        # 1. Load Embedding Model
        self._embedding_model = self._load_embedding_model(embedding_model_name)

        # 2. Load LLM (Gemini)
        self._llm_model = self._load_llm_model(llm_model_name)

    def _load_embedding_model(self, name: str):
        try:
            logger.info(f"Initializing Embedding Model: {name}")
            return TextEmbeddingModel.from_pretrained(name)
        except Exception as e:
            logger.error(f"Failed to load Embedding model: {e}")
            raise e

    def _load_llm_model(self, name: str):
        try:
            logger.info(f"Initializing LLM Model: {name}")
            # System instructions can be passed here or per request. 
            # We initialize the generic model here.
            return GenerativeModel(name)
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise e

    def get_embeddings(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        """
        Generates embeddings for a list of texts.
        Args:
            texts: List of strings to embed.
            task_type: 'RETRIEVAL_DOCUMENT' for storage, 'RETRIEVAL_QUERY' for search.
        """
        if not texts:
            return []

        try:
            # Wrap texts in TextEmbeddingInput for task specificity
            inputs = [TextEmbeddingInput(text=t, task_type=task_type) for t in texts]
            
            # Call API
            embeddings = self._embedding_model.get_embeddings(inputs)
            
            # Extract vectors
            return [embedding.values for embedding in embeddings]
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            # Simple retry logic could go here
            raise e

     # --- LLM Methods ---
    def generate_content(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """
        Wrapper to call Gemini.
        """
        try:
            # Configuration for RAG: Low temperature for factual grounding
            config = GenerationConfig(
                temperature=0.2,
                max_output_tokens=1024,
                top_p=0.8,
                top_k=40
            )

            # Safety Settings: Adjust as needed (Blocking 'None' allows for robust context handling)
            # In latest SDK, settings are dictionaries or SafetySetting objects
            safety_settings = [
                SafetySetting(
                    category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
                ),
                SafetySetting(
                    category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
                ),
                # Add others as needed
            ]

            # If system instruction is provided specifically
            if system_instruction:
                # Note: System instructions are usually set at model init or via specific methods depending on SDK version.
                # For simplicity in this wrapper, we often prepend it to the prompt or use the new 'system_instruction' param if re-initializing.
                # With a shared model instance, we simply prepend to the prompt string for "Stateless" RAG.
                pass 

            response = self._llm_model.generate_content(
                prompt,
                generation_config=config,
                safety_settings=safety_settings
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Gemini Generation Failed: {e}")
            raise e