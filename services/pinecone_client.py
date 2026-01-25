from random import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import logging
from pinecone import Pinecone
from utils.secrets import SecretManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PineconeClient:
    """
    Wrapper for Pinecone Vector Database.
    Handles authentication via Secret Manager and connection pooling.
    """
    
    def __init__(self, index_name: str = "youtube-transcripts"):
        self.index_name = index_name
        self._index = None
        self._connect()

    def _connect(self):
        """Establishes connection to Pinecone using the secret key."""
        try:
            # Dynamic Fetch: Get key from Secret Manager
            api_key = SecretManager.get_secret("PINECONE_API_KEY")
            
            # Initialize Client
            pc = Pinecone(api_key=api_key)
            
            # Initialize Index
            self._index = pc.Index(self.index_name)
            logger.info(f"Successfully connected to Pinecone Index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            raise RuntimeError("Database Connection Failed")

    @property
    def index(self):
        """Returns the initialized index object."""
        if self._index is None:
            self._connect()
        return self._index
    
    def check_video_exists(self, video_id: str) -> bool:
        """
        Checks if a video ID already exists in the vector database.
        
        Logic: 
        1. Create a dummy vector of zeros (Pinecone requires a vector for queries).
        2. Filter strictly by 'video_id'.
        3. Request top_k=1.
        4. If we get 1 result, the video is already indexed.
        """
        try:
            # Create a dummy vector of 0.0s matching the model dimension
            dummy_vector = [1.0] * 768
            
            response = self.index.query(
                vector=dummy_vector,
                filter={"video_id": video_id},
                top_k=1,
                include_metadata=False # We don't need the data, just the count
            )
            
            # Check if any matches were returned
            exists = len(response.get('matches', [])) > 0
            
            if exists:
                logger.info(f"Video {video_id} found in cache (Pinecone).")
            else:
                logger.info(f"Video {video_id} not found in cache.")
                
            return exists
            
        except Exception as e:
            logger.error(f"Error checking video existence in Pinecone: {e}")
            # If check fails, return False so we default to re-processing (safer)
            return False