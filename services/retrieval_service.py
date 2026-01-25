import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import List, Dict, Any, Tuple
from services.vertex_client import VertexClient
from services.pinecone_client import PineconeClient

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetrievalService:
    """
    Service to handle Semantic Search and Context Construction.
    """

    def __init__(self, vertex_client: VertexClient, pinecone_client: PineconeClient):
        """
        Dependency Injection of the singleton clients initialized in main.py.
        """
        self.embedder = vertex_client
        self.db = pinecone_client
        
        # Configuration
        self.TOP_K = 10  # Number of chunks to retrieve

    def _embed_query(self, query: str) -> List[float]:
        """
        Converts the user query into a vector.
        CRITICAL: Uses task_type='RETRIEVAL_QUERY' which optimizes the embedding 
        specifically for asking questions (as opposed to 'RETRIEVAL_DOCUMENT' used for storage).
        """
        try:
            # get_embeddings returns a list of lists, we take the first one since we have 1 query
            vectors = self.embedder.get_embeddings([query], task_type="RETRIEVAL_QUERY")
            if not vectors:
                raise ValueError("Embedding model returned empty result.")
            return vectors[0]
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise e

    def _search_pinecone(self, query_vector: List[float], video_id: str) -> List[Dict[str, Any]]:
        """
        Queries Pinecone for the most similar chunks, strictly filtered by video_id.
        """
        try:
            # Filter Syntax: Matches metadata field 'video_id' exactly
            metadata_filter = {"video_id": video_id}

            search_response = self.db.index.query(
                vector=query_vector,
                filter=metadata_filter,
                top_k=self.TOP_K,
                include_metadata=True # Essential to get the actual text content back
            )

            return search_response.get('matches', [])

        except Exception as e:
            logger.error(f"Pinecone search failed for video {video_id}: {e}")
            raise e

    def get_context(self, query: str, video_id: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Main execution method for Step 4.
        
        Args:
            query: The user's question.
            video_id: The ID of the video being chatted about.
            
        Returns:
            context_string: A single string containing all retrieved text joined together.
            sources: List of metadata (timestamps/text) for UI citation references.
        """
        if not query or not video_id:
            logger.warning("Invalid input for context retrieval.")
            return "", []

        logger.info(f"Retrieving context for query: '{query[:50]}...' in video: {video_id}")

        # 1. Vector Search (Semantic Search)
        # Convert Query to Vector
        query_vector = self._embed_query(query)
        
        # Query Pinecone with filter
        matches = self._search_pinecone(query_vector, video_id)

        if not matches:
            logger.warning(f"No relevant matches found for video {video_id}.")
            return "", []

        # 2. Context Construction
        context_parts = []
        sources = []

        # Iterate through matches (they are already sorted by similarity score)
        for match in matches:
            metadata = match.get('metadata', {})
            text_content = metadata.get('text_content', '')
            timestamp = metadata.get('start_time', 0)
            
            if text_content:
                # Append text to context list
                context_parts.append(text_content)
                
                # Keep track of source info for the UI/Debug
                sources.append({
                    "text": text_content,
                    "start_time": timestamp,
                    "score": match.get('score', 0.0)
                })

        # Combine into a single string
        # We join with newlines to help the LLM distinguish separate chunks
        context_string = "\n\n".join(context_parts)
        
        logger.info(f"Context constructed. Retrieved {len(matches)} chunks. Total length: {len(context_string)} chars.")
        
        return context_string, sources

# --- Usage Example ---
'''

# Initialize these ONCE at application startup
vertex_client = VertexClient()
pinecone_client = PineconeClient()

# Initialize Service with dependencies
retrieval_service = RetrievalService(vertex_client, pinecone_client)

# Use service

context, sources = retrieval_service.get_context(
    query="What does the speaker say about cloud functions?",
    video_id="abc123xyz"
)

# context is now ready to be injected into the LLM Prompt
'''