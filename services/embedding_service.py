import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import logging
from typing import List, Dict, Any, Generator

# Import the modular clients
from services.vertex_client import VertexClient
from services.pinecone_client import PineconeClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service to orchestrate the transformation of Chunked Documents -> Vectors -> Pinecone.
    """

    def __init__(self, vertex_client: VertexClient, pinecone_client: PineconeClient):
        """
        Args:
            vertex_client: Pre-initialized embedding model wrapper.
            pinecone_client: Pre-initialized vector DB wrapper.
        """
        self.embedder = vertex_client
        self.db = pinecone_client
        
        # Configuration for limits
        self.EMBEDDING_BATCH_SIZE = 10 
        self.UPSERT_BATCH_SIZE = 100

    def _generate_batches(self, data: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
        """Helper to slice data into manageable batches."""
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def process_and_store(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Main execution flow.
        1. Embeds text chunks in batches using Vertex AI.
        2. Upserts vectors in batches to Pinecone.
        """
        if not documents:
            logger.warning("No documents to process.")
            return False

        video_id = documents[0]['metadata'].get('video_id', 'unknown')
        logger.info(f"Processing embeddings for video {video_id}. Total documents: {len(documents)}")

        vectors_to_upsert = []

        # --- PHASE 1: Generate Embeddings ---
        # We batch this to respect Vertex AI request limits
        
        for batch in self._generate_batches(documents, self.EMBEDDING_BATCH_SIZE):
            texts = [doc['metadata']['text_content'] for doc in batch]
            
            try:
                # Use the injected client
                vector_values = self.embedder.get_embeddings(texts, task_type="RETRIEVAL_DOCUMENT")
                
                # Attach vectors to document objects
                for i, doc in enumerate(batch):
                    vectors_to_upsert.append({
                        "id": doc['id'],
                        "values": vector_values[i],
                        "metadata": doc['metadata']
                    })
                
                # Polite rate limiting
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to embed batch for video {video_id}: {e}")
                # In production, you might want to retry or partial fail
                continue

        # --- PHASE 2: Upsert to Pinecone ---
        
        if not vectors_to_upsert:
            logger.error("No vectors generated. Aborting upsert.")
            return False

        logger.info(f"Generated {len(vectors_to_upsert)} vectors. Upserting to Pinecone...")

        try:
            count = 0
            for batch in self._generate_batches(vectors_to_upsert, self.UPSERT_BATCH_SIZE):
                self.db.index.upsert(vectors=batch)
                count += len(batch)
                logger.info(f"Upserted {count}/{len(vectors_to_upsert)} vectors...")
            
            logger.info(f"Success: Video {video_id} is now indexed.")
            return True

        except Exception as e:
            logger.error(f"Pinecone Upsert Failed: {e}")
            raise e

# --- Initialization & Usage Example (In main.py or dependency container) ---
'''
# Initialize these ONCE at application startup
vertex_client = VertexClient()
pinecone_client = PineconeClient()

# Initialize Service with dependencies
embedding_service = EmbeddingService(vertex_client, pinecone_client)

# Use service
# embedding_service.process_and_store(documents)
'''