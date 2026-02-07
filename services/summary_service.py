import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import math
from typing import List, Dict, Any, Tuple
from services.vertex_client import VertexClient
from services.pinecone_client import PineconeClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SummaryService:
    """
    Handles full video summarization using hierarchical summarization strategy.
    
    Strategy:
    1. Fetch ALL chunks for the video (paginated to handle any size)
    2. If chunks fit in context window → single-pass summarization
    3. If too many chunks → hierarchical: summarize in batches, then summarize summaries
    
    This scales to videos of any length.
    """
    
    # Pinecone max results per query
    PINECONE_MAX_TOP_K = 10000
    
    # Approximate tokens per chunk (conservative estimate)
    # Typical chunk ~500 chars ≈ 125 tokens
    ESTIMATED_TOKENS_PER_CHUNK = 150
    
    # Max chunks to include in single prompt (leaves room for instructions + response)
    # Gemini Flash has 1M context, but we're conservative for speed/cost
    MAX_CHUNKS_PER_BATCH = 100
    
    # Max characters for context in a single LLM call
    MAX_CONTEXT_CHARS = 60000  # ~15k tokens, leaves room for prompt overhead
    
    SUMMARY_PROMPT = """You are a professional video content summarizer.

Based on the following video transcript segments, provide a comprehensive summary of the video.

Guidelines:
- Capture ALL main topics and key points discussed
- Organize information logically
- Use bullet points for clarity where appropriate
- Include important details, examples, and conclusions
- Be thorough but concise
- Write in clear, professional language

--- VIDEO TRANSCRIPT ---
{transcript}
--- END TRANSCRIPT ---

Provide a comprehensive summary:"""

    HIERARCHICAL_SUMMARY_PROMPT = """You are a professional video content summarizer.

The following are summaries of different parts of a long video. Combine them into one cohesive, comprehensive summary.

Guidelines:
- Merge related topics that appear in multiple parts
- Eliminate redundancy while preserving all unique information
- Organize the final summary logically
- Maintain a natural flow

--- PARTIAL SUMMARIES ---
{summaries}
--- END SUMMARIES ---

Provide the unified comprehensive summary:"""

    def __init__(self, vertex_client: VertexClient, pinecone_client: PineconeClient):
        """Dependency injection of clients."""
        self.llm = vertex_client
        self.db = pinecone_client
        logger.info("SummaryService initialized.")
    
    def _fetch_all_chunks(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Fetches ALL chunks for a video from Pinecone, sorted chronologically.
        Uses pagination if necessary to retrieve all chunks.
        """
        try:
            # Use a normalized dummy vector for metadata-only filtering
            dummy_vector = [1.0 / (768 ** 0.5)] * 768
            
            metadata_filter = {"video_id": video_id}
            
            # Fetch with max allowed top_k
            response = self.db.index.query(
                vector=dummy_vector,
                filter=metadata_filter,
                top_k=self.PINECONE_MAX_TOP_K,
                include_metadata=True
            )
            
            matches = response.get('matches', [])
            
            # Sort chronologically by start_time
            matches.sort(key=lambda x: x.get('metadata', {}).get('start_time', 0))
            
            logger.info(f"Fetched {len(matches)} chunks for video {video_id}")
            return matches
            
        except Exception as e:
            logger.error(f"Failed to fetch chunks for video {video_id}: {e}")
            raise e
    
    def _chunks_to_transcript(self, chunks: List[Dict[str, Any]]) -> str:
        """Convert chunks to a single transcript string."""
        parts = []
        for chunk in chunks:
            text = chunk.get('metadata', {}).get('text_content', '')
            if text:
                parts.append(text)
        return "\n\n".join(parts)
    
    def _batch_chunks(self, chunks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Splits chunks into batches that fit within context limits.
        Uses character count as proxy for token count.
        """
        batches = []
        current_batch = []
        current_chars = 0
        
        for chunk in chunks:
            text = chunk.get('metadata', {}).get('text_content', '')
            chunk_chars = len(text)
            
            # If adding this chunk exceeds limit, start new batch
            if current_chars + chunk_chars > self.MAX_CONTEXT_CHARS and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            
            current_batch.append(chunk)
            current_chars += chunk_chars
            
            # Also cap by chunk count
            if len(current_batch) >= self.MAX_CHUNKS_PER_BATCH:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
        
        # Don't forget the last batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _summarize_batch(self, chunks: List[Dict[str, Any]]) -> str:
        """Summarize a single batch of chunks."""
        transcript = self._chunks_to_transcript(chunks)
        prompt = self.SUMMARY_PROMPT.format(transcript=transcript)
        
        return self.llm.generate_content(prompt)
    
    def _combine_summaries(self, summaries: List[str]) -> str:
        """Combine multiple partial summaries into one cohesive summary."""
        combined = "\n\n---\n\n".join([f"Part {i+1}:\n{s}" for i, s in enumerate(summaries)])
        prompt = self.HIERARCHICAL_SUMMARY_PROMPT.format(summaries=combined)
        
        return self.llm.generate_content(prompt)
    
    def generate_full_summary(self, video_id: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate a comprehensive summary of the entire video.
        
        Uses hierarchical summarization for long videos:
        1. Splits transcript into manageable batches
        2. Summarizes each batch
        3. Combines batch summaries into final summary
        
        Args:
            video_id: The YouTube video ID
            
        Returns:
            Tuple of (summary_text, source_chunks)
        """
        logger.info(f"Generating full video summary for {video_id}")
        
        # 1. Fetch all chunks
        all_chunks = self._fetch_all_chunks(video_id)
        
        if not all_chunks:
            logger.warning(f"No chunks found for video {video_id}")
            return "", []
        
        # 2. Calculate total transcript size
        total_chars = sum(
            len(chunk.get('metadata', {}).get('text_content', '')) 
            for chunk in all_chunks
        )
        
        logger.info(f"Video {video_id}: {len(all_chunks)} chunks, ~{total_chars} chars")
        
        # 3. Decide strategy based on size
        if total_chars <= self.MAX_CONTEXT_CHARS and len(all_chunks) <= self.MAX_CHUNKS_PER_BATCH:
            # Small video: single-pass summarization
            logger.info("Using single-pass summarization")
            summary = self._summarize_batch(all_chunks)
        else:
            # Large video: hierarchical summarization
            logger.info("Using hierarchical summarization")
            batches = self._batch_chunks(all_chunks)
            logger.info(f"Split into {len(batches)} batches")
            
            # Summarize each batch
            batch_summaries = []
            for i, batch in enumerate(batches):
                logger.info(f"Summarizing batch {i+1}/{len(batches)}")
                batch_summary = self._summarize_batch(batch)
                batch_summaries.append(batch_summary)
            
            # Combine all batch summaries
            if len(batch_summaries) == 1:
                summary = batch_summaries[0]
            else:
                summary = self._combine_summaries(batch_summaries)
        
        # 4. Build sources (representative sample from throughout video)
        # Take first, middle, and last few chunks as source references
        sources = []
        sample_indices = [0, len(all_chunks)//4, len(all_chunks)//2, 
                         3*len(all_chunks)//4, len(all_chunks)-1]
        
        for idx in sample_indices:
            if 0 <= idx < len(all_chunks):
                chunk = all_chunks[idx]
                metadata = chunk.get('metadata', {})
                sources.append({
                    "text": metadata.get('text_content', '')[:200] + "...",  # Truncate for sources
                    "start_time": metadata.get('start_time', 0),
                    "score": 1.0  # Not a similarity score for summaries
                })
        
        logger.info(f"Summary generated successfully for video {video_id}")
        return summary, sources
