import logging
import uuid
import bisect
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChunkingService:
    """
    Service to handle the splitting of transcript text into semantic chunks
    suitable for Vector Database ingestion (RAG).
    """

    def __init__(self):
        # Configuration as defined in requirements
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200
        self.SEPARATORS = ["\n\n", "\n", " ", ""]
        
        # Initialize the LangChain splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            separators=self.SEPARATORS,
            strip_whitespace=True
        )

    def _create_text_map(self, transcript_segments: List[Dict[str, Any]]) -> Tuple[str, List[Tuple[int, float]]]:
        """
        Concatenates all transcript segments into one large string.
        Creates a mapping list where each entry is (character_index, start_time).
        
        Args:
            transcript_segments: List of dicts like [{"text": "Hello", "start": 0.5}, ...]
            
        Returns:
            full_text: The combined string.
            offset_map: List of tuples [(0, 0.5), (6, 1.2), ...] mapping char indices to timestamps.
        """
        full_text = ""
        offset_map = [] # List of (char_index, start_time)

        for segment in transcript_segments:
            text = segment.get('text', '')
            start_time = segment.get('start', 0.0)
            
            # Map the current character index to this segment's start time
            current_index = len(full_text)
            offset_map.append((current_index, start_time))
            
            # Append text with a space for natural flow
            full_text += text + " "
            
        return full_text, offset_map

    def _find_timestamp_for_chunk(self, chunk_start_index: int, offset_map: List[Tuple[int, float]]) -> float:
        """
        Uses binary search (bisect) to find the closest timestamp for a given character index.
        """
        # Create a list of just the indices for bisecting
        indices = [x[0] for x in offset_map]
        
        # Find the insertion point to the right of the chunk_start_index
        idx = bisect.bisect_right(indices, chunk_start_index)
        
        # We take idx - 1 to get the segment that started *before* or *at* this index
        if idx > 0:
            return offset_map[idx - 1][1]
        return 0.0

    def chunk_transcript(self, video_id: str, transcript_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main execution method.
        1. Pre-computes text and timestamp map.
        2. Splits text using RecursiveCharacterTextSplitter.
        3. Maps chunks back to timestamps.
        4. Enriches with metadata (video_id, ids, etc).
        """
        if not transcript_data:
            logger.warning(f"No transcript data provided for video {video_id}")
            return []

        logger.info(f"Starting chunking process for video: {video_id} with {len(transcript_data)} segments.")

        # 1. Pre-computation Logic
        full_text, offset_map = self._create_text_map(transcript_data)
        
        # 2. The Chunking Execution
        # We use split_text to get raw strings first
        raw_chunks = self.splitter.split_text(full_text)
        
        logger.info(f"Splitter generated {len(raw_chunks)} chunks for video {video_id}.")

        documents = []
        current_search_pos = 0

        # 3. Metadata Enrichment
        for index, chunk_text in enumerate(raw_chunks):
            # Find where this chunk actually sits in the full text
            # We start searching from current_search_pos to avoid finding earlier duplicates
            # Note: Due to overlap, we can't strictly advance current_search_pos by len(chunk).
            # We assume sequential order from the splitter.
            
            start_index = full_text.find(chunk_text, current_search_pos)
            
            # Fallback if find fails (should rarely happen with exact string slices)
            if start_index == -1:
                # Reset search from 0 if logic drifts, though unsafe for duplicates
                start_index = full_text.find(chunk_text)
                
            if start_index != -1:
                # Calculate approximate next search position (accounting for overlap logic conservatively)
                # We move forward by half the chunk length to be safe against skipping valid overlaps
                current_search_pos = start_index + (len(chunk_text) // 2)

                # Get Timestamp
                timestamp = self._find_timestamp_for_chunk(start_index, offset_map)
                
                # Construct The Object
                unique_id = f"{video_id}_{index}_{str(uuid.uuid4())[:8]}"
                
                doc = {
                    "id": unique_id,
                    "values": [], # To be filled by Embedding Service
                    "metadata": {
                        "video_id": video_id,
                        "text_content": chunk_text,
                        "start_time": timestamp,
                        "chunk_index": index,
                        "source": f"https://www.youtube.com/watch?v={video_id}&t={int(timestamp)}s"
                    }
                }
                documents.append(doc)
            else:
                logger.error(f"Could not map chunk {index} back to original text. Skipping.")

        # 4. Output Verification
        if not documents:
            logger.warning("Chunking resulted in 0 documents.")
        else:
            logger.info(f"Successfully created {len(documents)} enriched documents.")
            
        return documents

# --- Usage Example (For internal testing) ---
'''
if __name__ == "__main__":
    # Mock data resembling Step 1 output
    transcript_data = [
        {"text": "Hello everyone and welcome to the course.", "start": 0.0},
        {"text": "Today we are learning about GCP.", "start": 5.2},
        {"text": "It is a very powerful platform.", "start": 10.5},
        # ... imagine 1000s of lines here
    ]
    
    chunker = ChunkingService()
    documents = chunker.chunk_transcript(vid_id, transcript_data)

    for item in documents[:2]:
        print(item)
'''