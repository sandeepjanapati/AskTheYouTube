import re
import html
import logging
import requests
from typing import List, Dict, Optional, Any
from utils.secrets import SecretManager
from services.vertex_client import VertexClient 

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoProcessingError(Exception):
    """Custom exception for general video processing failures."""
    pass

class InvalidYouTubeURLError(VideoProcessingError):
    """Raised when the provided URL is not a valid YouTube link."""
    pass

class TranscriptUnavailableError(VideoProcessingError):
    """Raised when no transcripts (manual or auto) are available for the video."""
    pass

class YouTubeService:
    """
    Service class to handle YouTube video ID extraction and transcript fetching
    using RapidAPI to avoid Google Cloud IP blocking.
    """

    # RapidAPI Configuration
    _RAPID_API_HOST = "youtube-transcript3.p.rapidapi.com"
    _RAPID_API_URL = "https://youtube-transcript3.p.rapidapi.com/api/transcript"

    def __init__(self, vertex_client: Optional[VertexClient] = None):
        """
        Args:
            vertex_client: Used for the final fallback to ask Gemini for a summary.
        """
        self.vertex_client = vertex_client

    @staticmethod
    def extract_video_id(url: str) -> str:
        """
        Parses a YouTube URL to extract the 11-character Video ID.
        
        Handles formats:
        - standard: youtube.com/watch?v=VIDEO_ID
        - shortened: youtu.be/VIDEO_ID
        - embed: youtube.com/embed/VIDEO_ID
        - shorts: youtube.com/shorts/VIDEO_ID
        """
        if not url:
            raise InvalidYouTubeURLError("URL cannot be empty.")

        # Regex to capture the 11-char ID
        # Matches patterns: v=ID, embed/ID, shorts/ID, youtu.be/ID
        regex = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?|shorts)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
        
        match = re.search(regex, url)
        
        if not match:
            logger.error(f"Failed to extract Video ID from URL: {url}")
            raise InvalidYouTubeURLError("Invalid YouTube URL provided.")
        
        video_id = match.group(1)
        
        # Double check length just to be safe (Regex usually handles this via {11})
        if len(video_id) != 11:
            raise InvalidYouTubeURLError("Extracted Video ID is invalid (incorrect length).")
            
        logger.info(f"Successfully extracted Video ID: {video_id}")
        return video_id

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Sanitizes raw transcript text:
        1. Unescapes HTML entities (e.g., &#39; -> ')
        2. Removes HTML tags (e.g., <i>, <b>)
        3. Normalizes whitespace (removes newlines, non-breaking spaces)
        """
        if not text:
            return ""

        # 1. Decode HTML entities
        cleaned = html.unescape(text)

        # 2. Remove HTML tags (rare but possible in manual captions)
        cleaned = re.sub(r'<[^>]+>', '', cleaned)

        # 3. Normalize whitespace: replace newlines and &nbsp; with a single space
        cleaned = cleaned.replace('\n', ' ').replace('\xa0', ' ')
        
        # Remove extra spaces created by replacement
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned


    def get_transcript(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Main entry point. Tries RapidAPI first, then falls back.
        """
        try:
            return self._fetch_via_rapidapi(video_id)
        except (TranscriptUnavailableError, VideoProcessingError) as e:
            # Only fallback on real "no transcript" cases or quota
            logger.warning(f"RapidAPI failed legitimately ({str(e)}). Falling back to Gemini summary...")
            return self._fetch_via_gemini_fallback(video_id)
        except Exception as e:
            logger.error(f"Critical RapidAPI failure: {e}")
            raise  # Let it bubble up for monitoring

    def _fetch_via_rapidapi(self, video_id: str) -> List[Dict[str, Any]]:
        """Layer 1: RapidAPI"""
        logger.info(f"Attempting RapidAPI for {video_id}...")
        rapid_api_key = SecretManager.get_secret("rapidapi-key")

        if not rapid_api_key:
            raise VideoProcessingError("RapidAPI key not found.")
        
        querystring = {"videoId": video_id}
        headers = {
            "X-RapidAPI-Key": rapid_api_key,
            "X-RapidAPI-Host": self._RAPID_API_HOST
        }

        try:
            response = requests.get(self._RAPID_API_URL, headers=headers, params=querystring, timeout=15)
            
            if response.status_code == 429:
                raise VideoProcessingError("RapidAPI quota exceeded.")
            if response.status_code == 404:
                raise TranscriptUnavailableError(f"No transcript available for {video_id}")
            if response.status_code != 200:
                logger.error(f"RapidAPI failed {response.status_code}: {response.text[:300]}")
                raise VideoProcessingError(f"RapidAPI error: {response.status_code}")

            data = response.json()
            logger.debug(f"API response keys: {list(data.keys())}")

            # Use the structure that worked before
            raw_segments = data.get("transcript", [])
            if not isinstance(raw_segments, list) or not raw_segments:
                error_msg = data.get("error") or data.get("message") or "Empty or unexpected format"
                logger.warning(f"No valid transcript segments: {error_msg}")
                raise TranscriptUnavailableError("No transcript found in API response")

            cleaned = []
            for segment in raw_segments:
                raw_text = segment.get("text", "")
                start = segment.get("offset", 0.0)  # ← this API uses "offset"

                clean_txt = self._clean_text(raw_text)
                if clean_txt:
                    cleaned.append({
                        "text": clean_txt,
                        "start": float(start)
                    })

            if not cleaned:
                raise TranscriptUnavailableError("Transcript segments were empty after cleaning")

            logger.info(f"RapidAPI success: {len(cleaned)} segments")
            return cleaned

        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request failed: {req_err}")
            raise VideoProcessingError("Network/API connection issue")
        except ValueError as json_err:
            logger.error(f"Invalid JSON: {response.text[:300]}")
            raise VideoProcessingError("API returned invalid JSON")
        except Exception as e:
            logger.error(f"RapidAPI unexpected error: {e}")
            raise
        

    def _fetch_via_gemini_fallback(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Layer 3: Gemini Direct Query
        Asks Gemini to summarize the video URL based on its training data.
        Returns it as a single 'transcript segment' so RAG still works.
        """
        if not self.vertex_client:
            raise VideoProcessingError("Vertex Client not initialized for fallback.")

        logger.info("Asking Gemini for a fallback summary...")
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        prompt = (
            f"I cannot retrieve the transcript for this YouTube video: {url}. "
            f"Based on your internal knowledge, please provide a detailed summary "
            f"of what this video is likely about. Cover the main topics and key points."
        )

        try:
            # Generate content using the LLM
            summary = self.vertex_client.generate_content(prompt)
            
            if not summary:
                raise VideoProcessingError("Gemini returned empty summary.")

            # Return as a single chunk so it gets embedded and stored
            return [{
                "text": f"[FALLBACK SUMMARY BY AI]: {summary}",
                "start": 0.0
            }]
            
        except Exception as e:
            logger.error(f"Gemini Fallback failed: {e}")
            raise VideoProcessingError("All transcript retrieval methods failed.")


# --- Usage Example (For testing purposes) ---
if __name__ == "__main__":
    service = YouTubeService()
    
    # Test with a known video (Example: A Google Cloud Tech video)
    test_url = "https://www.youtube.com/watch?v=7t2alSnE2-I&t=788s" 
    
    try:
        vid_id = service.extract_video_id(test_url)
        transcript_data = service.get_transcript(vid_id)
        
        # Print first 3 chunks to verify output format
        print(f"\n--- Result Preview for {vid_id} ---")
        for chunk in transcript_data[:3]:
            print(chunk)
            
    except VideoProcessingError as e:
        print(f"Error: {e}")