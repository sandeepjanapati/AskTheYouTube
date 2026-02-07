import logging
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import Service Modules
from services.youtube_service import YouTubeService
from services.chunking_service import ChunkingService
from services.vertex_client import VertexClient
from services.pinecone_client import PineconeClient
from services.embedding_service import EmbeddingService
from services.retrieval_service import RetrievalService
from services.llm_service import LLMService
from services.intent_classifier import IntentClassifier, QueryIntent
from services.summary_service import SummaryService

# --- CONFIGURATION ---
# Configure logging to show up in Cloud Run logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AskTheYouTube-Backend")

# Initialize FastAPI App
app = FastAPI(title="AskTheYouTube API", version="1.0.0")

# --- CORS CONFIGURATION ---
# This is critical for allowing your Firebase Frontend to talk to this Cloud Run Backend
origins = [
    "http://localhost:5000",                # Local Firebase Serve
    "http://127.0.0.1:5500",                # Local Live Server (VS Code)
    "http://127.0.0.1:5501",                # Local Live Server (alternate port)
    "https://ask-yt.web.app",
    "https://ask-yt.firebaseapp.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"], 
    allow_headers=["*"],              
)

# --- DATA MODELS (Pydantic) ---
class VideoRequest(BaseModel):
    url: str

class VideoResponse(BaseModel):
    message: str
    video_id: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    video_id: str
    history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[Dict]] = []

# --- GLOBAL SERVICE INSTANCES ---
# We use global variables to hold service instances so they are initialized once on startup
# rather than recreated for every single request (Singleton pattern).
pinecone_client: PineconeClient = None
vertex_client: VertexClient = None
youtube_service: YouTubeService = None
chunking_service: ChunkingService = None
embedding_service: EmbeddingService = None
retrieval_service: RetrievalService = None
llm_service: LLMService = None
intent_classifier: IntentClassifier = None
summary_service: SummaryService = None

@app.on_event("startup")
async def startup_event():
    """
    Initializes connections to Google Cloud and Pinecone when the server starts.
    """
    global youtube_service, chunking_service, embedding_service, retrieval_service, llm_service, pinecone_client, vertex_client, intent_classifier, summary_service
    
    try:
        logger.info("Starting up: Initializing Services...")

        # 1. Initialize Clients (Connection logic)
        # VertexClient handles both Embedding Model and Gemini LLM connections
        vertex_client = VertexClient() 
        pinecone_client = PineconeClient()

        # 2. Initialize Logic Services (Dependency Injection)
        youtube_service = YouTubeService(vertex_client=vertex_client)
        chunking_service = ChunkingService()
        
        # Injects the clients into the orchestrators
        embedding_service = EmbeddingService(vertex_client, pinecone_client)
        retrieval_service = RetrievalService(vertex_client, pinecone_client)
        llm_service = LLMService(vertex_client)
        
        # 3. Initialize Intent Classification and Summary Services
        intent_classifier = IntentClassifier()
        summary_service = SummaryService(vertex_client, pinecone_client)
        
        logger.info("All services initialized successfully.")

    except Exception as e:
        logger.critical(f"Failed to initialize services on startup: {e}")
        raise RuntimeError("Service Initialization Failed")

# --- API ENDPOINTS ---

@app.get("/")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "running", "service": "AskTheYouTube Backend"}

@app.post("/process-video", response_model=VideoResponse)
async def process_video(request: VideoRequest):
    """
    Step 1-3 Orchestration:
    1. Extract Video ID
    2. CHECK: If ID exists in Pinecone, return success immediately.
    3. If not, Download Transcript
    4. Chunking
    5. Embed & Store in Pinecone
    """
    try:
        # 1. Validation & Extraction
        logger.info(f"Received request to process URL: {request.url}")
        video_id = youtube_service.extract_video_id(request.url)

        # 2. Check if we have already processed this video to save time and money
        if pinecone_client.check_video_exists(video_id):
            logger.info(f"Skipping processing for {video_id} - already in database.")
            return VideoResponse(
                message="Video loaded from cache.",
                video_id=video_id
            )

        # 3. Get Transcript
        transcript_data = youtube_service.get_transcript(video_id)
        if not transcript_data:
            raise HTTPException(status_code=400, detail="Could not retrieve transcript.")

        # 4. Chunking
        chunked_documents = chunking_service.chunk_transcript(video_id, transcript_data)
        if not chunked_documents:
            raise HTTPException(status_code=500, detail="Failed to generate text chunks.")

        # 5. Embedding & Storage
        success = embedding_service.process_and_store(chunked_documents)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store embeddings in database.")

        logger.info(f"Video {video_id} processed successfully.")
        return VideoResponse(
            message="Video processed and indexed successfully.",
            video_id=video_id
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        # Return generic error to client, detailed error in logs
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Intelligent Q&A Orchestration:
    1. Classify user intent (full summary vs specific question)
    2. Route to appropriate handler:
       - Full summary → Hierarchical summarization of entire video
       - Specific query → RAG-based retrieval and answer generation
    """
    try:
        if not request.query or not request.video_id:
            raise HTTPException(status_code=400, detail="Query and Video ID are required.")

        # 1. Classify Intent using lightweight LLM
        intent = intent_classifier.classify(request.query)
        logger.info(f"Query intent classified as: {intent.value}")
        
        # 2. Route based on intent
        if intent == QueryIntent.FULL_VIDEO_SUMMARY:
            # Full video summary: Use hierarchical summarization
            logger.info(f"Processing full video summary request for {request.video_id}")
            summary, sources = summary_service.generate_full_summary(request.video_id)
            
            if not summary:
                return ChatResponse(
                    response="I couldn't generate a summary. The video transcript may not be available.",
                    sources=[]
                )
            
            return ChatResponse(
                response=summary,
                sources=sources
            )
        
        else:
            # Specific query: Use RAG (Retrieval Augmented Generation)
            logger.info(f"Processing specific query for {request.video_id}")
            
            # Retrieval (Semantic Search)
            context_text, sources = retrieval_service.get_context(request.query, request.video_id)
            
            if not context_text:
                logger.warning(f"No context found for video {request.video_id}")
                return ChatResponse(
                    response="I couldn't find any relevant information in this video's transcript to answer your question.",
                    sources=[]
                )

            # Convert Pydantic models to dicts for LLM Service
            history_dicts = [msg.dict() for msg in request.history]

            # Generation (LLM)
            answer = llm_service.generate_answer(
                query=request.query,
                context=context_text,
                chat_history=history_dicts
            )

            return ChatResponse(
                response=answer,
                sources=sources
            )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the response.")

# For local testing via `python main.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)