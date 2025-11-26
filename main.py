import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

import sys
import logging
import uuid # Added import
from datetime import datetime # Added import

# Firestore setup
try:
    from google.cloud import firestore
    db = firestore.Client()
except ImportError:
    db = None
    logging.warning("google-cloud-firestore not installed. Firestore features will be disabled.")
from backend.src.config import CONFIG_SOURCES

import uvicorn

try:
    from langchain_core.runnables import RunnableConfig
    config = RunnableConfig(recursion_limit=100)
except Exception:
    config = None

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="INTELLISEARCH API",
    description="AI-powered research pipeline with web interface",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "https://deepsearch-56755551-95627.web.app",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

research_sessions: Dict[str, Dict] = {}
MAX_SESSIONS = 10

class ResearchRequest(BaseModel):
    query: str = Field(..., description="Research query or question")
    prompt_type: str = Field(default="general", description="Type of prompt to use for query generation.")
    search_mode: str = Field(default="fast", description="Search mode: 'fast' for quick results or 'ultra' for deep research.")

class ResearchSession(BaseModel):
    session_id: str
    query: str
    status: str
    prompt_type: str
    created_at: datetime
    updated_at: datetime
    progress: int = 0
    current_step: str = ""
    analysis_content: Optional[str] = None
    appendix_content: Optional[str] = None
    analysis_filename: Optional[str] = None
    appendix_filename: Optional[str] = None
    error_message: Optional[str] = None

class ResearchStatus(BaseModel):
    session_id: str
    status: str
    progress: int
    current_step: str
    estimated_completion: Optional[datetime] = None

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None


#==============
from backend.src.graph import app as workflow_app

async def run_workflow(initial_query: str, prompt_type: str, search_mode: str, session_id: str):
    if workflow_app is None:
        raise RuntimeError("Workflow not compiled. LangGraph not available.")

    # Import config values for search modes
    from backend.src.config import (
        MAX_SEARCH_QUERIES, MAX_SEARCH_RESULTS, MAX_AI_ITERATIONS,
        ULTRA_MAX_SEARCH_QUERIES, ULTRA_MAX_SEARCH_RESULTS, ULTRA_MAX_AI_ITERATIONS
    )

    # Determine settings based on search_mode
    if search_mode == "ultra":
        max_queries = ULTRA_MAX_SEARCH_QUERIES
        max_results = ULTRA_MAX_SEARCH_RESULTS
        max_iterations = ULTRA_MAX_AI_ITERATIONS
        logger.info(f"Running in 'ultra' mode: {max_queries} queries, {max_results} results, {max_iterations} iterations.")
    else: # Default to "fast"
        max_queries = MAX_SEARCH_QUERIES
        max_results = MAX_SEARCH_RESULTS
        max_iterations = MAX_AI_ITERATIONS
        logger.info(f"Running in 'fast' mode: {max_queries} queries, {max_results} results, {max_iterations} iterations.")

    # Initial state for the workflow
    initial_state = {
        "session_id": session_id,
        "new_query": initial_query,
        "prompt_type": prompt_type,
        "search_queries": [],
        "rationale": None,
        "data": [],
        "relevant_contexts": {},
        "relevant_chunks": [],
        "proceed": True,
        "visited_urls": [],
        "failed_urls": [],
        "iteration_count": 0,
        "analysis_content": None,
        "appendix_content": None,
        "error": None,
        "evaluation_response": None,
        "suggested_follow_up_queries": [],
        "approval_iteration_count": 0,
        "search_iteration_count": 0,
        "report_type": None,
        # Dynamic configuration based on search_mode
        "max_search_queries": max_queries,
        "max_search_results": max_results,
        "max_ai_iterations": max_iterations,
    }

    try:
        # Direct one-shot execution
        final_state = await workflow_app.ainvoke(initial_state)
        return final_state
    except Exception as e:
        logging.exception(f"Workflow execution failed: {e}")
        raise
#==============
async def run_research_pipeline(session_id: str, request: ResearchRequest):
    try:
        research_sessions[session_id]["status"] = "running"
        research_sessions[session_id]["current_step"] = "Starting research pipeline..."
        research_sessions[session_id]["progress"] = 5

        result = await run_workflow(request.query, request.prompt_type, request.search_mode, session_id)
        logging.info(f"[run_research_pipeline] Workflow result: {result}")

        if result:
            research_sessions[session_id].update({
                "analysis_content": result.get("analysis_content"),
                "appendix_content": result.get("appendix_content"),
                "status": "completed",
                "progress": 100,
                "current_step": "Research completed",
                "updated_at": datetime.now(),
                # The workflow already saved the reports and returned the filenames.
                "analysis_filename": result.get("analysis_filename"),
                "appendix_filename": result.get("appendix_filename"),
            })
            logger.info(f"Research session {session_id} completed successfully")
        else:
            research_sessions[session_id].update({
                "status": "failed",
                "error_message": "No result returned from workflow.",
                "current_step": "Workflow returned no result.",
                "updated_at": datetime.now(),
            })
            logger.error(f"Workflow returned no result for session {session_id}")
    except Exception as e:
        logger.error(f"Research pipeline failed for session {session_id}: {e}")
        session = research_sessions.get(session_id)
        if session:
            session["status"] = "failed"
            session["error_message"] = str(e)
            session["current_step"] = f"Error: {str(e)}"
            session["updated_at"] = datetime.now()

#====================
# --- API Endpoints ---
@app.get("/")
async def root():
    return {
        "message": "CRYSTAL DEEPSEARCH API is running",
        "status": "active",
        "timestamp": datetime.now().isoformat(),
    }

@app.post("/api/research")
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    if len(research_sessions) >= MAX_SESSIONS:
        raise HTTPException(status_code=429, detail="Maximum number of research sessions reached")

    session_id = str(uuid.uuid4())
    now = datetime.now()
    session = {
        "session_id": session_id,
        "query": request.query,
        "status": "pending",
        "prompt_type": request.prompt_type,
        "created_at": now,
        "updated_at": now,
        "progress": 0,
        "current_step": "Queued",
        "analysis_content": None,
        "appendix_content": None,
        "analysis_filename": None,
        "appendix_filename": None,
        "error_message": None,
    }
    research_sessions[session_id] = session
    background_tasks.add_task(run_research_pipeline, session_id, request)
    return {"session_id": session_id, "status": "started"}

@app.get("/api/research/{session_id}/status")
async def get_research_status(session_id: str):
    if session_id not in research_sessions:
        raise HTTPException(status_code=404, detail="Research session not found")
    session = research_sessions[session_id]
    return {
        "session_id": session_id,
        "status": session["status"],
        "progress": session["progress"],
        "current_step": session["current_step"],
        "updated_at": session["updated_at"],
    }

@app.get("/api/research/{session_id}/result")
async def get_research_result(session_id: str):
    if session_id not in research_sessions:
        raise HTTPException(status_code=404, detail="Research session not found")
    session = research_sessions[session_id]
    if session["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Research not completed. Status: {session['status']}")
    return {
        "session_id": session_id,
        "query": session["query"],
        "analysis_content": session["analysis_content"],
        "appendix_content": session["appendix_content"],
        "created_at": session["created_at"],
        "completed_at": session["updated_at"],
        "analysis_filename": session.get("analysis_filename"),
        "appendix_filename": session.get("appendix_filename"),
    }

@app.get("/api/research/sessions")
async def list_research_sessions(limit: int = 10, offset: int = 0):
    sessions = sorted(list(research_sessions.values()), key=lambda x: x["created_at"], reverse=True)
    return {
        "sessions": sessions[offset : offset + limit],
        "total": len(sessions),
        "limit": limit,
        "offset": offset,
    }

@app.delete("/api/research/{session_id}")
async def delete_research_session(session_id: str):
    if session_id not in research_sessions:
        raise HTTPException(status_code=404, detail="Research session not found")

    if not db:
        logger.warning("Firestore client not available. Cannot delete report files from Firestore.")

    session = research_sessions.pop(session_id)
    for key in ["analysis_filename", "appendix_filename"]:
        filename = session.get(key)
        if filename and db:
            try:
                db.collection("report_files").document(filename).delete()
                logger.info(f"Deleted report file from Firestore: {filename}")
            except Exception as e:
                logger.warning(f"Failed to delete file {filename} from Firestore: {e}")

    return APIResponse(success=True, message="Research session deleted successfully")

# --- Health & Debug Endpoints ---
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": get_current_date(),
        "version": "1.0.0",
        "services": {
            "google_ai": bool(GOOGLE_API_KEY),
            "serper_api": bool(SERPER_API_KEY),
        },
    }

@app.get("/api/debug")
async def debug_info():
    return {
        "google_api_key_set": bool(GOOGLE_API_KEY),
        "serper_api_key_set": bool(SERPER_API_KEY),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "python_path": sys.path[-3:],
    }


# Existing config info endpoint
@app.get("/api/config")
async def get_config():
    return {
        "prompt_types": ["general", "legal", "macro", "deepsearch", "person_search", "investment"],
        "limits": {
            "min_words": 500,
            "max_words": 2000,
            "max_query_length": 500,
        },
    }

# New endpoint: config values and sources
@app.get("/api/config/values")
async def get_config_values():
    """
    Returns all config keys, their values, and source (env or default).
    """
    return CONFIG_SOURCES


# --- Firestore Retrieval Endpoint ---
@app.get("/api/research/{session_id}/firestore_result")
async def get_firestore_result(session_id: str):
    if not db:
        raise HTTPException(status_code=503, detail="Firestore client not available")
    try:
        doc_ref = db.collection("research_reports").document(session_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        else:
            raise HTTPException(status_code=404, detail="Report not found in Firestore")
    except Exception as e:
        logger.error(f"Error retrieving report from Firestore: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving report from Firestore")

#=======================================

if __name__ == "__main__":

    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("ENVIRONMENT") != "production"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload, loop="asyncio")
