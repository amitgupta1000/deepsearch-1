import os
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from backend.src.logging_setup import logger
from backend.src.fss_capacity_check import get_fss_storage_usage

# Firestore setup
try:
    from google.cloud import firestore
    db = firestore.Client()
except ImportError:
    db = None
    logger.warning("google-cloud-firestore not installed. Firestore features will be disabled.")
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

app = FastAPI(
    title="CRYSTAL DEEPSEARCH API",
    description="AI-powered research pipeline with web interface",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
        "http://localhost:8000",
        "https://deepsearch-56755551-95627.web.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_SESSIONS = 10

class ResearchRequest(BaseModel):
    query: str = Field(..., description="Research query or question")
    prompt_type: str = Field(default="general", description="Type of prompt to use for query generation.")
    search_mode: str = Field(default="fast", description="Search mode: 'fast' for quick results or 'ultra' for deep research.")
    retrieval_method: str = Field(default="hybrid", description="Retrieval method: 'hybrid' or 'file_search'.")

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

async def run_workflow(initial_query: str, prompt_type: str, search_mode: str, retrieval_method: str, session_id: str) -> Dict[str, Any]:
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

    if retrieval_method == "file_search" or retrieval_method == "file_storage":
        logger.info("Using File Search retrieval method.")
    else:
        logger.info("Using Hybrid retrieval method.")

    # Initial state for the workflow
    initial_state = {
        "session_id": session_id,
        "new_query": initial_query,
        "prompt_type": prompt_type,
        "retrieval_method": retrieval_method,
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
        logger.exception(f"Workflow execution failed: {e}")
        raise
#==============
async def run_research_pipeline(session_id: str, request: ResearchRequest):
    try:
        if db:
            db.collection("research_sessions").document(session_id).update({
                "status": "running",
                "current_step": "Starting research pipeline...",
                "progress": 5,
                "updated_at": datetime.now(),
            })

        result = await run_workflow(request.query, request.prompt_type, request.search_mode, request.retrieval_method, session_id)
        logger.info(f"[run_research_pipeline] Workflow result: {result}")

        # --- Workflow Summary Log ---
        if result:
            retrieval_method = result.get("retrieval_method", "unknown")
            search_mode = "ultra" if result.get("max_search_queries", 0) > 10 else "fast"
            num_qa_pairs = len(result.get("qa_pairs", []))
            error_message = result.get("error")

            summary_lines = [
                "\n\n" + "="*25 + " WORKFLOW SUMMARY " + "="*25,
                f"Session ID:         {session_id}",
                f"Search Mode:        {search_mode.upper()}",
                f"Retrieval Method:   {retrieval_method.replace('_', ' ').title()}",
            ]
            if retrieval_method == "hybrid":
                summary_lines.append(f"Q&A Pairs Created:  {num_qa_pairs}")
            if error_message:
                summary_lines.append(f"Workflow Errors:    Yes (see logs for details)")
            summary_lines.append("="*70 + "\n")
            logger.info('\n'.join(summary_lines))

        session_update = {}
        if result:
            session_update = {
                "analysis_content": result.get("analysis_content"),
                "appendix_content": result.get("appendix_content"),
                "status": "completed",
                "progress": 100,
                "current_step": "Research completed",
                "updated_at": datetime.now(),
                "analysis_filename": result.get("analysis_filename"),
                "appendix_filename": result.get("appendix_filename"),
            }
            logger.info(f"Research session {session_id} completed successfully")
        else:
            session_update = {
                "status": "failed",
                "error_message": "No result returned from workflow.",
                "current_step": "Workflow returned no result.",
                "updated_at": datetime.now(),
            }
            logger.error(f"Workflow returned no result for session {session_id}")
        
        if db:
            db.collection("research_sessions").document(session_id).update(session_update)

    except Exception as e:
        logger.error(f"Research pipeline failed for session {session_id}: {e}")
        if db:
            db.collection("research_sessions").document(session_id).update({
                "status": "failed", "error_message": str(e), "current_step": f"Error: {str(e)}", "updated_at": datetime.now()
            })

#====================

# --- API Endpoints ---
@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"message": "CRYSTAL DEEPSEARCH API is running", "status": "active", "timestamp": datetime.now().isoformat()}

@app.post("/api/research")
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    if db:
        # Optional: Check for active sessions in Firestore to enforce MAX_SESSIONS
        active_sessions = db.collection("research_sessions").where("status", "in", ["running", "pending"]).stream()
        if len(list(active_sessions)) >= MAX_SESSIONS:
            raise HTTPException(status_code=429, detail="Maximum number of active research sessions reached")
    else:
        logger.warning("Firestore not available. Cannot enforce MAX_SESSIONS limit.")

    session_id = str(uuid.uuid4())
    now = datetime.now()
    # Normalize retrieval_method to 'file_search' if user provides 'file_storage'
    normalized_method = request.retrieval_method
    if normalized_method == "file_storage":
        normalized_method = "file_search"
    session_data = {
        # "session_id": session_id, # The document ID is the session_id
        "query": request.query,
        "status": "pending",
        "prompt_type": request.prompt_type,
        "retrieval_method": normalized_method,  # Save retriever choice
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
    if db:
        db.collection("research_sessions").document(session_id).set(session_data)
    else:
        raise HTTPException(status_code=503, detail="Firestore client not available. Cannot start research.")

    background_tasks.add_task(run_research_pipeline, session_id, request)
    return {"session_id": session_id, "status": "started"}

@app.get("/api/research/{session_id}/status")
async def get_research_status(session_id: str):
    if not db:
        raise HTTPException(status_code=503, detail="Firestore client not available")

    doc_ref = db.collection("research_sessions").document(session_id)
    doc = doc_ref.get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="Research session not found")

    session_data = doc.to_dict()
    return {
        "session_id": session_id,
        "status": session_data.get("status"),
        "progress": session_data.get("progress"),
        "current_step": session_data.get("current_step"),
        "updated_at": session_data.get("updated_at"),
    }

@app.get("/api/research/{session_id}/result")
async def get_research_result(session_id: str):
    if not db:
        raise HTTPException(status_code=503, detail="Firestore client not available")

    doc_ref = db.collection("research_sessions").document(session_id)
    doc = doc_ref.get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="Research session not found")
    
    session_data = doc.to_dict()
    if session_data.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Research not completed. Status: {session_data.get('status')}")
    
    return {
        "session_id": session_id,
        "query": session_data.get("query"),
        "analysis_content": session_data.get("analysis_content"),
        "appendix_content": session_data.get("appendix_content"),
        "created_at": session_data.get("created_at"),
        "completed_at": session_data.get("updated_at"),
        "analysis_filename": session_data.get("analysis_filename"),
        "appendix_filename": session_data.get("appendix_filename"),
    }

@app.get("/api/research/sessions")
async def list_research_sessions(limit: int = 10, offset: int = 0):
    """List research sessions with pagination."""
    if not db:
        raise HTTPException(status_code=503, detail="Firestore client not available")
    query = db.collection("research_sessions").order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit).offset(offset)
    sessions = [doc.to_dict() for doc in query.stream()]
    return {"success": True, "data": {"sessions": sessions[offset : offset + limit], "total": len(sessions), "limit": limit, "offset": offset}}

@app.delete("/api/research/{session_id}")
async def delete_research_session(session_id: str):
    """Delete a research session and its associated report files."""
    if not db:
        logger.warning("Firestore client not available. Cannot delete report files from Firestore.")
        raise HTTPException(status_code=503, detail="Firestore client not available")
    session_ref = db.collection("research_sessions").document(session_id)
    session_doc = session_ref.get()
    if not session_doc.exists:
        raise HTTPException(status_code=404, detail="Research session not found")
    session = session_doc.to_dict()
    for key in ["analysis_filename", "appendix_filename"]:
        filename = session.get(key)
        if filename and db:
            try:
                db.collection("report_files").document(filename).delete()
                logger.info(f"Deleted report file from Firestore: {filename}")
            except Exception as e:
                logger.warning(f"Failed to delete file {filename} from Firestore: {e}")
    session_ref.delete()
    return {"success": True, "message": "Research session deleted successfully"}

# --- File Download Endpoint ---
@app.get("/api/download/{filename}")
async def download_file_from_firestore(filename: str):
    """Download a report file directly from Firestore."""
    if not db:
        raise HTTPException(status_code=503, detail="Firestore client not available")
    try:
        doc_ref = db.collection("report_files").document(filename)
        doc = doc_ref.get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="File not found in Firestore")
        file_content = doc.to_dict().get("content")
        return Response(content=file_content, media_type="text/plain", headers={
            'Content-Disposition': f'attachment; filename="{filename}"'
        })
    except Exception as e:
        logger.error(f"Error retrieving file {filename} from Firestore: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving file from Firestore: {str(e)}")

# --- Health Endpoint ---
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": get_current_date(),
        "version": "1.0.0",
        "services": {
            "google_ai": bool(GOOGLE_API_KEY),
            "serper_api": bool(SERPER_API_KEY),
        },
    }



# Config endpoint (merged)
@app.get("/api/config")
async def get_config():
    """Get config info and sources."""
    return {
        "prompt_types": ["general", "legal", "macro", "deepsearch", "person_search", "investment"],
        "limits": {
            "min_words": 500,
            "max_words": 2000,
            "max_query_length": 500,
        },
        "sources": CONFIG_SOURCES,
    }




#=======================================

if __name__ == "__main__":

    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("ENVIRONMENT") != "production"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload, loop="asyncio")
