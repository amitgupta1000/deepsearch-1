# main.py moved to project root # all imports updated to reflect new location

import sys
import os
import logging
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

try:
    from langchain_core.runnables import RunnableConfig
    config = RunnableConfig(recursion_limit=100)
except Exception:
    config = None
from backend.src.graph import app as workflow_app

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI(
    title="INTELLISEARCH API",
    description="AI-powered research pipeline with web interface",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "https://deepsearch-56755551-95627.web.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage
research_sessions: Dict[str, Dict] = {}
MAX_SESSIONS = 10

# --- Pydantic Models ---
class ResearchRequest(BaseModel):
    query: str = Field(..., description="Research query or question")
    prompt_type: str = Field(default="general", description="Prompt type: general, legal, macro, etc.")

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

# --- Workflow Execution ---
async def run_workflow(
    initial_query: str,
    prompt_type: str,
    session_id: str
):
    if workflow_app is None:
        logging.error("LangGraph app is not compiled or imported. Cannot run workflow.")
        print("Workflow cannot be run due to errors in graph compilation or imports.")
        return None
    
    session = research_sessions[session_id]
    logging.info(f"Starting workflow for query: '{initial_query}'")
    initial_state = {
        "new_query": initial_query,
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
        "analysis_filename": None,
        "appendix_filename": None,
        "error": None,
        "evaluation_response": None,
        "suggested_follow_up_queries": [],
        "prompt_type": prompt_type,
        "approval_iteration_count": 0,
        "search_iteration_count": 0,
        "report_type": None
    }
    
    progress_map = {
        "create_queries": 10,
        "evaluate_search_results": 30,
        "extract_content": 50,
        "embed_and_retrieve": 70,
        "create_qa_pairs": 80,
        "AI_evaluate": 90,
        "write_report": 95
    }

    try:
        if config is not None:
            astream = workflow_app.astream(initial_state, config=config)
        else:
            astream = workflow_app.astream(initial_state)
        
        last_state = None
        async for step in astream:
            for key, value in step.items():
                logging.info("Node executed: %s", key)
                session["current_step"] = f"Executing step: {key}"
                session["progress"] = progress_map.get(key, session["progress"])
                session["updated_at"] = datetime.now()
            last_state = step
        logging.info(f"[run_workflow] Final state returned: analysis_content={str(last_state.get('analysis_content', ''))[:200]}")
        logging.info(f"[run_workflow] Final state returned: appendix_content={str(last_state.get('appendix_content', ''))[:200]}")
        return last_state

    except Exception as e:
        logging.exception(f"An error occurred during workflow execution: {e}")
        session["status"] = "failed"
        session["error_message"] = str(e)
        session["current_step"] = f"Workflow error: {str(e)}"
        session["updated_at"] = datetime.now()
        return None

async def run_research_pipeline(session_id: str, request: ResearchRequest):
    try:
        session = research_sessions[session_id]
        session["status"] = "running"
        session["current_step"] = "Starting research pipeline..."
        session["progress"] = 5

        # Use local run_workflow function
        result = await run_workflow(request.query, request.prompt_type, session_id)

        # After workflow completes
        if result:
            logging.info(f"[run_research_pipeline] Setting session['analysis_content']: {str(result.get('analysis_content', ''))[:200]}")
            logging.info(f"[run_research_pipeline] Setting session['appendix_content']: {str(result.get('appendix_content', ''))[:200]}")
            session["analysis_content"] = result.get("analysis_content")
            session["appendix_content"] = result.get("appendix_content")
            session["analysis_filename"] = result.get("analysis_filename")
            session["appendix_filename"] = result.get("appendix_filename")
            session["status"] = "completed"
            session["progress"] = 100
            session["current_step"] = "Research completed"
            session["updated_at"] = datetime.now()
            logger.info(f"Research session {session_id} completed successfully")
        else:
            session["status"] = "failed"
            session["error_message"] = "No result returned from workflow."
            session["current_step"] = "Workflow returned no result."
            session["updated_at"] = datetime.now()
            logger.error(f"Workflow returned no result for session {session_id}")
    except Exception as e:
        logger.error(f"Research pipeline failed for session {session_id}: {e}")
        session = research_sessions.get(session_id)
        if session:
            session["status"] = "failed"
            session["error_message"] = str(e)
            session["current_step"] = f"Error: {str(e)}"
            session["updated_at"] = datetime.now()
            session["status"] = "failed"

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
    }

@app.get("/api/research/{session_id}/download")
async def download_report(session_id: str, content_type: str = "analysis"):
    if session_id not in research_sessions:
        raise HTTPException(status_code=404, detail="Research session not found")
    session = research_sessions[session_id]
    if session["status"] != "completed":
        raise HTTPException(status_code=400, detail="Report not available for download")

    filename_map = {"analysis": "analysis_filename", "appendix": "appendix_filename"}
    if content_type not in filename_map:
        raise HTTPException(status_code=400, detail="Content type must be 'analysis' or 'appendix'")

    file_to_serve = session.get(filename_map[content_type])
    if not file_to_serve:
        raise HTTPException(status_code=404, detail=f"{content_type.title()} file not found")

    file_path = file_to_serve
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            filename=f"CrystalSearch-{content_type}-{session_id[:8]}.txt",
            media_type="text/plain",
        )
    else:
        logger.error(f"File not found at path: {file_path}")
        raise HTTPException(status_code=404, detail=f"{content_type.title()} text file not found")

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

    session = research_sessions.pop(session_id)
    for key in ["analysis_filename", "appendix_filename"]:
        filename = session.get(key)
        if filename and os.path.exists(filename):
            try:
                os.remove(filename)
                logger.info(f"Deleted report file: {filename}")
            except Exception as e:
                logger.warning(f"Failed to delete file {filename}: {e}")

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

@app.post("/api/research/test-workflow")
async def test_research_workflow(request: ResearchRequest):
    """
    Runs the research workflow synchronously for testing purposes and returns the final state.
    """
    initial_state = {
        "new_query": request.query,
        "prompt_type": request.prompt_type,
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
        "analysis_filename": "TestReport_analysis.txt",
        "appendix_filename": "TestReport_appendix.txt",
        "error": None,
        "evaluation_response": None,
        "suggested_follow_up_queries": [],
        "approval_iteration_count": 0,
        "search_iteration_count": 0,
        "report_type": None
    }
    
    final_state = None
    try:
        if config is not None:
            astream = workflow_app.astream(initial_state, config=config)
        else:
            astream = workflow_app.astream(initial_state)

        # Aynchronously iterate through the stream to execute the graph
        async for step in astream:
            # log steps for visibility during run
            for key, value in step.items():
                logger.info(f"Executing node: {key}")
        
        # Get the final state
        final_state = await astream.get_final_output()

    except Exception as e:
        logger.exception(f"An error occurred during workflow test execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if final_state:
        return final_state
    else:
        raise HTTPException(status_code=500, detail="Workflow did not produce a final state.")

# --- Main Execution Block ---
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("ENVIRONMENT") != "production"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload, loop="asyncio")
