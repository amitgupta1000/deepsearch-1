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

async def run_workflow(
    initial_query: str,
    prompt_type: str
):
    if workflow_app is None:
        logging.error("LangGraph app is not compiled or imported. Cannot run workflow.")
        print("Workflow cannot be run due to errors in graph compilation or imports.")
        return None
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
        "report": None,
        "report_filename": "IntelliSearchReport",
        "error": None,
        "evaluation_response": None,
        "suggested_follow_up_queries": [],
        "prompt_type": prompt_type,
        "approval_iteration_count": 0,
        "search_iteration_count": 0,
        "report_type": None
    }
    try:
        if config is not None:
            astream = workflow_app.astream(initial_state, config=config)
        else:
            astream = workflow_app.astream(initial_state)
        executed_nodes = []
        async for step in astream:
            for key, value in step.items():
                logging.info("Node executed: %s", key)
                executed_nodes.append(key)
        logging.info("Workflow finished successfully.")
        final_report_filename = initial_state.get("report_filename", "No report file generated.")
        logging.info("Check for report file: %s.txt", final_report_filename)
        final_error_state = initial_state.get('error')
        if final_error_state:
            logging.warning("Workflow completed with errors: %s", final_error_state)
        else:
            logging.info("Workflow completed successfully without errors.")
        return initial_state
    except Exception as e:
        logging.exception(f"An error occurred during workflow execution: {e}")
        return None


import sys
import os
import logging
import uuid
import asyncio
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import json


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "https://deepsearch-56755551-95627.web.app",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

research_sessions: Dict[str, Dict] = {}
MAX_SESSIONS = 10

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
    report_content: Optional[str] = None
    report_filename: Optional[str] = None
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


# Docs endpoint for testing
@app.get("/docs")
async def docs():
    return {"docs_url": app.docs_url, "redoc_url": app.redoc_url}

# Root endpoint
@app.get("/")
async def root():
    return {"message": "INTELLISEARCH API is running", "status": "active", "timestamp": datetime.now().isoformat()}


# POST endpoint to start research
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
        "report_content": None,
        "report_filename": None,
        "error_message": None
    }
    research_sessions[session_id] = session
    # Run pipeline in background
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
        "updated_at": session["updated_at"]
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": get_current_date(),
        "version": "1.0.0",
        "services": {
            "google_ai": bool(GOOGLE_API_KEY),
            "serper_api": bool(SERPER_API_KEY)
        }
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
            "max_query_length": 500
        }
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
        "report_content": session["report_content"],
        "created_at": session["created_at"],
        "completed_at": session["updated_at"]
    }

@app.get("/api/research/{session_id}/download")
async def download_report(session_id: str, content_type: str = "full"):
    if session_id not in research_sessions:
        raise HTTPException(status_code=404, detail="Research session not found")
    session = research_sessions[session_id]
    if session["status"] != "completed" or not session["report_filename"]:
        raise HTTPException(status_code=400, detail="Report not available for download")
    if content_type not in ["full", "appendix"]:
        raise HTTPException(status_code=400, detail="Content type must be 'full' or 'appendix'")
    if content_type == "appendix":
        file_suffix = "_appendix"
        filename_suffix = "appendix"
    else:
        file_suffix = ""
        filename_suffix = "report"
    text_path = os.path.join("..", "..", session["report_filename"])
    if file_suffix:
        file_path = text_path.replace(".txt", f"{file_suffix}.txt")
    else:
        file_path = text_path
    media_type = "text/plain"
    filename = f"intellisearch-{filename_suffix}-{session_id[:8]}.txt"
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=media_type
        )
    else:
        raise HTTPException(status_code=404, detail=f"{filename_suffix.title()} text file not found")

@app.get("/api/research/sessions")
async def list_research_sessions(limit: int = 10, offset: int = 0):
    sessions = list(research_sessions.values())
    sessions.sort(key=lambda x: x["created_at"], reverse=True)
    return {
        "sessions": sessions[offset:offset+limit],
        "total": len(sessions),
        "limit": limit,
        "offset": offset
    }

@app.delete("/api/research/{session_id}")
async def delete_research_session(session_id: str):
    if session_id not in research_sessions:
        raise HTTPException(status_code=404, detail="Research session not found")
    session = research_sessions[session_id]
    if session.get("report_filename"):
        try:
            report_path = os.path.join("..", "..", session["report_filename"])
            if os.path.exists(report_path):
                os.remove(report_path)
        except Exception as e:
            logger.warning(f"Failed to delete report file: {e}")
    del research_sessions[session_id]
    return APIResponse(success=True, message="Research session deleted successfully")

async def run_research_pipeline(session_id: str, request: ResearchRequest):
    try:
        session = research_sessions[session_id]
        session["status"] = "running"
        session["current_step"] = "Starting research pipeline..."
        session["progress"] = 5

        # Use local run_workflow function
        result = None
        try:
            result = await run_workflow(request.query, request.prompt_type)
        except Exception as wf_error:
            session["status"] = "failed"
            session["error_message"] = str(wf_error)
            session["current_step"] = f"Workflow error: {str(wf_error)}"
            session["updated_at"] = datetime.now()
            logger.error(f"Workflow failed for session {session_id}: {wf_error}")
            return

        # After workflow completes
        if result:
            session["report_content"] = result.get("report")
            session["report_filename"] = result.get("report_filename")
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


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("ENVIRONMENT") != "production"
    uvicorn.run(app, host="0.0.0.0", port=port, reload=reload, loop="asyncio")
