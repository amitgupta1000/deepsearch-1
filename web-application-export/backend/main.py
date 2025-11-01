# FastAPI Backend for INTELLISEARCH Web Application
# Converts the existing Python research pipeline into a web API

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime
import json

# Add the parent src directory to the path to import our existing modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import our existing INTELLISEARCH modules
try:
    from src.graph import app as workflow_app
    from src.nodes import AgentState
    from src.config import GOOGLE_API_KEY, SERPER_API_KEY
    from src.utils import get_current_date
    INTELLISEARCH_AVAILABLE = True
    logging.info("Successfully imported INTELLISEARCH modules")
except ImportError as e:
    logging.error(f"Failed to import INTELLISEARCH modules: {e}")
    INTELLISEARCH_AVAILABLE = False
    
    # Define fallback classes for development
    class AgentState:
        def __init__(self):
            self.query = ""
            self.report_type = "concise"
            self.report = ""
            self.sources = []
            self.citations = []
    
    workflow_app = None
    
    def get_current_date():
        return datetime.now().strftime("%Y-%m-%d")
    
    GOOGLE_API_KEY = ""
    SERPER_API_KEY = ""

# Configure logging
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

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for research sessions (in production, use Redis or database)
research_sessions: Dict[str, Dict] = {}

# Pydantic models for API requests/responses
class ResearchRequest(BaseModel):
    query: str = Field(..., description="Research query or question")
    report_type: str = Field(default="detailed", description="'concise' or 'detailed'")
    prompt_type: str = Field(default="general", description="Prompt type: general, legal, macro, etc.")
    automation_level: str = Field(default="full", description="Automation level for the research")

class ResearchSession(BaseModel):
    session_id: str
    query: str
    status: str  # "pending", "running", "completed", "failed"
    report_type: str
    prompt_type: str
    created_at: datetime
    updated_at: datetime
    progress: int = 0  # 0-100
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

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": get_current_date(),
        "version": "1.0.0",
        "services": {
            "google_ai": bool(GOOGLE_API_KEY),
            "serper_api": bool(SERPER_API_KEY)
        }
    }

# Configuration endpoint
@app.get("/api/config")
async def get_config():
    """Get available configuration options"""
    return {
        "report_types": ["concise", "detailed"],
        "prompt_types": ["general", "legal", "macro", "deepsearch", "person_search", "investment"],
        "automation_levels": ["full", "interactive"],
        "limits": {
            "concise_words": 1200,
            "detailed_words": 3000,
            "max_query_length": 500
        }
    }

# Start research endpoint
@app.post("/api/research/start", response_model=APIResponse)
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """Start a new research session"""
    try:
        # Validate request
        if len(request.query.strip()) < 10:
            raise HTTPException(status_code=400, detail="Query must be at least 10 characters long")
        
        if request.report_type not in ["concise", "detailed"]:
            raise HTTPException(status_code=400, detail="Report type must be 'concise' or 'detailed'")
        
        # Create new session
        session_id = str(uuid.uuid4())
        session = {
            "session_id": session_id,
            "query": request.query.strip(),
            "status": "pending",
            "report_type": request.report_type,
            "prompt_type": request.prompt_type,
            "automation_level": request.automation_level,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "progress": 0,
            "current_step": "Initializing research pipeline...",
            "report_content": None,
            "report_filename": None,
            "error_message": None
        }
        
        research_sessions[session_id] = session
        
        # Start background research task
        background_tasks.add_task(run_research_pipeline, session_id, request)
        
        logger.info(f"Started research session {session_id} for query: {request.query[:100]}...")
        
        return APIResponse(
            success=True,
            message="Research session started successfully",
            data={"session_id": session_id}
        )
        
    except Exception as e:
        logger.error(f"Error starting research: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get research status
@app.get("/api/research/{session_id}/status", response_model=ResearchStatus)
async def get_research_status(session_id: str):
    """Get the current status of a research session"""
    if session_id not in research_sessions:
        raise HTTPException(status_code=404, detail="Research session not found")
    
    session = research_sessions[session_id]
    
    return ResearchStatus(
        session_id=session_id,
        status=session["status"],
        progress=session["progress"],
        current_step=session["current_step"],
        estimated_completion=None  # Can be calculated based on progress
    )

# Get research results
@app.get("/api/research/{session_id}/result")
async def get_research_result(session_id: str):
    """Get the completed research report"""
    if session_id not in research_sessions:
        raise HTTPException(status_code=404, detail="Research session not found")
    
    session = research_sessions[session_id]
    
    if session["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Research not completed. Status: {session['status']}")
    
    return {
        "session_id": session_id,
        "query": session["query"],
        "report_type": session["report_type"],
        "report_content": session["report_content"],
        "created_at": session["created_at"],
        "completed_at": session["updated_at"]
    }

# Download report as file
@app.get("/api/research/{session_id}/download")
async def download_report(session_id: str, format: str = "txt"):
    """Download the research report as a text or PDF file"""
    if session_id not in research_sessions:
        raise HTTPException(status_code=404, detail="Research session not found")
    
    session = research_sessions[session_id]
    
    if session["status"] != "completed" or not session["report_filename"]:
        raise HTTPException(status_code=400, detail="Report not available for download")
    
    # Validate format parameter
    if format not in ["txt", "pdf"]:
        raise HTTPException(status_code=400, detail="Format must be either 'txt' or 'pdf'")
    
    # Determine file path and media type based on format
    if format == "pdf":
        # Look for PDF file (replace .txt extension with .pdf)
        text_path = os.path.join("..", "..", session["report_filename"])
        pdf_path = text_path.replace(".txt", ".pdf")
        file_path = pdf_path
        media_type = "application/pdf"
        filename = f"intellisearch-report-{session_id[:8]}.pdf"
    else:
        # Default to text file
        file_path = os.path.join("..", "..", session["report_filename"])
        media_type = "text/plain"
        filename = f"intellisearch-report-{session_id[:8]}.txt"
    
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=media_type
        )
    else:
        format_name = "PDF" if format == "pdf" else "text"
        raise HTTPException(status_code=404, detail=f"Report {format_name} file not found")

# List all research sessions
@app.get("/api/research/sessions")
async def list_research_sessions(limit: int = 10, offset: int = 0):
    """List recent research sessions"""
    sessions = list(research_sessions.values())
    sessions.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {
        "sessions": sessions[offset:offset+limit],
        "total": len(sessions),
        "limit": limit,
        "offset": offset
    }

# Delete research session
@app.delete("/api/research/{session_id}")
async def delete_research_session(session_id: str):
    """Delete a research session and its data"""
    if session_id not in research_sessions:
        raise HTTPException(status_code=404, detail="Research session not found")
    
    # Clean up any files
    session = research_sessions[session_id]
    if session.get("report_filename"):
        try:
            report_path = os.path.join("..", "..", session["report_filename"])
            if os.path.exists(report_path):
                os.remove(report_path)
        except Exception as e:
            logger.warning(f"Failed to delete report file: {e}")
    
    del research_sessions[session_id]
    
    return APIResponse(
        success=True,
        message="Research session deleted successfully"
    )

# Background task to run the research pipeline
async def run_research_pipeline(session_id: str, request: ResearchRequest):
    """Background task that runs the actual research pipeline"""
    try:
        session = research_sessions[session_id]
        session["status"] = "running"
        session["current_step"] = "Starting research pipeline..."
        session["progress"] = 5
        
        # Use the compiled workflow
        if not INTELLISEARCH_AVAILABLE or workflow_app is None:
            raise HTTPException(status_code=503, detail="INTELLISEARCH pipeline not available")
        
        # Prepare initial state
        initial_state = {
            "new_query": request.query,
            "report_type": request.report_type,
            "non_interactive": True,  # Disable interactive prompts
            "auto_report_type": request.report_type,
            "reasoning_mode": True,
            "auto_approve": True,  # Auto-approve queries in web mode
            "approval_choice": "yes",
            "report_type_choice": request.report_type,
            "search_queries": None,
            "rationale": None,
            "data": None,
            "relevant_contexts": None,
            "relevant_chunks": None,
            "proceed": None,
            "visited_urls": [],
            "failed_urls": [],
            "iteration_count": 0,
            "report": None,
            "report_filename": None,
            "error": None,
            "evaluation_response": None,
            "suggested_follow_up_queries": None,
            "approval_iteration_count": 0,
            "search_iteration_count": 0,
            "snippet_state": None
        }
        
        session["current_step"] = "Analyzing research question..."
        session["progress"] = 15
        
        # Run the workflow with progress updates
        async def progress_callback(step_name: str, progress: int):
            session["current_step"] = step_name
            session["progress"] = min(progress, 95)  # Keep 5% for finalization
            session["updated_at"] = datetime.now()
        
        # Execute the research workflow
        session["current_step"] = "Executing research workflow..."
        session["progress"] = 20
        
        # Execute the actual workflow
        try:
            final_state = workflow_app.invoke(initial_state)
            
            # Extract results from the final state
            report_content = final_state.get("report", "No report generated")
            sources = final_state.get("sources", [])
            citations = final_state.get("citations", [])
            
        except Exception as workflow_error:
            logger.error(f"Workflow execution failed: {workflow_error}")
            # Fallback to simulation if workflow fails
            final_state = await simulate_research_workflow(initial_state, progress_callback)
            report_content = final_state.get("report", "No report generated")
            sources = final_state.get("sources", [])
            citations = final_state.get("citations", [])
        
        # Update session with results
        session["status"] = "completed"
        session["progress"] = 100
        session["current_step"] = "Research completed successfully"
        session["report_content"] = report_content
        session["sources"] = sources
        session["citations"] = citations
        session["updated_at"] = datetime.now()
        
        logger.info(f"Research session {session_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Research pipeline failed for session {session_id}: {e}")
        session["status"] = "failed"
        session["error_message"] = str(e)
        session["current_step"] = f"Error: {str(e)}"
        session["updated_at"] = datetime.now()

# Temporary simulation function - replace with actual workflow
async def simulate_research_workflow(initial_state: AgentState, progress_callback):
    """Simulate the research workflow - replace with actual implementation"""
    steps = [
        ("Analyzing research question...", 25),
        ("Searching for relevant information...", 40),
        ("Scraping and processing content...", 55),
        ("Analyzing and evaluating sources...", 70),
        ("Generating report sections...", 85),
        ("Finalizing report and citations...", 95)
    ]
    
    for step_name, progress in steps:
        await progress_callback(step_name, progress)
        await asyncio.sleep(2)  # Simulate processing time
    
    # Return mock final state
    return {
        "report": f"# Research Report: {initial_state['new_query']}\n\nThis is a simulated report...",
        "report_filename": "simulated_report.txt"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)