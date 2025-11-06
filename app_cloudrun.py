# app_cloudrun.py - INTELLISEARCH Cloud Run Application
# Optimized FastAPI server for Google Cloud Run deployment

import asyncio
import logging
import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime

# FastAPI and server imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="INTELLISEARCH Research API",
    description="AI-powered research and document analysis service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for application state
research_app = None
agent_state_class = None
startup_complete = False

# Store for background tasks
active_tasks: Dict[str, Dict[str, Any]] = {}

async def initialize_application():
    """Initialize the INTELLISEARCH application components."""
    global research_app, agent_state_class, startup_complete
    
    try:
        logger.info("Initializing INTELLISEARCH application...")
        
        # Import and run setup
        import setup
        logger.info("Setup completed successfully")
        
        # Import the LangGraph application
        from src.graph import app as langgraph_app
        research_app = langgraph_app
        
        # Import AgentState
        from src.nodes import AgentState
        agent_state_class = AgentState
        
        # Validate configuration
        from src.config import validate_production_config
        validate_production_config()
        
        startup_complete = True
        logger.info("INTELLISEARCH application initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        startup_complete = False
        raise

@app.on_event("startup")
async def startup_event():
    """FastAPI startup event handler."""
    await initialize_application()

@app.on_event("shutdown")
async def shutdown_event():
    """FastAPI shutdown event handler."""
    logger.info("Shutting down INTELLISEARCH application")

# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run."""
    if not startup_complete:
        raise HTTPException(status_code=503, detail="Application still starting up")
    
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development")
    })

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint for Cloud Run."""
    if not startup_complete or research_app is None:
        raise HTTPException(status_code=503, detail="Application not ready")
    
    return JSONResponse({
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "langgraph_app": research_app is not None,
            "agent_state": agent_state_class is not None
        }
    })

# API Models
from pydantic import BaseModel, Field

class ResearchRequest(BaseModel):
    query: str = Field(..., description="Research query to process")
    prompt_type: str = Field(default="general", description="Type of prompt to use")
    automation_profile: str = Field(default="full", description="Automation level")
    enable_automation: bool = Field(default=True, description="Enable automated processing")

class ResearchResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Research endpoint
@app.post("/research", response_model=ResearchResponse)
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """Start a research task."""
    if not startup_complete or research_app is None:
        raise HTTPException(status_code=503, detail="Application not ready")
    
    # Generate task ID
    task_id = f"task_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{len(active_tasks)}"
    
    # Initialize task tracking
    active_tasks[task_id] = {
        "status": "started",
        "query": request.query,
        "start_time": datetime.utcnow(),
        "progress": "Initializing research workflow...",
        "result": None,
        "error": None
    }
    
    # Start background task
    background_tasks.add_task(run_research_task, task_id, request)
    
    logger.info(f"Started research task {task_id} for query: {request.query}")
    
    return ResearchResponse(
        task_id=task_id,
        status="started",
        message=f"Research task started for query: {request.query}"
    )

async def run_research_task(task_id: str, request: ResearchRequest):
    """Run the research workflow as a background task."""
    try:
        # Update task status
        active_tasks[task_id]["status"] = "running"
        active_tasks[task_id]["progress"] = "Running research workflow..."
        
        # Import RunnableConfig if available
        try:
            from langchain_core.runnables import RunnableConfig
            config = RunnableConfig(recursion_limit=100)
        except Exception:
            config = None
        
        # Define initial state
        initial_state = {
            "new_query": request.query,
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
            "prompt_type": request.prompt_type,
            "approval_iteration_count": 0,
            "search_iteration_count": 0,
            "report_type": None,
            
            # Automation settings
            "non_interactive": request.enable_automation,
            "auto_approve": request.enable_automation,
            "approval_choice": "yes" if request.enable_automation else None,
            "auto_report_type": "unified" if request.enable_automation else None,
            "report_type_choice": "unified" if request.enable_automation else None,
            "new_query_override": None
        }
        
        # Apply automation configuration
        if request.automation_profile == "full":
            initial_state.update({
                "non_interactive": True,
                "auto_approve": True,
                "approval_choice": "yes",
                "auto_report_type": "unified",
                "report_type_choice": "unified"
            })
        
        # Run the workflow
        active_tasks[task_id]["progress"] = "Executing research workflow..."
        
        if config is not None:
            astream = research_app.astream(initial_state, config=config)
        else:
            astream = research_app.astream(initial_state)
        
        # Track execution
        executed_nodes = []
        async for step in astream:
            for key, value in step.items():
                executed_nodes.append(key)
                active_tasks[task_id]["progress"] = f"Executing: {key}"
                logger.info(f"Task {task_id} - Node executed: {key}")
        
        # Task completed successfully
        active_tasks[task_id]["status"] = "completed"
        active_tasks[task_id]["progress"] = "Research completed successfully"
        active_tasks[task_id]["result"] = {
            "executed_nodes": executed_nodes,
            "final_state": {
                "report_filename": initial_state.get("report_filename"),
                "iteration_count": initial_state.get("iteration_count"),
                "search_queries": initial_state.get("search_queries"),
                "error": initial_state.get("error")
            }
        }
        
        logger.info(f"Task {task_id} completed successfully")
        
    except Exception as e:
        # Task failed
        active_tasks[task_id]["status"] = "failed"
        active_tasks[task_id]["error"] = str(e)
        active_tasks[task_id]["progress"] = f"Failed: {str(e)}"
        
        logger.error(f"Task {task_id} failed: {e}")

@app.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get the status of a research task."""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = active_tasks[task_id]
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress"),
        result=task.get("result"),
        error=task.get("error")
    )

@app.get("/tasks")
async def list_tasks():
    """List all active tasks."""
    return {
        "tasks": {
            task_id: {
                "status": task["status"],
                "query": task["query"],
                "start_time": task["start_time"].isoformat(),
                "progress": task.get("progress")
            }
            for task_id, task in active_tasks.items()
        }
    }

# Direct research endpoint for simple use
@app.post("/research/direct")
async def direct_research(request: ResearchRequest):
    """Run research directly and wait for completion (for simple queries)."""
    if not startup_complete or research_app is None:
        raise HTTPException(status_code=503, detail="Application not ready")
    
    try:
        # This is a simplified version for quick queries
        # In production, you might want to add timeout handling
        
        logger.info(f"Starting direct research for query: {request.query}")
        
        # Import and run the workflow (simplified version)
        from src.config import validate_production_config
        validate_production_config()
        
        # Quick research function
        async def quick_research():
            try:
                from langchain_core.runnables import RunnableConfig
                config = RunnableConfig(recursion_limit=100)
            except Exception:
                config = None
                
            initial_state = {
                "new_query": request.query,
                "prompt_type": request.prompt_type,
                "non_interactive": True,
                "auto_approve": True,
                "approval_choice": "yes",
                "auto_report_type": "unified",
                "report_type_choice": "unified",
                # Add other required fields with defaults
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
                "approval_iteration_count": 0,
                "search_iteration_count": 0,
                "report_type": None,
                "new_query_override": None
            }
            
            if config is not None:
                astream = research_app.astream(initial_state, config=config)
            else:
                astream = research_app.astream(initial_state)
            
            executed_nodes = []
            async for step in astream:
                for key, value in step.items():
                    executed_nodes.append(key)
            
            return {
                "query": request.query,
                "executed_nodes": executed_nodes,
                "report_filename": initial_state.get("report_filename"),
                "status": "completed",
                "error": initial_state.get("error")
            }
        
        result = await quick_research()
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Direct research failed: {e}")
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "INTELLISEARCH Research API",
        "version": "1.0.0",
        "status": "ready" if startup_complete else "starting",
        "endpoints": {
            "health": "/health",
            "research": "/research",
            "direct_research": "/research/direct",
            "task_status": "/task/{task_id}",
            "docs": "/docs"
        }
    }

# Main entry point for Cloud Run
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    
    # Configure uvicorn for Cloud Run
    uvicorn.run(
        "app_cloudrun:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
        loop="asyncio",
        # Cloud Run optimizations
        workers=1,  # Single worker for Cloud Run
        timeout_keep_alive=300,  # Keep connections alive
        timeout_notify=300,  # Graceful shutdown timeout
        max_requests=1000,  # Restart worker after N requests
        max_requests_jitter=100,  # Add jitter to max_requests
    )