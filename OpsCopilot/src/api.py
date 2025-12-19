"""
FastAPI application for Ops Copilot.

This module provides the REST API endpoints for incident analysis
using both LangGraph and LangChain implementations.

Endpoints:
- POST /api/v1/analyze - Analyze incident (default: LangGraph)
- POST /api/v1/analyze/langgraph - Analyze using LangGraph
- POST /api/v1/analyze/langchain - Analyze using LangChain
- GET /api/v1/health - Health check
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
from typing import Dict, Any
import asyncio

from .common.config import settings, setup_environment, validate_configuration
from .common.models import AnalyzeRequest, AnalyzeResponse
from .langgraph.workflow import analyze_incident as langgraph_analyze
from .langgraph.workflow import analyze_incident_stream as langgraph_stream
from .langchain.orchestrator import analyze_incident as langchain_analyze


# ============================================================================
# Application Setup
# ============================================================================

# Setup environment variables
setup_environment()

# Validate configuration
try:
    validate_configuration()
except ValueError as e:
    print(f"❌ Configuration error: {e}")
    print("Please check your .env file and ensure all required variables are set.")
    exit(1)

# Create FastAPI app
app = FastAPI(
    title="Ops Copilot API",
    description="AI-powered incident response assistant",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.get("/api/v1/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        Dict with status and version
    
    Example:
        GET /api/v1/health
        
        Response:
        {
            "status": "healthy",
            "version": "1.0.0"
        }
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "llm_model": settings.llm_model
    }


# ============================================================================
# Analysis Endpoints
# ============================================================================

@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_incident_default(request: AnalyzeRequest) -> Dict[str, Any]:
    """
    Analyze an incident using the default implementation (LangGraph).
    
    Args:
        request: AnalyzeRequest with incident description and context
    
    Returns:
        AnalyzeResponse with analysis results
    
    Example:
        POST /api/v1/analyze
        
        Request:
        {
            "incident_description": "API gateway returning 500 errors",
            "context": {
                "service": "api-gateway",
                "environment": "production"
            }
        }
        
        Response:
        {
            "request_id": "req_abc123",
            "timestamp": "2025-12-20T00:00:00Z",
            "latency_ms": 8523,
            "result": {
                "hypothesis": "Database driver incompatibility...",
                "confidence": 0.85,
                "next_actions": [...],
                "commands": [...],
                "citations": [...]
            },
            "metadata": {
                "tools_used": ["log_search", "metrics_query"],
                "runbooks_retrieved": 5,
                "iterations": 1
            }
        }
    """
    try:
        # Use LangGraph by default
        result = await langgraph_analyze(
            incident_description=request.incident_description,
            context=request.context,
            correlation_id=request.incident_id
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/api/v1/analyze/langgraph", response_model=AnalyzeResponse)
async def analyze_incident_langgraph(request: AnalyzeRequest) -> Dict[str, Any]:
    """
    Analyze an incident using LangGraph (state machine approach).
    
    This endpoint uses the LangGraph implementation with state machine,
    conditional routing, and iteration loops.
    
    Args:
        request: AnalyzeRequest with incident description and context
    
    Returns:
        AnalyzeResponse with analysis results
    """
    try:
        result = await langgraph_analyze(
            incident_description=request.incident_description,
            context=request.context,
            correlation_id=request.incident_id
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LangGraph analysis failed: {str(e)}"
        )


@app.post("/api/v1/analyze/langchain", response_model=AnalyzeResponse)
async def analyze_incident_langchain(request: AnalyzeRequest) -> Dict[str, Any]:
    """
    Analyze an incident using pure LangChain (sequential chains approach).
    
    This endpoint uses the LangChain implementation with sequential
    chains and explicit coordination.
    
    Args:
        request: AnalyzeRequest with incident description and context
    
    Returns:
        AnalyzeResponse with analysis results
    """
    try:
        result = await langchain_analyze(
            incident_description=request.incident_description,
            context=request.context,
            correlation_id=request.incident_id
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LangChain analysis failed: {str(e)}"
        )


# ============================================================================
# Streaming Endpoint
# ============================================================================

@app.post("/api/v1/analyze/stream")
async def analyze_incident_streaming(request: AnalyzeRequest):
    """
    Analyze an incident with streaming updates.
    
    This endpoint streams intermediate results as the analysis progresses,
    allowing clients to see real-time updates.
    
    Args:
        request: AnalyzeRequest with incident description and context
    
    Returns:
        StreamingResponse with Server-Sent Events (SSE)
    
    Example:
        POST /api/v1/analyze/stream
        
        Response (SSE):
        data: {"node": "planning", "status": "processing", "data": {...}}
        data: {"node": "tool_execution", "status": "processing", "data": {...}}
        data: {"node": "complete", "status": "done", "data": {...}}
    """
    async def event_generator():
        """Generate Server-Sent Events"""
        try:
            async for update in langgraph_stream(
                incident_description=request.incident_description,
                context=request.context,
                correlation_id=request.incident_id
            ):
                # Format as SSE
                import json
                yield f"data: {json.dumps(update)}\n\n"
                
        except Exception as e:
            error_event = {
                "node": "error",
                "status": "failed",
                "error": str(e)
            }
            import json
            yield f"data: {json.dumps(error_event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Application startup event.
    
    Initializes connections and resources.
    """
    print("=" * 80)
    print("🚀 Ops Copilot API Starting...")
    print("=" * 80)
    print(f"   LLM Model: {settings.llm_model}")
    print(f"   Max Iterations: {settings.max_iterations}")
    print(f"   Request Timeout: {settings.request_timeout}s")
    print(f"   Embedding Model: {settings.embedding_model}")
    print("=" * 80)
    
    # Initialize Pinecone (if needed)
    try:
        from .langgraph.rag import init_pinecone
        init_pinecone()
        print("✅ Pinecone initialized")
    except Exception as e:
        print(f"⚠️  Pinecone initialization failed: {e}")
    
    print("✅ Ops Copilot API ready!")
    print("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown event.
    
    Cleanup resources.
    """
    print("\n" + "=" * 80)
    print("👋 Ops Copilot API shutting down...")
    print("=" * 80)


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )
