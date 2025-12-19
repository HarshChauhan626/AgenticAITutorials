"""
State definition for LangGraph workflow.

This module defines the state schema that is passed between
all nodes in the LangGraph execution.
"""

from typing import TypedDict, List, Dict, Any, Optional
from langchain.schema import BaseMessage


class AgentState(TypedDict):
    """
    Complete state schema for the Ops Copilot LangGraph workflow.
    
    This state object is passed between all nodes and accumulates
    information throughout the execution. Each node can read from
    and write to this state.
    
    State Flow:
    1. Input fields are set initially
    2. Intermediate fields are populated during execution
    3. Output fields contain the final result
    """
    
    # ========================================================================
    # INPUT FIELDS (Set at initialization)
    # ========================================================================
    
    incident_description: str
    """The incident description provided by the user"""
    
    context: Dict[str, str]
    """Additional context like service, environment, severity"""
    
    correlation_id: str
    """Unique ID for tracking this request"""
    
    # ========================================================================
    # ITERATION TRACKING
    # ========================================================================
    
    iteration_count: int
    """Current iteration number (prevents infinite loops)"""
    
    start_time: float
    """Unix timestamp when processing started"""
    
    # ========================================================================
    # TOOL EXECUTION
    # ========================================================================
    
    tools_to_execute: List[str]
    """List of tool names to execute (e.g., ['log_search', 'metrics_query'])"""
    
    tool_results: Dict[str, Any]
    """Results from executed tools, keyed by tool name"""
    
    tools_used: List[str]
    """History of all tools that have been executed"""
    
    # ========================================================================
    # RAG RETRIEVAL
    # ========================================================================
    
    runbooks: List[Dict[str, Any]]
    """Retrieved runbooks from hybrid RAG search"""
    
    # ========================================================================
    # EVIDENCE AGGREGATION
    # ========================================================================
    
    evidence: List[Dict[str, Any]]
    """Aggregated evidence from all sources (logs, metrics, deployments)"""
    
    timeline: List[Dict[str, Any]]
    """Chronological timeline of events"""
    
    # ========================================================================
    # LLM INTERACTION
    # ========================================================================
    
    messages: List[BaseMessage]
    """Chat history for LLM interactions"""
    
    current_hypothesis: Optional[str]
    """Current hypothesis about the root cause"""
    
    confidence: Optional[float]
    """Confidence score for the hypothesis (0.0 to 1.0)"""
    
    reasoning: Optional[str]
    """Detailed reasoning behind the hypothesis"""
    
    # ========================================================================
    # OUTPUT FIELDS (Final result)
    # ========================================================================
    
    final_response: Optional[Dict[str, Any]]
    """The complete final response to return to the user"""
    
    error: Optional[str]
    """Error message if something went wrong"""
    
    # ========================================================================
    # METADATA
    # ========================================================================
    
    metadata: Dict[str, Any]
    """Additional metadata (tokens used, cache hits, etc.)"""


# ============================================================================
# State Initialization Helper
# ============================================================================

def create_initial_state(
    incident_description: str,
    context: Dict[str, str],
    correlation_id: str
) -> AgentState:
    """
    Create an initial state object for a new incident analysis.
    
    This function initializes all required fields with default values
    and sets up the state for the first node in the graph.
    
    Args:
        incident_description: The incident description from the user
        context: Additional context (service, environment, etc.)
        correlation_id: Unique request ID
    
    Returns:
        AgentState: Initialized state object
    
    Example:
        >>> state = create_initial_state(
        ...     incident_description="API returning 500 errors",
        ...     context={"service": "api-gateway"},
        ...     correlation_id="req_abc123"
        ... )
        >>> state["iteration_count"]
        0
    """
    import time
    
    return AgentState(
        # Input
        incident_description=incident_description,
        context=context,
        correlation_id=correlation_id,
        
        # Iteration tracking
        iteration_count=0,
        start_time=time.time(),
        
        # Tool execution
        tools_to_execute=[],
        tool_results={},
        tools_used=[],
        
        # RAG
        runbooks=[],
        
        # Evidence
        evidence=[],
        timeline=[],
        
        # LLM
        messages=[],
        current_hypothesis=None,
        confidence=None,
        reasoning=None,
        
        # Output
        final_response=None,
        error=None,
        
        # Metadata
        metadata={}
    )


# ============================================================================
# State Validation
# ============================================================================

def validate_state(state: AgentState) -> bool:
    """
    Validate that the state has all required fields.
    
    Args:
        state: The state to validate
    
    Returns:
        bool: True if valid, False otherwise
    
    Raises:
        ValueError: If critical fields are missing
    """
    required_fields = [
        "incident_description",
        "context",
        "correlation_id",
        "iteration_count"
    ]
    
    missing = [field for field in required_fields if field not in state]
    
    if missing:
        raise ValueError(f"State missing required fields: {missing}")
    
    return True


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Example: Create initial state
    state = create_initial_state(
        incident_description="API gateway returning 500 errors",
        context={"service": "api-gateway", "environment": "production"},
        correlation_id="req_abc123"
    )
    
    print("✅ Initial state created:")
    print(f"   Incident: {state['incident_description']}")
    print(f"   Service: {state['context']['service']}")
    print(f"   Correlation ID: {state['correlation_id']}")
    print(f"   Iteration: {state['iteration_count']}")
    
    # Validate state
    try:
        validate_state(state)
        print("✅ State is valid")
    except ValueError as e:
        print(f"❌ State validation failed: {e}")
