"""
LangGraph workflow construction and execution.

This module builds the complete LangGraph state machine and provides
the main entry point for incident analysis using LangGraph.

Graph Structure:
    parse_input → planning → tool_execution → rag_retrieval →
    evidence_aggregation → reasoning → decision →
    [continue → planning] OR [generate → response_generation] → END
"""

import uuid
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import AgentState, create_initial_state
from .nodes import (
    parse_input_node,
    planning_node,
    tool_execution_node,
    rag_retrieval_node,
    evidence_aggregation_node,
    reasoning_node,
    should_continue,
    response_generation_node
)
from ..common.config import settings


# ============================================================================
# Graph Construction
# ============================================================================

def create_ops_copilot_graph(checkpointer=None):
    """
    Create the complete LangGraph workflow for Ops Copilot.
    
    This function builds the state machine that orchestrates the entire
    incident analysis process. The graph includes all nodes and edges
    that define the execution flow.
    
    Args:
        checkpointer: Optional checkpointer for state persistence
                     (default: in-memory SQLite)
    
    Returns:
        Compiled LangGraph application ready for execution
    
    Graph Flow:
    1. parse_input: Initialize state and create initial message
    2. planning: Decide which tools to execute (LLM)
    3. tool_execution: Run tools in parallel
    4. rag_retrieval: Retrieve runbooks using hybrid RAG
    5. evidence_aggregation: Combine all evidence
    6. reasoning: Generate hypothesis (LLM)
    7. decision: Check if we should continue or generate response
       - If continue: Loop back to planning for more tools
       - If generate: Proceed to response generation
    8. response_generation: Create final structured response (LLM)
    
    Features:
    - Iteration limits (max 5 loops)
    - Conditional routing based on confidence
    - State checkpointing for debugging
    - Timeout protection
    
    Example:
        >>> app = create_ops_copilot_graph()
        >>> result = app.invoke(initial_state)
    """
    # Initialize the graph with state schema
    workflow = StateGraph(AgentState)
    
    # ========================================================================
    # Add Nodes
    # ========================================================================
    
    # Node 1: Parse input and initialize state
    workflow.add_node("parse_input", parse_input_node)
    
    # Node 2: Planning - decide which tools to use (LLM call)
    workflow.add_node("planning", planning_node)
    
    # Node 3: Tool execution - run tools in parallel
    workflow.add_node("tool_execution", tool_execution_node)
    
    # Node 4: RAG retrieval - get relevant runbooks
    workflow.add_node("rag_retrieval", rag_retrieval_node)
    
    # Node 5: Evidence aggregation - combine all sources
    workflow.add_node("evidence_aggregation", evidence_aggregation_node)
    
    # Node 6: Reasoning - generate hypothesis (LLM call)
    workflow.add_node("reasoning", reasoning_node)
    
    # Node 7: Response generation - create final output (LLM call)
    workflow.add_node("response_generation", response_generation_node)
    
    # ========================================================================
    # Define Edges (Execution Flow)
    # ========================================================================
    
    # Set entry point
    workflow.set_entry_point("parse_input")
    
    # Linear flow through initial nodes
    workflow.add_edge("parse_input", "planning")
    workflow.add_edge("planning", "tool_execution")
    workflow.add_edge("tool_execution", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "evidence_aggregation")
    workflow.add_edge("evidence_aggregation", "reasoning")
    
    # Conditional edge: decide whether to continue or generate response
    workflow.add_conditional_edges(
        "reasoning",  # Source node
        should_continue,  # Decision function
        {
            "continue": "planning",  # Loop back for more iterations
            "generate": "response_generation"  # Proceed to final response
        }
    )
    
    # Final edge to END
    workflow.add_edge("response_generation", END)
    
    # ========================================================================
    # Compile Graph
    # ========================================================================
    
    # Create checkpointer if not provided
    if checkpointer is None:
        # Use in-memory SQLite for checkpointing
        checkpointer = SqliteSaver.from_conn_string(":memory:")
    
    # Compile the graph with checkpointing
    app = workflow.compile(checkpointer=checkpointer)
    
    return app


# ============================================================================
# Main Execution Function
# ============================================================================

async def analyze_incident(
    incident_description: str,
    context: Dict[str, str],
    correlation_id: str = None
) -> Dict[str, Any]:
    """
    Analyze an incident using the LangGraph workflow.
    
    This is the main entry point for incident analysis. It creates
    the initial state, executes the graph, and returns the final result.
    
    Args:
        incident_description: Description of the incident
        context: Additional context (service, environment, severity)
        correlation_id: Optional request ID (auto-generated if not provided)
    
    Returns:
        Dict containing the final analysis result
    
    Example:
        >>> result = await analyze_incident(
        ...     incident_description="API returning 500 errors",
        ...     context={"service": "api-gateway", "environment": "production"}
        ... )
        >>> print(result["hypothesis"])
        "Database driver incompatibility..."
    
    Process:
    1. Generate correlation ID if not provided
    2. Create initial state
    3. Create graph
    4. Execute graph with state
    5. Extract final response
    6. Return result
    """
    import time
    
    # Generate correlation ID if not provided
    if correlation_id is None:
        correlation_id = f"req_{uuid.uuid4().hex[:12]}"
    
    # Create initial state
    initial_state = create_initial_state(
        incident_description=incident_description,
        context=context,
        correlation_id=correlation_id
    )
    
    # Create graph
    app = create_ops_copilot_graph()
    
    # Configuration for execution
    config = {
        "configurable": {
            "thread_id": correlation_id
        }
    }
    
    # Track start time
    start_time = time.time()
    
    # Execute graph
    # Note: For async execution, use app.ainvoke()
    final_state = await app.ainvoke(initial_state, config)
    
    # Calculate total latency
    latency_ms = int((time.time() - start_time) * 1000)
    
    # Extract final response
    final_response = final_state.get("final_response", {})
    
    # Build complete response
    result = {
        "request_id": correlation_id,
        "timestamp": time.time(),
        "latency_ms": latency_ms,
        "result": final_response,
        "metadata": {
            "tools_used": final_state.get("tools_used", []),
            "runbooks_retrieved": len(final_state.get("runbooks", [])),
            "iterations": final_state.get("iteration_count", 0),
            "cache_hit": False
        }
    }
    
    return result


# ============================================================================
# Streaming Execution
# ============================================================================

async def analyze_incident_stream(
    incident_description: str,
    context: Dict[str, str],
    correlation_id: str = None
):
    """
    Analyze an incident with streaming updates.
    
    This function streams intermediate results as the graph executes,
    allowing the client to see progress in real-time.
    
    Args:
        incident_description: Description of the incident
        context: Additional context
        correlation_id: Optional request ID
    
    Yields:
        Dict with node name and current state
    
    Example:
        >>> async for update in analyze_incident_stream(...):
        ...     print(f"Node: {update['node']}, Status: {update['status']}")
    
    Streamed Updates:
    - Node name (e.g., "planning", "tool_execution")
    - Current status
    - Intermediate data (tools selected, hypothesis, etc.)
    """
    # Generate correlation ID if not provided
    if correlation_id is None:
        correlation_id = f"req_{uuid.uuid4().hex[:12]}"
    
    # Create initial state
    initial_state = create_initial_state(
        incident_description=incident_description,
        context=context,
        correlation_id=correlation_id
    )
    
    # Create graph
    app = create_ops_copilot_graph()
    
    # Configuration
    config = {
        "configurable": {
            "thread_id": correlation_id
        }
    }
    
    # Stream execution
    async for event in app.astream(initial_state, config):
        # Extract node name and output
        node_name = list(event.keys())[0]
        node_output = event[node_name]
        
        # Yield update
        yield {
            "node": node_name,
            "status": "processing",
            "data": {
                "iteration": node_output.get("iteration_count", 0),
                "tools_used": node_output.get("tools_used", []),
                "hypothesis": node_output.get("current_hypothesis"),
                "confidence": node_output.get("confidence")
            }
        }
    
    # Final update
    yield {
        "node": "complete",
        "status": "done",
        "data": {
            "final_response": node_output.get("final_response")
        }
    }


# ============================================================================
# Graph Visualization
# ============================================================================

def visualize_graph(output_path: str = "graph.png"):
    """
    Generate a visual representation of the graph.
    
    This creates a diagram showing all nodes and edges in the workflow,
    useful for documentation and debugging.
    
    Args:
        output_path: Path to save the graph image
    
    Example:
        >>> visualize_graph("opscopilot_graph.png")
    """
    # Create graph
    app = create_ops_copilot_graph()
    
    # Get graph visualization
    graph_image = app.get_graph().draw_mermaid_png()
    
    # Save to file
    with open(output_path, "wb") as f:
        f.write(graph_image)
    
    print(f"✅ Graph visualization saved to {output_path}")


# ============================================================================
# Debug Execution
# ============================================================================

async def debug_analyze(
    incident_description: str,
    context: Dict[str, str]
):
    """
    Run analysis with detailed debug output.
    
    This function prints detailed information about each node execution,
    useful for debugging and understanding the workflow.
    
    Args:
        incident_description: Description of the incident
        context: Additional context
    """
    print("=" * 80)
    print("DEBUG MODE: Ops Copilot Analysis")
    print("=" * 80)
    
    # Create initial state
    correlation_id = f"debug_{uuid.uuid4().hex[:8]}"
    
    print(f"\n📋 Request Details:")
    print(f"   Correlation ID: {correlation_id}")
    print(f"   Incident: {incident_description}")
    print(f"   Service: {context.get('service', 'unknown')}")
    print(f"   Environment: {context.get('environment', 'unknown')}")
    
    # Stream execution with debug output
    async for update in analyze_incident_stream(
        incident_description=incident_description,
        context=context,
        correlation_id=correlation_id
    ):
        node = update["node"]
        status = update["status"]
        data = update["data"]
        
        print(f"\n{'='*80}")
        print(f"🔵 Node: {node.upper()}")
        print(f"{'='*80}")
        
        if node == "planning":
            print(f"   Iteration: {data.get('iteration', 0)}")
            print(f"   Tools selected: {data.get('tools_used', [])}")
        
        elif node == "reasoning":
            print(f"   Hypothesis: {data.get('hypothesis', 'N/A')[:100]}...")
            print(f"   Confidence: {data.get('confidence', 0):.2f}")
        
        elif node == "complete":
            print(f"\n✅ ANALYSIS COMPLETE")
            final = data.get('final_response', {})
            print(f"   Final Hypothesis: {final.get('hypothesis', 'N/A')}")
            print(f"   Confidence: {final.get('confidence', 0):.2f}")
            print(f"   Actions: {len(final.get('next_actions', []))}")


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    # Example 1: Basic analysis
    async def example_basic():
        result = await analyze_incident(
            incident_description="API gateway returning 500 errors since 2pm",
            context={
                "service": "api-gateway",
                "environment": "production",
                "severity": "critical"
            }
        )
        
        print("✅ Analysis complete:")
        print(f"   Request ID: {result['request_id']}")
        print(f"   Latency: {result['latency_ms']}ms")
        print(f"   Hypothesis: {result['result']['hypothesis']}")
    
    # Example 2: Streaming analysis
    async def example_streaming():
        async for update in analyze_incident_stream(
            incident_description="Database connection timeouts",
            context={"service": "api-gateway"}
        ):
            print(f"Update: {update['node']} - {update['status']}")
    
    # Example 3: Debug mode
    async def example_debug():
        await debug_analyze(
            incident_description="High memory usage in payment service",
            context={"service": "payment-service", "environment": "production"}
        )
    
    # Run example
    asyncio.run(example_basic())
