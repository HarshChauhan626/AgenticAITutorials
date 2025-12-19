"""
Graph node implementations for LangGraph workflow.

This module contains all the node functions that process the state
in the LangGraph execution. Each node performs a specific task and
updates the state accordingly.

Node Flow:
1. parse_input_node - Initialize state and create initial message
2. planning_node - Decide which tools to execute (LLM call)
3. tool_execution_node - Execute tools in parallel
4. rag_retrieval_node - Retrieve runbooks using hybrid RAG
5. evidence_aggregation_node - Aggregate evidence from all sources
6. reasoning_node - Generate hypothesis (LLM call)
7. decision_node - Decide whether to continue or generate response
8. response_generation_node - Generate final structured response (LLM call)
"""

import time
import json
import asyncio
from typing import Dict, Any, List
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from .state import AgentState
from ..common.config import settings
from ..common.tools import ToolRegistry
from ..common.models import NextAction, Command, Citation


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """You are an expert SRE assistant analyzing production incidents.

Your goal is to:
1. Identify the root cause of incidents
2. Suggest actionable remediation steps
3. Provide evidence-based recommendations
4. Cite all sources for your claims

CRITICAL RULES:
- Be concise and actionable
- Prioritize high-impact actions first
- Only suggest commands that are safe to run
- Always cite your sources
- If uncertain, say so explicitly
"""


# ============================================================================
# Node 1: Parse Input
# ============================================================================

def parse_input_node(state: AgentState) -> AgentState:
    """
    Initialize the state and create the initial message for the LLM.
    
    This is the first node in the graph. It sets up all the tracking
    variables and creates the initial prompt for the planning phase.
    
    Args:
        state: Current agent state
    
    Returns:
        AgentState: Updated state with initialized fields
    
    What it does:
    - Sets iteration_count to 0
    - Records start_time
    - Initializes empty lists for tools, evidence, etc.
    - Creates initial system and human messages
    """
    # Initialize iteration tracking
    state["iteration_count"] = 0
    state["start_time"] = time.time()
    
    # Initialize tool tracking
    state["tools_used"] = []
    state["tool_results"] = {}
    state["tools_to_execute"] = []
    
    # Initialize evidence tracking
    state["evidence"] = []
    state["timeline"] = []
    state["runbooks"] = []
    
    # Initialize metadata
    state["metadata"] = {
        "cache_hit": False,
        "total_tokens": 0
    }
    
    # Create initial messages for LLM
    state["messages"] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""
Analyze this production incident:

INCIDENT: {state['incident_description']}

SERVICE: {state['context'].get('service', 'unknown')}
ENVIRONMENT: {state['context'].get('environment', 'production')}
SEVERITY: {state['context'].get('severity', 'unknown')}

What should we investigate first?
        """.strip())
    ]
    
    return state


# ============================================================================
# Node 2: Planning (LLM Call #1)
# ============================================================================

class ToolPlan(BaseModel):
    """Schema for tool selection output"""
    tools: List[str] = Field(
        description="List of tools to execute (log_search, metrics_query, deploy_history, runbook_search)"
    )
    reasoning: str = Field(
        description="Why these tools were selected"
    )


def planning_node(state: AgentState) -> AgentState:
    """
    Decide which tools to execute based on the incident description.
    
    This node uses the LLM to intelligently select which tools will
    provide the most relevant information for diagnosing the incident.
    
    Args:
        state: Current agent state
    
    Returns:
        AgentState: Updated state with tools_to_execute populated
    
    What it does:
    - Sends planning prompt to LLM
    - Parses LLM response to extract tool list
    - Updates state with selected tools
    - Increments iteration count
    
    LLM Input:
        - System prompt
        - Incident description
        - Available tools list
    
    LLM Output:
        - List of tool names to execute
        - Reasoning for tool selection
    """
    # Create LLM with structured output
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature
    )
    
    # Create parser for structured output
    parser = PydanticOutputParser(pydantic_object=ToolPlan)
    
    # Create planning prompt
    planning_prompt = f"""
Available tools:
1. log_search - Search application logs for errors and patterns
2. metrics_query - Query time-series metrics from Prometheus
3. deploy_history - Get recent deployment history
4. runbook_search - Search runbook corpus for remediation steps

Based on the incident description, which tools should we use to investigate?

{parser.get_format_instructions()}
"""
    
    # Add planning prompt to messages
    state["messages"].append(HumanMessage(content=planning_prompt))
    
    # Call LLM
    response = llm(state["messages"])
    
    # Parse response
    try:
        plan = parser.parse(response.content)
        state["tools_to_execute"] = plan.tools
        
        # Add response to messages
        state["messages"].append(AIMessage(content=response.content))
        
    except Exception as e:
        # Fallback: use all tools if parsing fails
        print(f"Warning: Failed to parse planning response: {e}")
        state["tools_to_execute"] = [
            "log_search",
            "metrics_query",
            "deploy_history",
            "runbook_search"
        ]
    
    # Increment iteration
    state["iteration_count"] += 1
    
    return state


# ============================================================================
# Node 3: Tool Execution
# ============================================================================

async def tool_execution_node(state: AgentState) -> AgentState:
    """
    Execute selected tools in parallel.
    
    This node runs all selected tools concurrently to minimize latency.
    Each tool gathers specific evidence about the incident.
    
    Args:
        state: Current agent state
    
    Returns:
        AgentState: Updated state with tool_results populated
    
    What it does:
    - Creates async tasks for each selected tool
    - Executes all tools in parallel using asyncio.gather()
    - Stores results in state["tool_results"]
    - Tracks which tools were used in state["tools_used"]
    - Handles tool failures gracefully
    
    Tools executed:
    - log_search: Searches Elasticsearch for error logs
    - metrics_query: Queries Prometheus for metric spikes
    - deploy_history: Fetches recent deployments
    - runbook_search: Handled separately in RAG node
    """
    # Create tool registry
    registry = ToolRegistry()
    
    # Get service from context
    service = state["context"].get("service", "unknown")
    
    # Create tool configurations
    tool_configs = []
    
    for tool_name in state["tools_to_execute"]:
        if tool_name == "log_search":
            tool_configs.append({
                "name": "log_search",
                "params": {
                    "query": "error OR 500 OR timeout",
                    "service": service,
                    "time_range": "last_1h",
                    "max_results": 100
                }
            })
        
        elif tool_name == "metrics_query":
            tool_configs.append({
                "name": "metrics_query",
                "params": {
                    "service": service,
                    "metric_name": "http_requests_total",
                    "time_range": "last_1h",
                    "aggregation": "rate"
                }
            })
        
        elif tool_name == "deploy_history":
            tool_configs.append({
                "name": "deploy_history",
                "params": {
                    "service": service,
                    "time_range": "last_6h",
                    "limit": 10
                }
            })
        
        # runbook_search is handled in RAG node
    
    # Execute tools in parallel
    if tool_configs:
        results = await registry.execute_multiple(tool_configs)
        
        # Store results
        for tool_name, result in results.items():
            state["tool_results"][tool_name] = result
            
            # Track tool usage
            if tool_name not in state["tools_used"]:
                state["tools_used"].append(tool_name)
    
    return state


# ============================================================================
# Node 4: RAG Retrieval
# ============================================================================

def rag_retrieval_node(state: AgentState) -> AgentState:
    """
    Retrieve relevant runbooks using hybrid RAG.
    
    This node performs hybrid retrieval (vector + keyword search)
    followed by reranking to find the most relevant runbooks.
    
    Args:
        state: Current agent state
    
    Returns:
        AgentState: Updated state with runbooks populated
    
    What it does:
    - Performs vector search using Pinecone
    - Performs keyword search using Elasticsearch/BM25
    - Combines results using Reciprocal Rank Fusion
    - Reranks using cross-encoder
    - Stores top 5 runbooks in state
    
    Note: This is a simplified version. Full implementation
    would use the RAG pipeline from rag.py
    """
    # Import RAG components (will be implemented in rag.py)
    from .rag import hybrid_retrieval
    
    # Perform hybrid retrieval
    runbooks = hybrid_retrieval(
        query=state["incident_description"],
        service=state["context"].get("service"),
        top_k=5
    )
    
    # Store runbooks
    state["runbooks"] = runbooks
    
    # Track runbook search as used
    if "runbook_search" not in state["tools_used"]:
        state["tools_used"].append("runbook_search")
    
    return state


# ============================================================================
# Node 5: Evidence Aggregation
# ============================================================================

def evidence_aggregation_node(state: AgentState) -> AgentState:
    """
    Aggregate evidence from all sources into a structured format.
    
    This node combines data from logs, metrics, deployments, and
    runbooks into a unified evidence list and timeline.
    
    Args:
        state: Current agent state
    
    Returns:
        AgentState: Updated state with evidence and timeline populated
    
    What it does:
    - Extracts key findings from log search results
    - Identifies metric spikes and anomalies
    - Correlates deployments with incident timing
    - Builds chronological timeline of events
    - Stores structured evidence for LLM reasoning
    """
    evidence = []
    timeline = []
    
    # Extract evidence from logs
    if "log_search" in state["tool_results"]:
        log_result = state["tool_results"]["log_search"]
        if log_result.get("success"):
            data = log_result.get("data", {})
            evidence.append({
                "source": "logs",
                "finding": data.get("error_pattern", "Errors detected"),
                "count": data.get("total_count", 0)
            })
    
    # Extract evidence from metrics
    if "metrics_query" in state["tool_results"]:
        metrics_result = state["tool_results"]["metrics_query"]
        if metrics_result.get("success"):
            data = metrics_result.get("data", {})
            metadata = data.get("metadata", {})
            
            if metadata.get("spike_detected"):
                evidence.append({
                    "source": "metrics",
                    "finding": f"Error rate spike: {metadata.get('peak', 0)} req/s",
                    "baseline": metadata.get("baseline", 0)
                })
                
                # Add to timeline
                if metadata.get("spike_start"):
                    timeline.append({
                        "timestamp": metadata["spike_start"],
                        "type": "anomaly",
                        "description": "Error rate spike detected"
                    })
    
    # Extract evidence from deployments
    if "deploy_history" in state["tool_results"]:
        deploy_result = state["tool_results"]["deploy_history"]
        if deploy_result.get("success"):
            data = deploy_result.get("data", {})
            deployments = data.get("deployments", [])
            
            if deployments:
                latest = deployments[0]
                evidence.append({
                    "source": "deployment",
                    "finding": f"Recent deployment: {latest.get('version', 'unknown')}",
                    "details": latest.get("changes", [])
                })
                
                # Add to timeline
                timeline.append({
                    "timestamp": latest.get("timestamp"),
                    "type": "deployment",
                    "description": f"Deployed {latest.get('version', 'unknown')}"
                })
            
            # Check correlation
            correlation = data.get("correlation", {})
            if correlation.get("likely_related"):
                evidence.append({
                    "source": "correlation",
                    "finding": f"Deployment occurred {correlation.get('time_diff_minutes', 0)} minutes before incident",
                    "confidence": 0.85
                })
    
    # Sort timeline chronologically
    timeline.sort(key=lambda x: x.get("timestamp", ""))
    
    # Update state
    state["evidence"] = evidence
    state["timeline"] = timeline
    
    return state


# ============================================================================
# Node 6: Reasoning (LLM Call #2)
# ============================================================================

class HypothesisOutput(BaseModel):
    """Schema for hypothesis generation output"""
    hypothesis: str = Field(description="Root cause hypothesis")
    confidence: float = Field(description="Confidence score 0-1", ge=0.0, le=1.0)
    reasoning: str = Field(description="Detailed reasoning")


def reasoning_node(state: AgentState) -> AgentState:
    """
    Generate hypothesis about the root cause using LLM.
    
    This node analyzes all gathered evidence and formulates
    a hypothesis about what caused the incident.
    
    Args:
        state: Current agent state
    
    Returns:
        AgentState: Updated state with hypothesis and confidence
    
    What it does:
    - Formats evidence, timeline, and runbooks for LLM
    - Sends reasoning prompt to LLM
    - Parses hypothesis and confidence score
    - Updates state with current hypothesis
    
    LLM Input:
        - Incident description
        - Aggregated evidence from all sources
        - Timeline of events
        - Retrieved runbooks
    
    LLM Output:
        - Root cause hypothesis
        - Confidence score (0-1)
        - Detailed reasoning
    """
    # Create LLM
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature
    )
    
    # Create parser
    parser = PydanticOutputParser(pydantic_object=HypothesisOutput)
    
    # Format evidence
    evidence_str = "\n".join([
        f"- [{e['source']}] {e['finding']}"
        for e in state["evidence"]
    ])
    
    # Format timeline
    timeline_str = "\n".join([
        f"- {t['timestamp']}: {t['description']}"
        for t in state["timeline"]
    ])
    
    # Format runbooks
    runbooks_str = "\n\n".join([
        f"[{i+1}] {rb.get('title', 'Untitled')} (score: {rb.get('score', 0):.2f})\n"
        f"   {rb.get('content', '')[:200]}..."
        for i, rb in enumerate(state["runbooks"][:3])  # Top 3 runbooks
    ])
    
    # Create reasoning prompt
    reasoning_prompt = f"""
Analyze the following evidence and form a hypothesis about the root cause:

INCIDENT: {state['incident_description']}

EVIDENCE:
{evidence_str}

TIMELINE:
{timeline_str}

RELEVANT RUNBOOKS:
{runbooks_str}

Based on this evidence, what is the most likely root cause?

{parser.get_format_instructions()}
"""
    
    # Add to messages
    state["messages"].append(HumanMessage(content=reasoning_prompt))
    
    # Call LLM
    response = llm(state["messages"])
    
    # Parse response
    try:
        result = parser.parse(response.content)
        state["current_hypothesis"] = result.hypothesis
        state["confidence"] = result.confidence
        state["reasoning"] = result.reasoning
        
        # Add to messages
        state["messages"].append(AIMessage(content=response.content))
        
    except Exception as e:
        print(f"Warning: Failed to parse reasoning response: {e}")
        state["current_hypothesis"] = "Unable to determine root cause"
        state["confidence"] = 0.0
        state["reasoning"] = str(e)
    
    return state


# ============================================================================
# Node 7: Decision (Conditional Routing)
# ============================================================================

def should_continue(state: AgentState) -> str:
    """
    Decide whether to continue iteration or generate final response.
    
    This is a conditional routing function that determines the next
    step in the graph based on iteration count, confidence, and timing.
    
    Args:
        state: Current agent state
    
    Returns:
        str: "continue" to loop back to planning, "generate" to create response
    
    Decision Logic:
    1. If iteration_count >= 5: Stop (max iterations reached)
    2. If confidence >= 0.8: Stop (high confidence achieved)
    3. If evidence count >= 3: Stop (sufficient evidence gathered)
    4. If elapsed time > 25s: Stop (approaching timeout)
    5. Otherwise: Continue (need more data)
    """
    # Check iteration limit (prevent infinite loops)
    if state["iteration_count"] >= settings.max_iterations:
        return "generate"
    
    # Check confidence threshold
    if state.get("confidence", 0) >= 0.8:
        return "generate"
    
    # Check evidence sufficiency
    if len(state["evidence"]) >= 3:
        return "generate"
    
    # Check timeout (leave 5s buffer for response generation)
    elapsed = time.time() - state["start_time"]
    if elapsed > (settings.request_timeout - 5):
        return "generate"
    
    # Continue if we need more data
    return "continue"


# ============================================================================
# Node 8: Response Generation (LLM Call #3)
# ============================================================================

def response_generation_node(state: AgentState) -> AgentState:
    """
    Generate the final structured response.
    
    This is the final node that creates the complete response
    with hypothesis, actions, commands, and citations.
    
    Args:
        state: Current agent state
    
    Returns:
        AgentState: Updated state with final_response populated
    
    What it does:
    - Uses LLM with structured output to generate complete response
    - Ensures response includes all required fields
    - Validates response against schema
    - Stores final response in state
    
    LLM Input:
        - Current hypothesis and confidence
        - All evidence and timeline
        - Retrieved runbooks
    
    LLM Output (Structured):
        - Refined hypothesis
        - 3-5 prioritized next actions
        - Specific commands to run
        - Citations for all claims
    """
    # Create LLM with structured output
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature
    )
    
    # Format evidence for final prompt
    evidence_str = "\n".join([
        f"- [{e['source']}] {e['finding']}"
        for e in state["evidence"]
    ])
    
    # Create final prompt
    final_prompt = f"""
Based on all the evidence gathered, provide a complete incident analysis:

HYPOTHESIS: {state.get('current_hypothesis', 'Unknown')}
CONFIDENCE: {state.get('confidence', 0.0)}

EVIDENCE:
{evidence_str}

Provide a complete response with:
1. Refined hypothesis (if needed)
2. 3-5 prioritized next actions (with priority, estimated time, rationale)
3. Specific commands to run (with safety flags)
4. Citations for all claims (source, reference, excerpt)

Format as JSON with keys: hypothesis, confidence, next_actions, commands, citations
"""
    
    # Add to messages
    state["messages"].append(HumanMessage(content=final_prompt))
    
    # Call LLM
    response = llm(state["messages"])
    
    # Parse JSON response
    try:
        final_response = json.loads(response.content)
        
        # Ensure required fields exist
        if "hypothesis" not in final_response:
            final_response["hypothesis"] = state.get("current_hypothesis", "Unknown")
        
        if "confidence" not in final_response:
            final_response["confidence"] = state.get("confidence", 0.0)
        
        # Store final response
        state["final_response"] = final_response
        
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse final response as JSON: {e}")
        
        # Fallback response
        state["final_response"] = {
            "hypothesis": state.get("current_hypothesis", "Unable to determine"),
            "confidence": state.get("confidence", 0.0),
            "next_actions": [
                {
                    "action": "Manual investigation required",
                    "priority": "high",
                    "estimated_time": "unknown",
                    "rationale": "Automated analysis incomplete"
                }
            ],
            "commands": [],
            "citations": [
                {
                    "source": e["source"],
                    "reference": "Automated evidence gathering",
                    "excerpt": e["finding"]
                }
                for e in state["evidence"]
            ]
        }
    
    return state


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    from .state import create_initial_state
    
    # Example: Test parse_input_node
    state = create_initial_state(
        incident_description="API gateway returning 500 errors",
        context={"service": "api-gateway", "environment": "production"},
        correlation_id="req_test123"
    )
    
    # Run parse_input_node
    state = parse_input_node(state)
    
    print("✅ Parse input node completed:")
    print(f"   Iteration: {state['iteration_count']}")
    print(f"   Messages: {len(state['messages'])}")
    print(f"   Start time: {state['start_time']}")
