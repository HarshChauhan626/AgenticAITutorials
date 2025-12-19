"""
Sequential chains implementation for Ops Copilot.

This module implements the analysis workflow using LangChain's
sequential chains instead of a state machine. Each chain performs
a specific task and passes its output to the next chain.

Chain Flow:
1. Planning Chain - Decide which tools to use
2. Tool Execution Chain - Execute selected tools
3. RAG Chain - Retrieve relevant runbooks
4. Analysis Chain - Generate hypothesis
5. Action Chain - Generate next actions
"""

from typing import Dict, Any, List
from langchain.chains import LLMChain, SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from ..common.config import settings
from ..common.models import NextAction, Command, Citation


# ============================================================================
# Output Schemas
# ============================================================================

class ToolPlan(BaseModel):
    """Schema for tool planning output"""
    tools: List[str] = Field(
        description="Comma-separated list of tools to execute"
    )
    reasoning: str = Field(
        description="Why these tools were selected"
    )


class HypothesisOutput(BaseModel):
    """Schema for hypothesis generation"""
    hypothesis: str = Field(description="Root cause hypothesis")
    confidence: float = Field(description="Confidence 0-1", ge=0.0, le=1.0)
    reasoning: str = Field(description="Detailed reasoning")


# ============================================================================
# Chain 1: Planning Chain
# ============================================================================

def create_planning_chain() -> LLMChain:
    """
    Create the planning chain that decides which tools to use.
    
    This chain analyzes the incident description and selects
    appropriate tools for investigation.
    
    Returns:
        LLMChain: Planning chain
    
    Input Variables:
        - incident_description: The incident description
        - service: Service name
    
    Output Variables:
        - tools_to_use: Comma-separated list of tool names
        - planning_reasoning: Why these tools were selected
    
    How it works:
    1. Receives incident description and service
    2. LLM analyzes what information is needed
    3. Selects appropriate tools (log_search, metrics_query, etc.)
    4. Returns tool list and reasoning
    """
    # Create LLM
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature
    )
    
    # Create prompt template
    planning_prompt = PromptTemplate(
        input_variables=["incident_description", "service"],
        template="""
You are an expert SRE analyzing a production incident.

INCIDENT: {incident_description}
SERVICE: {service}

Available tools:
1. log_search - Search application logs for errors and patterns
2. metrics_query - Query time-series metrics from Prometheus
3. deploy_history - Get recent deployment history
4. runbook_search - Search runbook corpus for remediation steps

Based on the incident, which tools should we use to investigate?
Provide a comma-separated list of tool names.

Also explain your reasoning for selecting these tools.

Tools to use (comma-separated):
""".strip()
    )
    
    # Create chain
    chain = LLMChain(
        llm=llm,
        prompt=planning_prompt,
        output_key="tools_to_use"
    )
    
    return chain


# ============================================================================
# Chain 2: Analysis Chain
# ============================================================================

def create_analysis_chain() -> LLMChain:
    """
    Create the analysis chain that generates a hypothesis.
    
    This chain analyzes all gathered evidence and formulates
    a hypothesis about the root cause.
    
    Returns:
        LLMChain: Analysis chain
    
    Input Variables:
        - incident_description: The incident description
        - evidence: Formatted evidence from tools
        - runbooks: Retrieved runbooks
    
    Output Variables:
        - hypothesis: Root cause hypothesis
        - confidence: Confidence score
        - analysis_reasoning: Detailed reasoning
    
    How it works:
    1. Receives incident description, evidence, and runbooks
    2. LLM analyzes all information
    3. Generates hypothesis about root cause
    4. Provides confidence score and reasoning
    """
    # Create LLM
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature
    )
    
    # Create prompt template
    analysis_prompt = PromptTemplate(
        input_variables=["incident_description", "evidence", "runbooks"],
        template="""
You are an expert SRE analyzing a production incident.

INCIDENT: {incident_description}

EVIDENCE GATHERED:
{evidence}

RELEVANT RUNBOOKS:
{runbooks}

Based on this evidence, what is the most likely root cause?

Provide:
1. A clear hypothesis about the root cause
2. Your confidence level (0.0 to 1.0)
3. Detailed reasoning for your hypothesis

Format your response as:
HYPOTHESIS: [your hypothesis]
CONFIDENCE: [0.0 to 1.0]
REASONING: [your detailed reasoning]
""".strip()
    )
    
    # Create chain
    chain = LLMChain(
        llm=llm,
        prompt=analysis_prompt,
        output_key="analysis_result"
    )
    
    return chain


# ============================================================================
# Chain 3: Action Generation Chain
# ============================================================================

def create_action_chain() -> LLMChain:
    """
    Create the action generation chain.
    
    This chain generates prioritized next actions based on
    the hypothesis and evidence.
    
    Returns:
        LLMChain: Action generation chain
    
    Input Variables:
        - hypothesis: The root cause hypothesis
        - evidence: Evidence summary
    
    Output Variables:
        - next_actions: JSON array of actions
    
    How it works:
    1. Receives hypothesis and evidence
    2. LLM generates 3-5 prioritized actions
    3. Each action includes priority, time estimate, and rationale
    4. Returns actions as JSON array
    """
    # Create LLM
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature
    )
    
    # Create prompt template
    action_prompt = PromptTemplate(
        input_variables=["hypothesis", "evidence"],
        template="""
You are an expert SRE providing remediation guidance.

ROOT CAUSE HYPOTHESIS: {hypothesis}

EVIDENCE: {evidence}

Based on this hypothesis, suggest 3-5 prioritized next actions to resolve the incident.

For each action, provide:
- action: Clear description of what to do
- priority: high, medium, or low
- estimated_time: How long it will take (e.g., "5 minutes", "1 hour")
- rationale: Why this action is recommended

Format as a JSON array. Example:
[
  {{
    "action": "Rollback to previous version",
    "priority": "high",
    "estimated_time": "5 minutes",
    "rationale": "Fastest path to mitigation"
  }}
]

Actions (JSON array):
""".strip()
    )
    
    # Create chain
    chain = LLMChain(
        llm=llm,
        prompt=action_prompt,
        output_key="next_actions"
    )
    
    return chain


# ============================================================================
# Chain 4: Command Generation Chain
# ============================================================================

def create_command_chain() -> LLMChain:
    """
    Create the command generation chain.
    
    This chain generates specific commands to execute based
    on the recommended actions.
    
    Returns:
        LLMChain: Command generation chain
    
    Input Variables:
        - next_actions: The recommended actions
        - service: Service name
    
    Output Variables:
        - commands: JSON array of commands
    
    How it works:
    1. Receives recommended actions
    2. LLM generates specific executable commands
    3. Marks each command as safe_to_run or not
    4. Returns commands as JSON array
    """
    # Create LLM
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature
    )
    
    # Create prompt template
    command_prompt = PromptTemplate(
        input_variables=["next_actions", "service"],
        template="""
You are an expert SRE providing specific commands to execute.

RECOMMENDED ACTIONS: {next_actions}
SERVICE: {service}

For the recommended actions, provide specific commands that can be executed.

For each command, provide:
- description: What the command does
- command: The actual command to run
- safe_to_run: true if safe to auto-execute, false if requires approval

IMPORTANT: Mark commands as safe_to_run=false if they:
- Modify production state (rollbacks, restarts, config changes)
- Delete data
- Have potential side effects

Mark as safe_to_run=true only for:
- Read-only queries
- Log viewing
- Metric queries

Format as JSON array. Example:
[
  {{
    "description": "Check error logs",
    "command": "kubectl logs deployment/{service} --tail=100",
    "safe_to_run": true
  }}
]

Commands (JSON array):
""".strip()
    )
    
    # Create chain
    chain = LLMChain(
        llm=llm,
        prompt=command_prompt,
        output_key="commands"
    )
    
    return chain


# ============================================================================
# Combined Sequential Chain
# ============================================================================

def create_sequential_chain() -> SequentialChain:
    """
    Create the complete sequential chain.
    
    This combines all individual chains into a single sequential
    workflow that processes the incident from start to finish.
    
    Returns:
        SequentialChain: Complete analysis chain
    
    Flow:
    1. Planning → tools_to_use
    2. Analysis → hypothesis, confidence, reasoning
    3. Actions → next_actions
    4. Commands → commands
    
    Input Variables:
        - incident_description
        - service
        - evidence (populated by tool execution)
        - runbooks (populated by RAG)
    
    Output Variables:
        - tools_to_use
        - hypothesis
        - confidence
        - analysis_reasoning
        - next_actions
        - commands
    
    Note: Tool execution and RAG retrieval happen between
    planning and analysis chains (handled by orchestrator)
    """
    # Create individual chains
    planning_chain = create_planning_chain()
    analysis_chain = create_analysis_chain()
    action_chain = create_action_chain()
    command_chain = create_command_chain()
    
    # Combine into sequential chain
    overall_chain = SequentialChain(
        chains=[
            planning_chain,
            # Tool execution happens here (in orchestrator)
            # RAG retrieval happens here (in orchestrator)
            analysis_chain,
            action_chain,
            command_chain
        ],
        input_variables=[
            "incident_description",
            "service",
            "evidence",  # Populated by orchestrator
            "runbooks"   # Populated by orchestrator
        ],
        output_variables=[
            "tools_to_use",
            "analysis_result",
            "next_actions",
            "commands"
        ],
        verbose=True
    )
    
    return overall_chain


# ============================================================================
# Utility Functions
# ============================================================================

def parse_analysis_result(analysis_text: str) -> Dict[str, Any]:
    """
    Parse the analysis result text into structured data.
    
    Args:
        analysis_text: Raw text from analysis chain
    
    Returns:
        Dict with hypothesis, confidence, and reasoning
    
    Example Input:
        HYPOTHESIS: Database driver incompatibility
        CONFIDENCE: 0.85
        REASONING: Evidence shows...
    
    Example Output:
        {
            "hypothesis": "Database driver incompatibility",
            "confidence": 0.85,
            "reasoning": "Evidence shows..."
        }
    """
    lines = analysis_text.strip().split('\n')
    result = {
        "hypothesis": "",
        "confidence": 0.0,
        "reasoning": ""
    }
    
    for line in lines:
        if line.startswith("HYPOTHESIS:"):
            result["hypothesis"] = line.replace("HYPOTHESIS:", "").strip()
        elif line.startswith("CONFIDENCE:"):
            try:
                result["confidence"] = float(line.replace("CONFIDENCE:", "").strip())
            except ValueError:
                result["confidence"] = 0.5
        elif line.startswith("REASONING:"):
            result["reasoning"] = line.replace("REASONING:", "").strip()
    
    return result


def format_evidence(evidence: List[Dict[str, Any]]) -> str:
    """
    Format evidence list into a string for LLM prompt.
    
    Args:
        evidence: List of evidence dictionaries
    
    Returns:
        Formatted evidence string
    
    Example:
        - [logs] Database connection timeout (78% of 247 errors)
        - [metrics] Error rate spike: 23.4 req/s (baseline: 2.1)
        - [deployment] Recent deployment: v2.3.5
    """
    if not evidence:
        return "No evidence gathered"
    
    formatted = []
    for e in evidence:
        source = e.get("source", "unknown")
        finding = e.get("finding", "")
        formatted.append(f"- [{source}] {finding}")
    
    return "\n".join(formatted)


def format_runbooks(runbooks: List[Dict[str, Any]]) -> str:
    """
    Format runbooks list into a string for LLM prompt.
    
    Args:
        runbooks: List of runbook dictionaries
    
    Returns:
        Formatted runbooks string
    
    Example:
        [1] "Debugging 500 Errors" (score: 0.94)
            When API gateway returns 500 errors, check...
        
        [2] "Database Timeouts" (score: 0.91)
            Common causes of timeouts include...
    """
    if not runbooks:
        return "No runbooks found"
    
    formatted = []
    for i, rb in enumerate(runbooks, 1):
        title = rb.get("title", "Untitled")
        score = rb.get("score", 0.0)
        content = rb.get("content", "")[:200]  # First 200 chars
        
        formatted.append(
            f"[{i}] \"{title}\" (score: {score:.2f})\n"
            f"    {content}..."
        )
    
    return "\n\n".join(formatted)


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Example: Create and test individual chains
    
    # Test planning chain
    planning_chain = create_planning_chain()
    planning_result = planning_chain({
        "incident_description": "API returning 500 errors",
        "service": "api-gateway"
    })
    
    print("✅ Planning chain result:")
    print(f"   Tools: {planning_result['tools_to_use']}")
    
    # Test analysis chain
    analysis_chain = create_analysis_chain()
    analysis_result = analysis_chain({
        "incident_description": "API returning 500 errors",
        "evidence": "- [logs] Database timeout errors\n- [metrics] Error spike detected",
        "runbooks": "[1] Database Connection Issues\n    Check connection pool..."
    })
    
    print("\n✅ Analysis chain result:")
    print(f"   Result: {analysis_result['analysis_result'][:100]}...")
