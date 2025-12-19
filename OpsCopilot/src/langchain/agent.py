"""
Agent-based implementation using LangChain agents.

This module implements the analysis workflow using LangChain's
agent framework with tools. The agent can dynamically decide
which tools to use and when to use them.

Agent Flow:
1. Agent receives incident description
2. Agent selects and executes tools
3. Agent analyzes results
4. Agent generates recommendations
"""

from typing import List, Dict, Any
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool, StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from ..common.config import settings
from ..common.tools import ToolRegistry


# ============================================================================
# Tool Schemas
# ============================================================================

class LogSearchInput(BaseModel):
    """Input schema for log search tool"""
    query: str = Field(description="Search query for logs")
    service: str = Field(description="Service name to filter by")
    time_range: str = Field(
        default="last_1h",
        description="Time range (last_1h, last_6h, last_24h)"
    )


class MetricsQueryInput(BaseModel):
    """Input schema for metrics query tool"""
    service: str = Field(description="Service name to query")
    metric_name: str = Field(
        default="http_requests_total",
        description="Metric name to query"
    )
    time_range: str = Field(
        default="last_1h",
        description="Time range for query"
    )


class DeployHistoryInput(BaseModel):
    """Input schema for deployment history tool"""
    service: str = Field(description="Service name")
    time_range: str = Field(
        default="last_6h",
        description="How far back to look"
    )


# ============================================================================
# Tool Wrappers
# ============================================================================

def create_log_search_tool() -> StructuredTool:
    """
    Create the log search tool for the agent.
    
    This tool allows the agent to search application logs
    for errors and patterns.
    
    Returns:
        StructuredTool: Log search tool
    
    How it works:
    1. Agent calls tool with query and service
    2. Tool searches Elasticsearch
    3. Returns log entries and error patterns
    4. Agent uses results for analysis
    """
    registry = ToolRegistry()
    
    async def log_search(query: str, service: str, time_range: str = "last_1h") -> str:
        """
        Search application logs for errors and patterns.
        
        Args:
            query: Search query (e.g., "500", "error", "timeout")
            service: Service name to filter by
            time_range: Time range to search
        
        Returns:
            String summary of log search results
        """
        result = await registry.execute_tool(
            "log_search",
            query=query,
            service=service,
            time_range=time_range
        )
        
        if result.get("success"):
            data = result.get("data", {})
            pattern = data.get("error_pattern", "No pattern detected")
            count = data.get("total_count", 0)
            return f"Found {count} log entries. Pattern: {pattern}"
        else:
            return f"Log search failed: {result.get('error', 'Unknown error')}"
    
    return StructuredTool.from_function(
        func=log_search,
        name="log_search",
        description="Search application logs for errors and patterns. Use this to find error messages, stack traces, and log patterns.",
        args_schema=LogSearchInput
    )


def create_metrics_query_tool() -> StructuredTool:
    """
    Create the metrics query tool for the agent.
    
    This tool allows the agent to query time-series metrics
    from Prometheus.
    
    Returns:
        StructuredTool: Metrics query tool
    
    How it works:
    1. Agent calls tool with service and metric name
    2. Tool queries Prometheus
    3. Returns metric data and spike detection
    4. Agent uses results to identify anomalies
    """
    registry = ToolRegistry()
    
    async def metrics_query(
        service: str,
        metric_name: str = "http_requests_total",
        time_range: str = "last_1h"
    ) -> str:
        """
        Query time-series metrics from Prometheus.
        
        Args:
            service: Service name to query
            metric_name: Metric to query
            time_range: Time range for query
        
        Returns:
            String summary of metrics data
        """
        result = await registry.execute_tool(
            "metrics_query",
            service=service,
            metric_name=metric_name,
            time_range=time_range
        )
        
        if result.get("success"):
            data = result.get("data", {})
            metadata = data.get("metadata", {})
            
            if metadata.get("spike_detected"):
                baseline = metadata.get("baseline", 0)
                peak = metadata.get("peak", 0)
                return f"Spike detected! Baseline: {baseline}, Peak: {peak}"
            else:
                return "No anomalies detected in metrics"
        else:
            return f"Metrics query failed: {result.get('error', 'Unknown error')}"
    
    return StructuredTool.from_function(
        func=metrics_query,
        name="metrics_query",
        description="Query time-series metrics from Prometheus. Use this to detect spikes, anomalies, and trends in system metrics.",
        args_schema=MetricsQueryInput
    )


def create_deploy_history_tool() -> StructuredTool:
    """
    Create the deployment history tool for the agent.
    
    This tool allows the agent to retrieve recent deployments
    and correlate them with incidents.
    
    Returns:
        StructuredTool: Deployment history tool
    
    How it works:
    1. Agent calls tool with service name
    2. Tool fetches recent deployments
    3. Returns deployment info and correlation
    4. Agent uses results to identify deployment-related issues
    """
    registry = ToolRegistry()
    
    async def deploy_history(service: str, time_range: str = "last_6h") -> str:
        """
        Get recent deployment history for a service.
        
        Args:
            service: Service name
            time_range: How far back to look
        
        Returns:
            String summary of deployment history
        """
        result = await registry.execute_tool(
            "deploy_history",
            service=service,
            time_range=time_range
        )
        
        if result.get("success"):
            data = result.get("data", {})
            deployments = data.get("deployments", [])
            correlation = data.get("correlation", {})
            
            if deployments:
                latest = deployments[0]
                version = latest.get("version", "unknown")
                likely_related = correlation.get("likely_related", False)
                
                if likely_related:
                    return f"Recent deployment: {version}. Likely related to incident!"
                else:
                    return f"Recent deployment: {version}. Timing doesn't correlate with incident."
            else:
                return "No recent deployments found"
        else:
            return f"Deployment history query failed: {result.get('error', 'Unknown error')}"
    
    return StructuredTool.from_function(
        func=deploy_history,
        name="deploy_history",
        description="Get recent deployment history. Use this to check if recent code changes might have caused the incident.",
        args_schema=DeployHistoryInput
    )


# ============================================================================
# Agent Creation
# ============================================================================

def create_ops_copilot_agent() -> AgentExecutor:
    """
    Create the Ops Copilot agent with tools.
    
    This agent can dynamically select and execute tools to
    investigate incidents and generate recommendations.
    
    Returns:
        AgentExecutor: Configured agent ready for execution
    
    Agent Capabilities:
    - Searches logs for error patterns
    - Queries metrics for anomalies
    - Checks deployment history
    - Analyzes evidence
    - Generates recommendations
    
    How it works:
    1. Agent receives incident description
    2. Agent decides which tools to use
    3. Agent executes tools and analyzes results
    4. Agent iterates if needed (max 5 iterations)
    5. Agent generates final recommendations
    
    Example:
        >>> agent = create_ops_copilot_agent()
        >>> result = agent.invoke({
        ...     "input": "API returning 500 errors"
        ... })
    """
    # Create LLM
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature
    )
    
    # Create tools
    tools = [
        create_log_search_tool(),
        create_metrics_query_tool(),
        create_deploy_history_tool()
    ]
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an expert SRE assistant analyzing production incidents.

Your goal is to:
1. Investigate the incident using available tools
2. Identify the root cause
3. Suggest actionable remediation steps

PROCESS:
1. First, use log_search to find error patterns
2. Then, use metrics_query to detect anomalies
3. Check deploy_history for recent changes
4. Analyze all evidence together
5. Form a hypothesis about the root cause
6. Suggest 3-5 prioritized next actions

IMPORTANT:
- Use tools systematically
- Cite evidence for your conclusions
- Be specific and actionable
- Prioritize high-impact actions

Available tools: {tools}
Tool names: {tool_names}
"""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=settings.max_iterations,
        max_execution_time=settings.request_timeout,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )
    
    return agent_executor


# ============================================================================
# Agent Execution Helper
# ============================================================================

async def analyze_with_agent(
    incident_description: str,
    context: Dict[str, str]
) -> Dict[str, Any]:
    """
    Analyze an incident using the agent.
    
    This is a convenience function that creates the agent,
    executes it, and formats the result.
    
    Args:
        incident_description: Description of the incident
        context: Additional context (service, environment, etc.)
    
    Returns:
        Dict with analysis results
    
    Example:
        >>> result = await analyze_with_agent(
        ...     incident_description="API returning 500 errors",
        ...     context={"service": "api-gateway"}
        ... )
        >>> print(result["output"])
    """
    # Create agent
    agent = create_ops_copilot_agent()
    
    # Build input
    service = context.get("service", "unknown")
    environment = context.get("environment", "production")
    
    input_text = f"""
Analyze this production incident:

INCIDENT: {incident_description}
SERVICE: {service}
ENVIRONMENT: {environment}

Please investigate and provide:
1. Root cause hypothesis
2. Confidence level
3. Prioritized next actions
4. Specific commands to run
"""
    
    # Execute agent
    result = await agent.ainvoke({"input": input_text})
    
    # Format result
    return {
        "output": result["output"],
        "intermediate_steps": result.get("intermediate_steps", []),
        "tools_used": [
            step[0].tool for step in result.get("intermediate_steps", [])
        ]
    }


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def example():
        # Create agent
        agent = create_ops_copilot_agent()
        
        # Test with sample incident
        result = await agent.ainvoke({
            "input": """
Analyze this incident:

INCIDENT: API gateway returning 500 errors since 2pm
SERVICE: api-gateway
ENVIRONMENT: production

Investigate and provide recommendations.
"""
        })
        
        print("✅ Agent analysis complete:")
        print(f"   Output: {result['output'][:200]}...")
        print(f"   Tools used: {len(result.get('intermediate_steps', []))}")
    
    # Run example
    asyncio.run(example())
