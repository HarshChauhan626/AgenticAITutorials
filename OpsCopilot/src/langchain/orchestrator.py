"""
Main orchestrator for pure LangChain implementation.

This module provides the main OpscopilotChain class that orchestrates
the entire incident analysis workflow using sequential chains.

Unlike LangGraph which uses a state machine, this implementation
uses a more traditional approach with sequential processing.
"""

import time
import json
import uuid
import asyncio
from typing import Dict, Any, List, Optional
from langchain.callbacks import get_openai_callback

from .chains import (
    create_planning_chain,
    create_analysis_chain,
    create_action_chain,
    create_command_chain,
    parse_analysis_result,
    format_evidence,
    format_runbooks
)
from .rag import hybrid_retrieval
from ..common.config import settings
from ..common.tools import ToolRegistry
from ..common.models import AnalyzeRequest, AnalyzeResponse, AnalysisResult


# ============================================================================
# Main Orchestrator Class
# ============================================================================

class OpscopilotChain:
    """
    Main orchestrator for incident analysis using pure LangChain.
    
    This class coordinates the entire analysis workflow:
    1. Planning - Decide which tools to use
    2. Tool Execution - Run tools in parallel
    3. RAG Retrieval - Get relevant runbooks
    4. Analysis - Generate hypothesis
    5. Action Generation - Create next actions
    6. Command Generation - Generate specific commands
    
    Unlike LangGraph's state machine approach, this uses sequential
    processing with explicit coordination between steps.
    
    Example:
        >>> copilot = OpscopilotChain()
        >>> result = await copilot.analyze_incident(
        ...     incident_description="API returning 500 errors",
        ...     context={"service": "api-gateway"}
        ... )
    """
    
    def __init__(self):
        """
        Initialize the orchestrator.
        
        Sets up all chains and tools needed for analysis.
        """
        # Create chains
        self.planning_chain = create_planning_chain()
        self.analysis_chain = create_analysis_chain()
        self.action_chain = create_action_chain()
        self.command_chain = create_command_chain()
        
        # Create tool registry
        self.tool_registry = ToolRegistry()
    
    async def analyze_incident(
        self,
        incident_description: str,
        context: Dict[str, str],
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze an incident and generate recommendations.
        
        This is the main entry point for incident analysis.
        It orchestrates all steps of the workflow.
        
        Args:
            incident_description: Description of the incident
            context: Additional context (service, environment, etc.)
            correlation_id: Optional request ID
        
        Returns:
            Dict with complete analysis results
        
        Process:
        1. Generate correlation ID
        2. Planning: Decide which tools to use
        3. Tool Execution: Run selected tools in parallel
        4. RAG Retrieval: Get relevant runbooks
        5. Evidence Aggregation: Combine all data
        6. Analysis: Generate hypothesis
        7. Action Generation: Create next actions
        8. Command Generation: Generate commands
        9. Build final response
        
        Example:
            >>> result = await copilot.analyze_incident(
            ...     incident_description="High error rate",
            ...     context={"service": "api-gateway"}
            ... )
            >>> print(result["hypothesis"])
        """
        # Generate correlation ID if not provided
        if correlation_id is None:
            correlation_id = f"req_{uuid.uuid4().hex[:12]}"
        
        # Track start time
        start_time = time.time()
        
        # Extract service from context
        service = context.get("service", "unknown")
        environment = context.get("environment", "production")
        
        # Track token usage
        with get_openai_callback() as cb:
            
            # ================================================================
            # Step 1: Planning - Decide which tools to use
            # ================================================================
            
            print(f"[{correlation_id}] Step 1: Planning...")
            
            planning_result = self.planning_chain({
                "incident_description": incident_description,
                "service": service
            })
            
            # Parse tools list
            tools_str = planning_result["tools_to_use"]
            tools_to_use = [t.strip() for t in tools_str.split(",")]
            
            print(f"[{correlation_id}] Tools selected: {tools_to_use}")
            
            # ================================================================
            # Step 2: Tool Execution - Run tools in parallel
            # ================================================================
            
            print(f"[{correlation_id}] Step 2: Executing tools...")
            
            tool_results = await self._execute_tools(
                tools_to_use,
                service,
                incident_description
            )
            
            print(f"[{correlation_id}] Tools executed: {len(tool_results)}")
            
            # ================================================================
            # Step 3: RAG Retrieval - Get relevant runbooks
            # ================================================================
            
            print(f"[{correlation_id}] Step 3: Retrieving runbooks...")
            
            runbooks = hybrid_retrieval(
                query=incident_description,
                service=service,
                top_k=5
            )
            
            print(f"[{correlation_id}] Runbooks retrieved: {len(runbooks)}")
            
            # ================================================================
            # Step 4: Evidence Aggregation - Combine all data
            # ================================================================
            
            print(f"[{correlation_id}] Step 4: Aggregating evidence...")
            
            evidence = self._aggregate_evidence(tool_results)
            
            # Format for LLM
            evidence_str = format_evidence(evidence)
            runbooks_str = format_runbooks(runbooks)
            
            # ================================================================
            # Step 5: Analysis - Generate hypothesis
            # ================================================================
            
            print(f"[{correlation_id}] Step 5: Generating hypothesis...")
            
            analysis_result = self.analysis_chain({
                "incident_description": incident_description,
                "evidence": evidence_str,
                "runbooks": runbooks_str
            })
            
            # Parse analysis result
            analysis = parse_analysis_result(analysis_result["analysis_result"])
            
            print(f"[{correlation_id}] Hypothesis: {analysis['hypothesis'][:50]}...")
            print(f"[{correlation_id}] Confidence: {analysis['confidence']}")
            
            # ================================================================
            # Step 6: Action Generation - Create next actions
            # ================================================================
            
            print(f"[{correlation_id}] Step 6: Generating actions...")
            
            action_result = self.action_chain({
                "hypothesis": analysis["hypothesis"],
                "evidence": evidence_str
            })
            
            # Parse actions (JSON)
            try:
                next_actions = json.loads(action_result["next_actions"])
            except json.JSONDecodeError:
                # Fallback if parsing fails
                next_actions = [{
                    "action": "Manual investigation required",
                    "priority": "high",
                    "estimated_time": "unknown",
                    "rationale": "Automated analysis incomplete"
                }]
            
            print(f"[{correlation_id}] Actions generated: {len(next_actions)}")
            
            # ================================================================
            # Step 7: Command Generation - Generate commands
            # ================================================================
            
            print(f"[{correlation_id}] Step 7: Generating commands...")
            
            command_result = self.command_chain({
                "next_actions": json.dumps(next_actions),
                "service": service
            })
            
            # Parse commands (JSON)
            try:
                commands = json.loads(command_result["commands"])
            except json.JSONDecodeError:
                commands = []
            
            print(f"[{correlation_id}] Commands generated: {len(commands)}")
            
            # ================================================================
            # Step 8: Build Citations
            # ================================================================
            
            citations = self._build_citations(evidence, runbooks)
            
            # ================================================================
            # Step 9: Build Final Response
            # ================================================================
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Build complete response
            final_response = {
                "request_id": correlation_id,
                "timestamp": time.time(),
                "latency_ms": latency_ms,
                "result": {
                    "hypothesis": analysis["hypothesis"],
                    "confidence": analysis["confidence"],
                    "reasoning": analysis["reasoning"],
                    "next_actions": next_actions,
                    "commands": commands,
                    "citations": citations
                },
                "metadata": {
                    "tools_used": tools_to_use,
                    "runbooks_retrieved": len(runbooks),
                    "total_tokens": cb.total_tokens,
                    "total_cost": cb.total_cost,
                    "cache_hit": False
                }
            }
            
            print(f"[{correlation_id}] Analysis complete! Latency: {latency_ms}ms")
            
            return final_response
    
    async def _execute_tools(
        self,
        tools_to_use: List[str],
        service: str,
        incident_description: str
    ) -> Dict[str, Any]:
        """
        Execute selected tools in parallel.
        
        Args:
            tools_to_use: List of tool names to execute
            service: Service name
            incident_description: Incident description
        
        Returns:
            Dict mapping tool names to their results
        
        How it works:
        1. Creates async tasks for each tool
        2. Executes all tools in parallel using asyncio.gather()
        3. Returns combined results
        """
        tool_configs = []
        
        for tool_name in tools_to_use:
            if tool_name == "log_search":
                tool_configs.append({
                    "name": "log_search",
                    "params": {
                        "query": "error OR 500 OR timeout",
                        "service": service,
                        "time_range": "last_1h"
                    }
                })
            
            elif tool_name == "metrics_query":
                tool_configs.append({
                    "name": "metrics_query",
                    "params": {
                        "service": service,
                        "metric_name": "http_requests_total",
                        "time_range": "last_1h"
                    }
                })
            
            elif tool_name == "deploy_history":
                tool_configs.append({
                    "name": "deploy_history",
                    "params": {
                        "service": service,
                        "time_range": "last_6h"
                    }
                })
        
        # Execute in parallel
        if tool_configs:
            results = await self.tool_registry.execute_multiple(tool_configs)
            return results
        
        return {}
    
    def _aggregate_evidence(self, tool_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Aggregate evidence from tool results.
        
        Args:
            tool_results: Results from executed tools
        
        Returns:
            List of evidence dictionaries
        
        How it works:
        1. Extracts key findings from each tool result
        2. Structures evidence with source, finding, and details
        3. Returns unified evidence list
        """
        evidence = []
        
        # Extract from logs
        if "log_search" in tool_results:
            log_result = tool_results["log_search"]
            if log_result.get("success"):
                data = log_result.get("data", {})
                evidence.append({
                    "source": "logs",
                    "finding": data.get("error_pattern", "Errors detected"),
                    "count": data.get("total_count", 0)
                })
        
        # Extract from metrics
        if "metrics_query" in tool_results:
            metrics_result = tool_results["metrics_query"]
            if metrics_result.get("success"):
                data = metrics_result.get("data", {})
                metadata = data.get("metadata", {})
                
                if metadata.get("spike_detected"):
                    evidence.append({
                        "source": "metrics",
                        "finding": f"Error rate spike: {metadata.get('peak', 0)} req/s",
                        "baseline": metadata.get("baseline", 0)
                    })
        
        # Extract from deployments
        if "deploy_history" in tool_results:
            deploy_result = tool_results["deploy_history"]
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
        
        return evidence
    
    def _build_citations(
        self,
        evidence: List[Dict[str, Any]],
        runbooks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build citations from evidence and runbooks.
        
        Args:
            evidence: Evidence list
            runbooks: Retrieved runbooks
        
        Returns:
            List of citation dictionaries
        
        How it works:
        1. Creates citations for each evidence item
        2. Creates citations for each runbook
        3. Returns combined citation list
        """
        citations = []
        
        # Citations from evidence
        for e in evidence:
            citations.append({
                "source": e["source"],
                "reference": f"{e['source']} data",
                "excerpt": e["finding"],
                "timestamp": e.get("timestamp")
            })
        
        # Citations from runbooks
        for rb in runbooks:
            citations.append({
                "source": "runbook",
                "reference": rb.get("id", ""),
                "excerpt": rb.get("content", "")[:200],
                "timestamp": None
            })
        
        return citations


# ============================================================================
# Convenience Function
# ============================================================================

async def analyze_incident(
    incident_description: str,
    context: Dict[str, str],
    correlation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze an incident using the LangChain orchestrator.
    
    This is a convenience function that creates an OpscopilotChain
    instance and runs the analysis.
    
    Args:
        incident_description: Description of the incident
        context: Additional context
        correlation_id: Optional request ID
    
    Returns:
        Dict with analysis results
    
    Example:
        >>> result = await analyze_incident(
        ...     incident_description="API returning 500 errors",
        ...     context={"service": "api-gateway"}
        ... )
    """
    copilot = OpscopilotChain()
    return await copilot.analyze_incident(
        incident_description=incident_description,
        context=context,
        correlation_id=correlation_id
    )


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def example():
        # Create orchestrator
        copilot = OpscopilotChain()
        
        # Analyze incident
        result = await copilot.analyze_incident(
            incident_description="API gateway returning 500 errors since 2pm",
            context={
                "service": "api-gateway",
                "environment": "production",
                "severity": "critical"
            }
        )
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nRequest ID: {result['request_id']}")
        print(f"Latency: {result['latency_ms']}ms")
        print(f"\nHypothesis: {result['result']['hypothesis']}")
        print(f"Confidence: {result['result']['confidence']}")
        print(f"\nNext Actions: {len(result['result']['next_actions'])}")
        print(f"Commands: {len(result['result']['commands'])}")
        print(f"Citations: {len(result['result']['citations'])}")
        print(f"\nTools Used: {result['metadata']['tools_used']}")
        print(f"Total Tokens: {result['metadata']['total_tokens']}")
        print(f"Cost: ${result['metadata']['total_cost']:.4f}")
    
    # Run example
    asyncio.run(example())
