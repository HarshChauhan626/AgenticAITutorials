"""
Tool implementations for Ops Copilot.

This module contains all the tools that the agent can use to gather
evidence about incidents:
- Log search (Elasticsearch)
- Metrics query (Prometheus)
- Deployment history
- Runbook search (RAG)
- Ticket creation
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import aiohttp
from elasticsearch import AsyncElasticsearch
from abc import ABC, abstractmethod

from ..common.config import settings
from ..common.models import (
    LogSearchOutput,
    MetricsQueryOutput,
    DeployHistoryOutput,
    RunbookSearchOutput
)


# ============================================================================
# Base Tool Class
# ============================================================================

class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    Provides common functionality like error handling, timing,
    and result formatting.
    """
    
    def __init__(self):
        """Initialize the tool"""
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with given parameters.
        
        Args:
            **kwargs: Tool-specific parameters
        
        Returns:
            Dict[str, Any]: Tool output with success flag and data
        """
        pass
    
    async def _execute_with_timing(self, **kwargs) -> Dict[str, Any]:
        """
        Execute tool and track execution time.
        
        Args:
            **kwargs: Tool parameters
        
        Returns:
            Dict[str, Any]: Tool output with execution_time_ms added
        """
        start_time = datetime.utcnow()
        
        try:
            result = await self.execute(**kwargs)
            result["success"] = True
        except Exception as e:
            result = {
                "success": False,
                "error": str(e),
                "data": {}
            }
        
        # Calculate execution time
        elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
        result["execution_time_ms"] = int(elapsed)
        
        return result


# ============================================================================
# Log Search Tool
# ============================================================================

class LogSearchTool(BaseTool):
    """
    Search application logs using Elasticsearch.
    
    This tool queries Elasticsearch to find error patterns,
    stack traces, and relevant log entries for the incident.
    """
    
    def __init__(self):
        """Initialize Elasticsearch client"""
        super().__init__()
        
        # Create async Elasticsearch client
        self.es_client = AsyncElasticsearch(
            hosts=[settings.elasticsearch_url],
            basic_auth=(
                settings.elasticsearch_username,
                settings.elasticsearch_password
            ) if settings.elasticsearch_username else None
        )
    
    async def execute(
        self,
        query: str,
        service: str,
        time_range: str = "last_1h",
        max_results: int = 100
    ) -> Dict[str, Any]:
        """
        Search logs for errors and patterns.
        
        Args:
            query: Search query (e.g., "500", "error", "timeout")
            service: Service name to filter by
            time_range: Time range (e.g., "last_1h", "last_6h")
            max_results: Maximum number of results to return
        
        Returns:
            Dict containing logs, error patterns, and statistics
        """
        # Parse time range
        time_filter = self._parse_time_range(time_range)
        
        # Build Elasticsearch query
        es_query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"message": query}},
                        {"term": {"service": service}},
                        {
                            "range": {
                                "timestamp": {
                                    "gte": time_filter,
                                    "lte": "now"
                                }
                            }
                        }
                    ]
                }
            },
            "sort": [{"timestamp": "desc"}],
            "size": max_results
        }
        
        # Execute search
        response = await self.es_client.search(
            index="logs",
            body=es_query
        )
        
        # Extract logs
        logs = [
            {
                "timestamp": hit["_source"]["timestamp"],
                "level": hit["_source"].get("level", "INFO"),
                "service": hit["_source"]["service"],
                "message": hit["_source"]["message"],
                "trace_id": hit["_source"].get("trace_id")
            }
            for hit in response["hits"]["hits"]
        ]
        
        # Detect error patterns
        error_pattern = self._detect_error_pattern(logs)
        
        return {
            "logs": logs,
            "error_pattern": error_pattern,
            "total_count": response["hits"]["total"]["value"]
        }
    
    def _parse_time_range(self, time_range: str) -> str:
        """
        Convert time range string to Elasticsearch format.
        
        Args:
            time_range: Human-readable time range (e.g., "last_1h")
        
        Returns:
            str: Elasticsearch time format (e.g., "now-1h")
        """
        # Map common time ranges
        time_map = {
            "last_1h": "now-1h",
            "last_6h": "now-6h",
            "last_24h": "now-24h",
            "last_7d": "now-7d"
        }
        
        return time_map.get(time_range, "now-1h")
    
    def _detect_error_pattern(self, logs: List[Dict]) -> str:
        """
        Analyze logs to detect common error patterns.
        
        Args:
            logs: List of log entries
        
        Returns:
            str: Description of the most common error pattern
        """
        if not logs:
            return "No errors found"
        
        # Count error message frequencies
        error_counts = {}
        for log in logs:
            # Extract error type from message
            message = log.get("message", "")
            
            # Simple pattern extraction (can be improved with regex)
            if "timeout" in message.lower():
                error_type = "timeout"
            elif "connection" in message.lower():
                error_type = "connection"
            elif "500" in message:
                error_type = "500_error"
            else:
                error_type = "other"
            
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Find most common error
        if error_counts:
            most_common = max(error_counts.items(), key=lambda x: x[1])
            error_type, count = most_common
            percentage = (count / len(logs)) * 100
            
            return f"{error_type.replace('_', ' ').title()} ({percentage:.0f}% of {len(logs)} errors)"
        
        return "Various errors detected"


# ============================================================================
# Metrics Query Tool
# ============================================================================

class MetricsQueryTool(BaseTool):
    """
    Query time-series metrics from Prometheus.
    
    This tool retrieves metrics data to detect spikes, anomalies,
    and trends in system behavior.
    """
    
    async def execute(
        self,
        service: str,
        metric_name: str = "http_requests_total",
        time_range: str = "last_1h",
        aggregation: str = "rate"
    ) -> Dict[str, Any]:
        """
        Query Prometheus metrics.
        
        Args:
            service: Service name to query
            metric_name: Metric name (e.g., "http_requests_total")
            time_range: Time range for query
            aggregation: Aggregation function (rate, sum, avg)
        
        Returns:
            Dict containing data points and metadata (spikes, baseline, etc.)
        """
        # Build PromQL query
        promql = self._build_promql(service, metric_name, aggregation)
        
        # Query Prometheus
        async with aiohttp.ClientSession() as session:
            url = f"{settings.prometheus_url}/api/v1/query_range"
            params = {
                "query": promql,
                "start": self._get_start_time(time_range),
                "end": "now",
                "step": "1m"  # 1 minute resolution
            }
            
            async with session.get(url, params=params) as response:
                data = await response.json()
        
        # Extract data points
        data_points = []
        if data["status"] == "success" and data["data"]["result"]:
            for value in data["data"]["result"][0]["values"]:
                timestamp, metric_value = value
                data_points.append({
                    "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
                    "value": float(metric_value)
                })
        
        # Detect anomalies
        metadata = self._detect_anomalies(data_points)
        
        return {
            "data_points": data_points,
            "metadata": metadata
        }
    
    def _build_promql(
        self,
        service: str,
        metric_name: str,
        aggregation: str
    ) -> str:
        """
        Build PromQL query string.
        
        Args:
            service: Service name
            metric_name: Metric name
            aggregation: Aggregation function
        
        Returns:
            str: PromQL query
        """
        # Build query based on aggregation type
        if aggregation == "rate":
            return f'rate({metric_name}{{service="{service}", status="500"}}[5m])'
        elif aggregation == "sum":
            return f'sum({metric_name}{{service="{service}"}})'
        else:
            return f'{metric_name}{{service="{service}"}}'
    
    def _get_start_time(self, time_range: str) -> str:
        """
        Calculate start time for query.
        
        Args:
            time_range: Time range string
        
        Returns:
            str: Unix timestamp
        """
        # Parse time range and calculate start time
        hours_map = {
            "last_1h": 1,
            "last_6h": 6,
            "last_24h": 24
        }
        
        hours = hours_map.get(time_range, 1)
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        return str(int(start_time.timestamp()))
    
    def _detect_anomalies(self, data_points: List[Dict]) -> Dict[str, Any]:
        """
        Detect spikes and anomalies in metrics data.
        
        Args:
            data_points: List of time-series data points
        
        Returns:
            Dict with spike detection results
        """
        if len(data_points) < 2:
            return {"spike_detected": False}
        
        # Calculate baseline (average of first half)
        mid_point = len(data_points) // 2
        baseline_values = [p["value"] for p in data_points[:mid_point]]
        baseline = sum(baseline_values) / len(baseline_values) if baseline_values else 0
        
        # Find peak value
        peak_value = max(p["value"] for p in data_points)
        peak_point = next(p for p in data_points if p["value"] == peak_value)
        
        # Detect spike (peak > 3x baseline)
        spike_detected = peak_value > (baseline * 3)
        
        return {
            "spike_detected": spike_detected,
            "spike_start": peak_point["timestamp"] if spike_detected else None,
            "baseline": round(baseline, 2),
            "peak": round(peak_value, 2)
        }


# ============================================================================
# Deployment History Tool
# ============================================================================

class DeployHistoryTool(BaseTool):
    """
    Retrieve recent deployment history.
    
    This tool fetches deployment information to correlate
    incidents with recent code changes.
    """
    
    async def execute(
        self,
        service: str,
        time_range: str = "last_6h",
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get deployment history for a service.
        
        Args:
            service: Service name
            time_range: How far back to look
            limit: Maximum number of deployments to return
        
        Returns:
            Dict containing deployments and correlation data
        """
        # Calculate time window
        hours = int(time_range.replace("last_", "").replace("h", ""))
        since = datetime.utcnow() - timedelta(hours=hours)
        
        # Query deployment API (mock implementation)
        # In production, this would call your actual deployment API
        deployments = await self._fetch_deployments(service, since, limit)
        
        # Correlate with incident timing
        correlation = self._correlate_with_incident(deployments)
        
        return {
            "deployments": deployments,
            "correlation": correlation
        }
    
    async def _fetch_deployments(
        self,
        service: str,
        since: datetime,
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch deployments from deployment API.
        
        Args:
            service: Service name
            since: Start time
            limit: Max results
        
        Returns:
            List of deployment records
        """
        # Mock implementation - replace with actual API call
        # Example: GET /api/deployments?service={service}&since={since}&limit={limit}
        
        return [
            {
                "version": "v2.3.5",
                "timestamp": (datetime.utcnow() - timedelta(minutes=30)).isoformat(),
                "author": "john.doe@company.com",
                "status": "completed",
                "changes": ["Updated database driver", "Fixed memory leak"],
                "rollback_available": True
            }
        ]
    
    def _correlate_with_incident(
        self,
        deployments: List[Dict]
    ) -> Dict[str, Any]:
        """
        Correlate deployments with incident timing.
        
        Args:
            deployments: List of deployments
        
        Returns:
            Dict with correlation information
        """
        if not deployments:
            return {"likely_related": False}
        
        # Get most recent deployment
        latest = deployments[0]
        deploy_time = datetime.fromisoformat(latest["timestamp"])
        
        # Assume incident started ~5 minutes after deployment
        # (In production, use actual incident start time)
        incident_start = datetime.utcnow()
        time_diff = (incident_start - deploy_time).total_seconds() / 60
        
        # Consider related if deployment was within 30 minutes of incident
        likely_related = time_diff < 30
        
        return {
            "incident_start": incident_start.isoformat(),
            "deployment_time": deploy_time.isoformat(),
            "time_diff_minutes": int(time_diff),
            "likely_related": likely_related
        }


# ============================================================================
# Tool Registry
# ============================================================================

class ToolRegistry:
    """
    Registry of all available tools.
    
    Provides a centralized way to access and execute tools.
    """
    
    def __init__(self):
        """Initialize all tools"""
        self.tools = {
            "log_search": LogSearchTool(),
            "metrics_query": MetricsQueryTool(),
            "deploy_history": DeployHistoryTool()
        }
    
    async def execute_tool(
        self,
        tool_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool-specific parameters
        
        Returns:
            Dict with tool output
        
        Raises:
            ValueError: If tool name is not recognized
        """
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool = self.tools[tool_name]
        return await tool._execute_with_timing(**kwargs)
    
    async def execute_multiple(
        self,
        tool_configs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute multiple tools in parallel.
        
        Args:
            tool_configs: List of dicts with 'name' and 'params' keys
        
        Returns:
            Dict mapping tool names to their outputs
        """
        # Create tasks for parallel execution
        tasks = {
            config["name"]: self.execute_tool(
                config["name"],
                **config.get("params", {})
            )
            for config in tool_configs
        }
        
        # Execute in parallel
        results = await asyncio.gather(
            *tasks.values(),
            return_exceptions=True
        )
        
        # Map results back to tool names
        return {
            name: result
            for name, result in zip(tasks.keys(), results)
        }


# ============================================================================
# Usage Example
# ============================================================================

async def main():
    """Example usage of tools"""
    
    # Create tool registry
    registry = ToolRegistry()
    
    # Execute single tool
    log_result = await registry.execute_tool(
        "log_search",
        query="500",
        service="api-gateway",
        time_range="last_1h"
    )
    
    print("✅ Log search result:")
    print(f"   Found {log_result['total_count']} logs")
    print(f"   Pattern: {log_result['error_pattern']}")
    print(f"   Execution time: {log_result['execution_time_ms']}ms")
    
    # Execute multiple tools in parallel
    results = await registry.execute_multiple([
        {"name": "log_search", "params": {"query": "500", "service": "api-gateway"}},
        {"name": "metrics_query", "params": {"service": "api-gateway"}},
        {"name": "deploy_history", "params": {"service": "api-gateway"}}
    ])
    
    print("\n✅ Parallel execution completed:")
    for tool_name, result in results.items():
        print(f"   {tool_name}: {result['execution_time_ms']}ms")


if __name__ == "__main__":
    asyncio.run(main())
