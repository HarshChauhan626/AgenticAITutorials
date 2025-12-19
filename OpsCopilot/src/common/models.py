"""
Common data models and schemas used across Ops Copilot.

This module defines:
- Request/Response schemas
- State models
- Tool output schemas
- Database models
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class Priority(str, Enum):
    """Priority levels for actions and incidents"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IncidentStatus(str, Enum):
    """Incident lifecycle status"""
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"


class ToolName(str, Enum):
    """Available tool names"""
    LOG_SEARCH = "log_search"
    METRICS_QUERY = "metrics_query"
    DEPLOY_HISTORY = "deploy_history"
    RUNBOOK_SEARCH = "runbook_search"
    TICKET_CREATE = "ticket_create"


# ============================================================================
# Request Models
# ============================================================================

class AnalyzeRequest(BaseModel):
    """
    Request schema for incident analysis endpoint.
    
    This is the main input to the Ops Copilot system.
    """
    
    incident_description: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Detailed description of the incident"
    )
    
    incident_id: Optional[str] = Field(
        None,
        regex=r'^INC-\d+$',
        description="Optional incident ID (format: INC-12345)"
    )
    
    context: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional context (service, environment, region, etc.)"
    )
    
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional configuration (include_commands, max_citations, etc.)"
    )
    
    @validator('incident_description')
    def validate_description(cls, v):
        """Ensure description is not empty after stripping whitespace"""
        if not v.strip():
            raise ValueError("Description cannot be empty")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "incident_description": "API gateway returning 500 errors since 2pm",
                "context": {
                    "service": "api-gateway",
                    "environment": "production",
                    "severity": "critical"
                },
                "options": {
                    "include_commands": True,
                    "max_citations": 5
                }
            }
        }


# ============================================================================
# Response Models
# ============================================================================

class NextAction(BaseModel):
    """
    A single recommended action to take.
    
    Represents one step in the remediation process.
    """
    
    action: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Description of the action to take"
    )
    
    priority: Priority = Field(
        ...,
        description="Priority level of this action"
    )
    
    estimated_time: str = Field(
        ...,
        description="Estimated time to complete (e.g., '5 minutes', '1 hour')"
    )
    
    rationale: str = Field(
        ...,
        description="Why this action is recommended"
    )


class Command(BaseModel):
    """
    A specific command to execute.
    
    Includes safety information to prevent destructive operations.
    """
    
    description: str = Field(
        ...,
        description="Human-readable description of what the command does"
    )
    
    command: str = Field(
        ...,
        description="The actual command to run"
    )
    
    safe_to_run: bool = Field(
        ...,
        description="Whether this command is safe to auto-execute"
    )


class Citation(BaseModel):
    """
    A citation for evidence or recommendations.
    
    Ensures all claims are backed by data sources.
    """
    
    source: str = Field(
        ...,
        regex=r'^(logs|metrics|deployment|runbook)$',
        description="Type of source (logs, metrics, deployment, runbook)"
    )
    
    reference: str = Field(
        ...,
        description="Reference identifier (query, URL, ID, etc.)"
    )
    
    excerpt: str = Field(
        ...,
        max_length=500,
        description="Relevant excerpt from the source"
    )
    
    timestamp: Optional[datetime] = Field(
        None,
        description="Timestamp of the cited data (if applicable)"
    )


class RelatedIncident(BaseModel):
    """
    A previously resolved similar incident.
    
    Helps learn from past resolutions.
    """
    
    incident_id: str = Field(
        ...,
        description="ID of the related incident"
    )
    
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score (0-1)"
    )
    
    resolution: str = Field(
        ...,
        description="How the incident was resolved"
    )


class AnalyzeResponse(BaseModel):
    """
    Complete response from incident analysis.
    
    This is the main output of the Ops Copilot system.
    """
    
    request_id: str = Field(
        ...,
        description="Unique request identifier for tracking"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the analysis was completed"
    )
    
    latency_ms: int = Field(
        ...,
        description="Total processing time in milliseconds"
    )
    
    result: 'AnalysisResult' = Field(
        ...,
        description="The analysis result"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (tools_used, cache_hit, etc.)"
    )


class AnalysisResult(BaseModel):
    """
    The core analysis result.
    
    Contains hypothesis, actions, commands, and citations.
    """
    
    hypothesis: str = Field(
        ...,
        min_length=20,
        max_length=1000,
        description="Root cause hypothesis"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the hypothesis (0-1)"
    )
    
    reasoning: Optional[str] = Field(
        None,
        description="Detailed reasoning behind the hypothesis"
    )
    
    next_actions: List[NextAction] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="Prioritized list of next actions"
    )
    
    commands: List[Command] = Field(
        default_factory=list,
        max_items=10,
        description="Specific commands to execute"
    )
    
    citations: List[Citation] = Field(
        ...,
        min_items=1,
        description="Citations for all claims"
    )
    
    related_incidents: List[RelatedIncident] = Field(
        default_factory=list,
        description="Similar past incidents"
    )
    
    @validator('next_actions')
    def validate_high_priority_action(cls, v):
        """Ensure at least one high-priority action exists"""
        high_priority = [a for a in v if a.priority == Priority.HIGH]
        if not high_priority:
            raise ValueError("Must have at least one high-priority action")
        return v


# ============================================================================
# Tool Output Models
# ============================================================================

class LogSearchOutput(BaseModel):
    """Output from log search tool"""
    
    success: bool = Field(..., description="Whether the search succeeded")
    
    logs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of log entries"
    )
    
    error_pattern: Optional[str] = Field(
        None,
        description="Detected error pattern (e.g., 'Database timeout (78%)')"
    )
    
    total_count: int = Field(
        default=0,
        description="Total number of matching logs"
    )
    
    execution_time_ms: int = Field(
        ...,
        description="Tool execution time in milliseconds"
    )


class MetricsQueryOutput(BaseModel):
    """Output from metrics query tool"""
    
    success: bool = Field(..., description="Whether the query succeeded")
    
    data_points: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Time-series data points"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata (spike_detected, baseline, peak, etc.)"
    )
    
    execution_time_ms: int = Field(
        ...,
        description="Tool execution time in milliseconds"
    )


class DeployHistoryOutput(BaseModel):
    """Output from deployment history tool"""
    
    success: bool = Field(..., description="Whether the query succeeded")
    
    deployments: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of recent deployments"
    )
    
    correlation: Optional[Dict[str, Any]] = Field(
        None,
        description="Correlation with incident timing"
    )
    
    execution_time_ms: int = Field(
        ...,
        description="Tool execution time in milliseconds"
    )


class RunbookSearchOutput(BaseModel):
    """Output from runbook search tool"""
    
    success: bool = Field(..., description="Whether the search succeeded")
    
    runbooks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Retrieved runbooks"
    )
    
    total_results: int = Field(
        default=0,
        description="Total number of matching runbooks"
    )
    
    execution_time_ms: int = Field(
        ...,
        description="Tool execution time in milliseconds"
    )


# ============================================================================
# State Models (for LangGraph)
# ============================================================================

class AgentState(BaseModel):
    """
    State schema for LangGraph workflow.
    
    This state is passed between all nodes in the graph and
    accumulates information throughout the execution.
    """
    
    # Input
    incident_description: str
    context: Dict[str, str] = Field(default_factory=dict)
    correlation_id: str
    
    # Iteration tracking
    iteration_count: int = 0
    start_time: float = 0.0
    
    # Tool execution
    tools_to_execute: List[str] = Field(default_factory=list)
    tool_results: Dict[str, Any] = Field(default_factory=dict)
    tools_used: List[str] = Field(default_factory=list)
    
    # RAG retrieval
    runbooks: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Evidence
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    
    # LLM interaction
    messages: List[Dict[str, str]] = Field(default_factory=list)
    current_hypothesis: Optional[str] = None
    confidence: Optional[float] = None
    
    # Output
    final_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# Database Models
# ============================================================================

class IncidentRecord(BaseModel):
    """
    Database model for storing incident records.
    
    Used for historical analysis and similarity search.
    """
    
    incident_id: str
    description: str
    hypothesis: str
    status: IncidentStatus
    severity: Priority
    affected_services: List[str]
    embedding: Optional[List[float]] = None  # For similarity search
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True  # Enable ORM compatibility


# ============================================================================
# Update forward references
# ============================================================================

AnalyzeResponse.update_forward_refs()


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Example: Create a sample request
    request = AnalyzeRequest(
        incident_description="API gateway returning 500 errors since 2pm",
        context={
            "service": "api-gateway",
            "environment": "production"
        }
    )
    
    print("✅ Request created:")
    print(request.json(indent=2))
    
    # Example: Create a sample response
    response = AnalyzeResponse(
        request_id="req_abc123",
        latency_ms=8523,
        result=AnalysisResult(
            hypothesis="Database driver incompatibility from v2.3.5 deployment",
            confidence=0.85,
            next_actions=[
                NextAction(
                    action="Rollback to v2.3.4 immediately",
                    priority=Priority.HIGH,
                    estimated_time="5 minutes",
                    rationale="Fastest path to mitigation"
                )
            ],
            citations=[
                Citation(
                    source="logs",
                    reference="Elasticsearch query",
                    excerpt="Database connection timeout (78% of errors)"
                )
            ]
        ),
        metadata={"tools_used": ["log_search", "metrics_query"]}
    )
    
    print("\n✅ Response created:")
    print(response.json(indent=2))
