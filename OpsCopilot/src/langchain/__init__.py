"""
Pure LangChain implementation of Ops Copilot.

This package contains the sequential chain-based implementation
using pure LangChain without LangGraph.
"""

from .orchestrator import OpscopilotChain, analyze_incident

__all__ = ["OpscopilotChain", "analyze_incident"]
