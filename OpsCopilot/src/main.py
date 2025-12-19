"""
Main entry point for Ops Copilot application.

This module provides the main entry point for running the application
in different modes: API server, CLI, or interactive mode.
"""

import asyncio
import argparse
import sys
from typing import Optional

from .common.config import settings, setup_environment, validate_configuration


# ============================================================================
# CLI Mode
# ============================================================================

async def run_cli(
    incident_description: str,
    service: str,
    environment: str = "production",
    implementation: str = "langgraph"
):
    """
    Run analysis in CLI mode.
    
    Args:
        incident_description: Description of the incident
        service: Service name
        environment: Environment name
        implementation: Which implementation to use (langgraph or langchain)
    """
    print("=" * 80)
    print("🔍 Ops Copilot - CLI Mode")
    print("=" * 80)
    print(f"Incident: {incident_description}")
    print(f"Service: {service}")
    print(f"Environment: {environment}")
    print(f"Implementation: {implementation}")
    print("=" * 80)
    
    # Select implementation
    if implementation == "langgraph":
        from .langgraph.workflow import analyze_incident
    else:
        from .langchain.orchestrator import analyze_incident
    
    # Run analysis
    result = await analyze_incident(
        incident_description=incident_description,
        context={
            "service": service,
            "environment": environment
        }
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("📊 ANALYSIS RESULTS")
    print("=" * 80)
    print(f"\nRequest ID: {result['request_id']}")
    print(f"Latency: {result['latency_ms']}ms")
    
    print(f"\n🎯 HYPOTHESIS:")
    print(f"   {result['result']['hypothesis']}")
    print(f"\n📈 CONFIDENCE: {result['result']['confidence']:.2f}")
    
    print(f"\n🔧 NEXT ACTIONS ({len(result['result']['next_actions'])}):")
    for i, action in enumerate(result['result']['next_actions'], 1):
        print(f"   {i}. [{action['priority'].upper()}] {action['action']}")
        print(f"      Time: {action['estimated_time']}")
        print(f"      Why: {action['rationale']}")
    
    if result['result'].get('commands'):
        print(f"\n💻 COMMANDS ({len(result['result']['commands'])}):")
        for i, cmd in enumerate(result['result']['commands'], 1):
            safe = "✅" if cmd['safe_to_run'] else "⚠️"
            print(f"   {i}. {safe} {cmd['description']}")
            print(f"      $ {cmd['command']}")
    
    print(f"\n📚 EVIDENCE:")
    print(f"   Tools used: {', '.join(result['metadata']['tools_used'])}")
    print(f"   Runbooks retrieved: {result['metadata']['runbooks_retrieved']}")
    print(f"   Citations: {len(result['result']['citations'])}")
    
    if 'total_tokens' in result['metadata']:
        print(f"\n💰 COST:")
        print(f"   Tokens: {result['metadata']['total_tokens']}")
        print(f"   Cost: ${result['metadata']['total_cost']:.4f}")
    
    print("\n" + "=" * 80)


# ============================================================================
# Interactive Mode
# ============================================================================

async def run_interactive():
    """
    Run in interactive mode.
    
    Allows user to analyze multiple incidents interactively.
    """
    print("=" * 80)
    print("🤖 Ops Copilot - Interactive Mode")
    print("=" * 80)
    print("Type 'exit' or 'quit' to exit")
    print("=" * 80)
    
    while True:
        # Get incident description
        print("\n📝 Describe the incident:")
        incident_description = input("> ").strip()
        
        if incident_description.lower() in ['exit', 'quit']:
            print("👋 Goodbye!")
            break
        
        if not incident_description:
            print("❌ Please provide an incident description")
            continue
        
        # Get service
        print("\n🔧 Service name:")
        service = input("> ").strip() or "unknown"
        
        # Get implementation choice
        print("\n⚙️  Implementation (langgraph/langchain) [langgraph]:")
        implementation = input("> ").strip() or "langgraph"
        
        # Run analysis
        try:
            await run_cli(
                incident_description=incident_description,
                service=service,
                implementation=implementation
            )
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")


# ============================================================================
# API Server Mode
# ============================================================================

def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Run the FastAPI server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
    """
    import uvicorn
    
    print("=" * 80)
    print("🚀 Starting Ops Copilot API Server")
    print("=" * 80)
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Docs: http://{host}:{port}/docs")
    print("=" * 80)
    
    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """
    Main entry point.
    
    Parses command-line arguments and runs in appropriate mode.
    """
    # Setup environment
    setup_environment()
    
    # Validate configuration
    try:
        validate_configuration()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\nPlease create a .env file with required variables:")
        print("  - OPENAI_API_KEY")
        print("  - PINECONE_API_KEY")
        print("  - POSTGRES_PASSWORD")
        print("\nSee .env.example for a complete template.")
        sys.exit(1)
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Ops Copilot - AI-powered incident response assistant"
    )
    
    parser.add_argument(
        "mode",
        choices=["api", "cli", "interactive"],
        help="Mode to run in: api (REST server), cli (single analysis), interactive (REPL)"
    )
    
    parser.add_argument(
        "--incident",
        "-i",
        help="Incident description (for CLI mode)"
    )
    
    parser.add_argument(
        "--service",
        "-s",
        help="Service name (for CLI mode)"
    )
    
    parser.add_argument(
        "--environment",
        "-e",
        default="production",
        help="Environment (for CLI mode)"
    )
    
    parser.add_argument(
        "--implementation",
        choices=["langgraph", "langchain"],
        default="langgraph",
        help="Implementation to use (for CLI mode)"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (for API mode)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (for API mode)"
    )
    
    args = parser.parse_args()
    
    # Run in appropriate mode
    if args.mode == "api":
        run_api_server(host=args.host, port=args.port)
    
    elif args.mode == "cli":
        if not args.incident or not args.service:
            print("❌ Error: --incident and --service are required for CLI mode")
            sys.exit(1)
        
        asyncio.run(run_cli(
            incident_description=args.incident,
            service=args.service,
            environment=args.environment,
            implementation=args.implementation
        ))
    
    elif args.mode == "interactive":
        asyncio.run(run_interactive())


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    main()
