"""
RAG implementation for pure LangChain.

This module is identical to the LangGraph RAG implementation
since the RAG pipeline is independent of the orchestration method.

For detailed documentation, see: src/langgraph/rag.py
"""

# Import everything from the LangGraph RAG module
# The RAG pipeline is the same regardless of orchestration method
from ..langgraph.rag import (
    init_pinecone,
    hybrid_retrieval,
    _vector_search,
    _keyword_search,
    _reciprocal_rank_fusion,
    _rerank_results,
    _load_runbook_documents,
    _format_results
)

__all__ = [
    "init_pinecone",
    "hybrid_retrieval",
]


# ============================================================================
# Note on RAG Implementation
# ============================================================================

"""
The RAG (Retrieval-Augmented Generation) pipeline is identical for both
LangGraph and pure LangChain implementations because:

1. RAG is a retrieval mechanism, not an orchestration method
2. Both approaches need the same hybrid retrieval (vector + keyword)
3. Both use the same reranking strategy
4. The only difference is HOW the RAG results are used:
   - LangGraph: RAG results stored in state, passed to reasoning node
   - LangChain: RAG results passed as input to analysis chain

Pipeline Steps (same for both):
1. Vector search (Pinecone) - Semantic similarity
2. Keyword search (BM25) - Exact term matching
3. Reciprocal Rank Fusion - Combine results
4. Cross-encoder reranking - Final relevance scoring

Usage in LangChain:
    from .rag import hybrid_retrieval
    
    runbooks = hybrid_retrieval(
        query=incident_description,
        service=service,
        top_k=5
    )
    
    # Format for chain input
    runbooks_str = format_runbooks_for_chain(runbooks)
    
    # Pass to analysis chain
    result = analysis_chain({
        "incident_description": incident_description,
        "evidence": evidence_str,
        "runbooks": runbooks_str
    })
"""


# ============================================================================
# Helper Functions for LangChain Integration
# ============================================================================

def format_runbooks_for_chain(runbooks: list) -> str:
    """
    Format runbooks for use in LangChain prompts.
    
    This is a convenience function that formats the runbook
    results from hybrid_retrieval() into a string suitable
    for inclusion in LLM prompts.
    
    Args:
        runbooks: List of runbook dicts from hybrid_retrieval()
    
    Returns:
        Formatted string for LLM prompt
    
    Example Output:
        [1] "Debugging 500 Errors" (score: 0.94)
            When API gateway returns 500 errors, check...
        
        [2] "Database Timeouts" (score: 0.91)
            Common causes include...
    """
    if not runbooks:
        return "No relevant runbooks found"
    
    formatted = []
    for i, rb in enumerate(runbooks, 1):
        title = rb.get("title", "Untitled")
        score = rb.get("score", 0.0)
        content = rb.get("content", "")[:300]  # First 300 chars
        
        formatted.append(
            f"[{i}] \"{title}\" (relevance: {score:.2f})\n"
            f"    {content}..."
        )
    
    return "\n\n".join(formatted)


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Example: Retrieve and format runbooks for LangChain
    
    # Step 1: Retrieve runbooks
    runbooks = hybrid_retrieval(
        query="API gateway returning 500 errors",
        service="api-gateway",
        top_k=5
    )
    
    print("✅ Retrieved runbooks:")
    print(f"   Count: {len(runbooks)}")
    
    # Step 2: Format for chain
    formatted = format_runbooks_for_chain(runbooks)
    
    print("\n✅ Formatted for LangChain:")
    print(formatted)
