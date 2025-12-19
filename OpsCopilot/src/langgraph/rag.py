"""
Hybrid RAG (Retrieval-Augmented Generation) implementation.

This module implements the hybrid retrieval pipeline that combines:
1. Vector search (Pinecone) - Semantic similarity
2. Keyword search (BM25/Elasticsearch) - Exact term matching
3. Reciprocal Rank Fusion - Combines results
4. Cross-encoder reranking - Final relevance scoring

The pipeline retrieves the most relevant runbooks for incident analysis.
"""

from typing import List, Dict, Any, Optional
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.schema import Document
import pinecone

from ..common.config import settings


# ============================================================================
# Initialize Pinecone
# ============================================================================

def init_pinecone():
    """
    Initialize Pinecone client.
    
    This should be called once at application startup.
    """
    pinecone.init(
        api_key=settings.pinecone_api_key,
        environment=settings.pinecone_environment
    )


# ============================================================================
# Hybrid Retrieval Function
# ============================================================================

def hybrid_retrieval(
    query: str,
    service: Optional[str] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Perform hybrid retrieval to find relevant runbooks.
    
    This function combines vector search and keyword search to find
    the most relevant runbooks for the given query.
    
    Args:
        query: The search query (incident description)
        service: Optional service name to filter results
        top_k: Number of final results to return (default: 5)
    
    Returns:
        List of runbook dictionaries with id, title, content, score
    
    Pipeline Steps:
    1. Generate query embedding (OpenAI)
    2. Vector search in Pinecone (top 20)
    3. Keyword search with BM25 (top 20)
    4. Combine with Reciprocal Rank Fusion
    5. Rerank with cross-encoder (top 5)
    6. Return final results
    
    Example:
        >>> runbooks = hybrid_retrieval(
        ...     query="API returning 500 errors",
        ...     service="api-gateway",
        ...     top_k=5
        ... )
        >>> len(runbooks)
        5
    """
    # Step 1: Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model
    )
    
    # Step 2: Vector search (Pinecone)
    vector_results = _vector_search(
        query=query,
        embeddings=embeddings,
        service=service,
        k=20
    )
    
    # Step 3: Keyword search (BM25)
    keyword_results = _keyword_search(
        query=query,
        service=service,
        k=20
    )
    
    # Step 4: Reciprocal Rank Fusion
    fused_results = _reciprocal_rank_fusion(
        vector_results,
        keyword_results,
        k=60  # RRF parameter
    )
    
    # Step 5: Cross-encoder reranking
    reranked_results = _rerank_results(
        query=query,
        documents=fused_results[:40],  # Rerank top 40
        top_k=top_k
    )
    
    # Step 6: Format and return
    return _format_results(reranked_results)


# ============================================================================
# Vector Search (Pinecone)
# ============================================================================

def _vector_search(
    query: str,
    embeddings: OpenAIEmbeddings,
    service: Optional[str],
    k: int = 20
) -> List[Document]:
    """
    Perform vector similarity search using Pinecone.
    
    Args:
        query: Search query
        embeddings: Embedding model
        service: Optional service filter
        k: Number of results to return
    
    Returns:
        List of Document objects with metadata
    
    How it works:
    1. Converts query to embedding vector (1536 dimensions)
    2. Searches Pinecone index for similar vectors
    3. Filters by service if provided
    4. Returns top K most similar documents
    """
    # Get vector store
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=settings.pinecone_index_name,
        embedding=embeddings
    )
    
    # Build search kwargs
    search_kwargs = {"k": k}
    
    # Add service filter if provided
    if service:
        search_kwargs["filter"] = {"service": {"$eq": service}}
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
    
    # Perform search
    documents = retriever.get_relevant_documents(query)
    
    return documents


# ============================================================================
# Keyword Search (BM25)
# ============================================================================

def _keyword_search(
    query: str,
    service: Optional[str],
    k: int = 20
) -> List[Document]:
    """
    Perform keyword search using BM25 algorithm.
    
    Args:
        query: Search query
        service: Optional service filter
        k: Number of results to return
    
    Returns:
        List of Document objects
    
    How it works:
    1. Loads runbook documents from storage
    2. Creates BM25 retriever with TF-IDF scoring
    3. Filters by service if provided
    4. Returns top K matches based on keyword overlap
    
    Note: In production, this would query Elasticsearch instead
    of loading all documents into memory.
    """
    # Load runbook documents
    # In production, this would query Elasticsearch
    documents = _load_runbook_documents(service=service)
    
    # Create BM25 retriever
    retriever = BM25Retriever.from_documents(
        documents,
        k=k
    )
    
    # Perform search
    results = retriever.get_relevant_documents(query)
    
    return results


# ============================================================================
# Reciprocal Rank Fusion
# ============================================================================

def _reciprocal_rank_fusion(
    vector_results: List[Document],
    keyword_results: List[Document],
    k: int = 60
) -> List[Document]:
    """
    Combine results from multiple retrievers using RRF.
    
    Reciprocal Rank Fusion (RRF) is a method to combine rankings
    from different retrieval systems. It gives higher scores to
    documents that rank well across multiple systems.
    
    Args:
        vector_results: Results from vector search
        keyword_results: Results from keyword search
        k: RRF constant (default: 60)
    
    Returns:
        Combined and sorted list of documents
    
    Formula:
        RRF_score(d) = sum(1 / (k + rank_i(d)))
        where rank_i(d) is the rank of document d in retriever i
    
    How it works:
    1. Assign scores to each document based on its rank in each list
    2. Sum scores across all retrievers
    3. Sort by combined score
    4. Return merged list
    """
    from collections import defaultdict
    
    scores = defaultdict(float)
    docs = {}
    
    # Score vector results
    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc.metadata.get("id", str(hash(doc.page_content)))
        scores[doc_id] += 1 / (k + rank)
        docs[doc_id] = doc
    
    # Score keyword results
    for rank, doc in enumerate(keyword_results, start=1):
        doc_id = doc.metadata.get("id", str(hash(doc.page_content)))
        scores[doc_id] += 1 / (k + rank)
        if doc_id not in docs:
            docs[doc_id] = doc
    
    # Sort by combined score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return documents in ranked order
    return [docs[doc_id] for doc_id, _ in ranked]


# ============================================================================
# Cross-Encoder Reranking
# ============================================================================

def _rerank_results(
    query: str,
    documents: List[Document],
    top_k: int = 5
) -> List[Document]:
    """
    Rerank documents using a cross-encoder model.
    
    Cross-encoders compute relevance scores by encoding the query
    and document together, providing more accurate relevance than
    bi-encoders (used in vector search).
    
    Args:
        query: Original search query
        documents: Documents to rerank
        top_k: Number of top results to return
    
    Returns:
        Reranked list of top K documents
    
    How it works:
    1. Creates query-document pairs
    2. Passes each pair through cross-encoder model
    3. Gets relevance score for each pair
    4. Sorts by relevance score
    5. Returns top K
    
    Model: cross-encoder/ms-marco-MiniLM-L-12-v2
    - Trained on MS MARCO dataset
    - Fast inference (~50ms per pair)
    - High accuracy for relevance scoring
    """
    # Create cross-encoder reranker
    compressor = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
        top_n=top_k
    )
    
    # Create base retriever (returns all documents)
    class ListRetriever:
        def __init__(self, docs):
            self.docs = docs
        
        def get_relevant_documents(self, query):
            return self.docs
    
    base_retriever = ListRetriever(documents)
    
    # Create compression retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    # Rerank
    reranked = compression_retriever.get_relevant_documents(query)
    
    return reranked


# ============================================================================
# Helper Functions
# ============================================================================

def _load_runbook_documents(service: Optional[str] = None) -> List[Document]:
    """
    Load runbook documents from storage.
    
    In production, this would query Elasticsearch or a database.
    For now, returns mock documents.
    
    Args:
        service: Optional service filter
    
    Returns:
        List of Document objects
    """
    # Mock implementation
    # In production, query Elasticsearch:
    # GET /runbooks/_search?q=service:{service}
    
    mock_runbooks = [
        Document(
            page_content="When API gateway returns 500 errors, check database connection pool settings...",
            metadata={
                "id": "rb_001",
                "title": "Debugging 500 Errors in API Gateway",
                "service": "api-gateway",
                "url": "https://runbooks.company.com/rb_001"
            }
        ),
        Document(
            page_content="Database connection timeouts are often caused by pool exhaustion...",
            metadata={
                "id": "rb_042",
                "title": "Database Connection Timeouts",
                "service": "api-gateway",
                "url": "https://runbooks.company.com/rb_042"
            }
        )
    ]
    
    # Filter by service if provided
    if service:
        mock_runbooks = [
            doc for doc in mock_runbooks
            if doc.metadata.get("service") == service
        ]
    
    return mock_runbooks


def _format_results(documents: List[Document]) -> List[Dict[str, Any]]:
    """
    Format Document objects into dictionaries.
    
    Args:
        documents: List of Document objects
    
    Returns:
        List of dictionaries with id, title, content, score, url
    """
    results = []
    
    for doc in documents:
        results.append({
            "id": doc.metadata.get("id", "unknown"),
            "title": doc.metadata.get("title", "Untitled"),
            "content": doc.page_content,
            "score": doc.metadata.get("score", 0.0),
            "url": doc.metadata.get("url", "")
        })
    
    return results


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Example: Perform hybrid retrieval
    runbooks = hybrid_retrieval(
        query="API gateway returning 500 errors since 2pm",
        service="api-gateway",
        top_k=5
    )
    
    print("✅ Hybrid retrieval completed:")
    print(f"   Found {len(runbooks)} runbooks")
    
    for i, rb in enumerate(runbooks, 1):
        print(f"\n{i}. {rb['title']} (score: {rb['score']:.2f})")
        print(f"   {rb['content'][:100]}...")
