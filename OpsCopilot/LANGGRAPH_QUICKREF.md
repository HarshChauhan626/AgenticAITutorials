# LangChain & LangGraph Quick Reference

## Overview

Quick reference guide for LangChain and LangGraph usage in Ops Copilot.

---

## Key Concepts

### LangGraph State

```python
from typing import TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    # Input
    incident_description: str
    context: Dict[str, str]
    
    # Processing
    iteration_count: int
    tools_to_execute: List[str]
    tool_results: Dict[str, Any]
    
    # Output
    final_response: Dict[str, Any]
```

### Graph Structure

```python
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("node_name", node_function)

# Add edges
workflow.add_edge("node1", "node2")

# Conditional edges
workflow.add_conditional_edges(
    "decision_node",
    decision_function,
    {"continue": "loop_back", "end": END}
)

# Compile
app = workflow.compile()
```

---

## Common Patterns

### 1. LLM with Structured Output

```python
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel

class Response(BaseModel):
    hypothesis: str
    confidence: float

llm = ChatOpenAI(model="gpt-4-turbo-preview")
structured_llm = llm.with_structured_output(Response)

result = structured_llm.invoke("Analyze this incident...")
# result is a Response object
```

### 2. Parallel Tool Execution

```python
import asyncio

async def execute_tools(state: AgentState):
    tasks = [
        tool1.execute(state),
        tool2.execute(state),
        tool3.execute(state)
    ]
    results = await asyncio.gather(*tasks)
    return results
```

### 3. RAG with Hybrid Retrieval

```python
from langchain.retrievers import EnsembleRetriever
from langchain.vectorstores import Pinecone
from langchain.retrievers import BM25Retriever

# Vector retriever
vector_retriever = pinecone_store.as_retriever(search_kwargs={"k": 20})

# Keyword retriever
keyword_retriever = BM25Retriever.from_texts(texts, k=20)

# Combine
ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, keyword_retriever],
    weights=[0.5, 0.5]
)

docs = ensemble.get_relevant_documents(query)
```

### 4. Checkpointing

```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string("checkpoints.db")
app = workflow.compile(checkpointer=memory)

# Execute with thread_id
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke(initial_state, config)

# Resume from checkpoint
continued = app.invoke(None, config)
```

### 5. Streaming

```python
async for event in app.astream(initial_state):
    node_name = list(event.keys())[0]
    node_output = event[node_name]
    print(f"Node {node_name}: {node_output}")
```

---

## Debugging

### Enable Tracing

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ops-copilot"
```

### Print State

```python
def debug_node(state):
    print(f"State: {state}")
    return state

workflow.add_node("debug", debug_node)
```

### Visualize Graph

```python
graph_image = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_image)
```

---

## Performance Tips

1. **Use async/await** for I/O operations
2. **Parallel tool execution** with `asyncio.gather()`
3. **Cache embeddings** in Redis
4. **Limit max_tokens** in LLM calls
5. **Use faster models** for simple tasks (gpt-3.5-turbo)
6. **Batch operations** when possible

---

## Common Errors

### Error: "State key not found"

**Solution:** Initialize all state keys in first node

```python
def init_node(state):
    state["iteration_count"] = 0
    state["tools_used"] = []
    return state
```

### Error: "Infinite loop detected"

**Solution:** Add iteration limit

```python
def should_continue(state):
    if state["iteration_count"] >= 5:
        return "end"
    return "continue"
```

### Error: "JSON decode error"

**Solution:** Use structured output

```python
llm = ChatOpenAI(model="gpt-4-turbo-preview")
structured_llm = llm.with_structured_output(YourModel)
```

---

## File Structure

```
opscopilot/
├── .agent/
│   └── workflows/
│       ├── setup-langchain.md
│       └── debug-langgraph.md
├── LANGGRAPH_ARCHITECTURE.md
├── LANGGRAPH_IMPLEMENTATION.md
└── src/
    ├── graph/
    │   ├── __init__.py
    │   ├── state.py
    │   ├── nodes.py
    │   └── workflow.py
    ├── tools/
    │   ├── log_search.py
    │   ├── metrics_query.py
    │   └── ...
    └── api.py
```

---

## Resources

- **LangChain Docs**: https://python.langchain.com/docs/
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **LangSmith**: https://smith.langchain.com
- **Ops Copilot Docs**:
  - [LANGGRAPH_ARCHITECTURE.md](./LANGGRAPH_ARCHITECTURE.md)
  - [LANGGRAPH_IMPLEMENTATION.md](./LANGGRAPH_IMPLEMENTATION.md)
  - [DATAFLOW.md](./DATAFLOW.md)

---

## Quick Start

```bash
# 1. Setup environment
cd opscopilot
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# 2. Install dependencies
pip install langchain langgraph langchain-openai

# 3. Set environment variables
export OPENAI_API_KEY=sk-...
export LANGCHAIN_TRACING_V2=true

# 4. Run example
python examples/simple_graph.py
```

---

This quick reference provides essential patterns and solutions for working with LangChain and LangGraph in Ops Copilot!
