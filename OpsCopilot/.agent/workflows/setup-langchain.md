---
description: Set up LangChain and LangGraph development environment
---

# Setup LangChain & LangGraph Environment

This workflow guides you through setting up the development environment for LangChain and LangGraph in the Ops Copilot project.

## Prerequisites

- Python 3.11 or higher
- pip package manager
- Virtual environment tool (venv or conda)

## Steps

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Install Core Dependencies

// turbo
```bash
pip install langchain==0.1.0
pip install langgraph==0.0.20
pip install langchain-openai==0.0.5
pip install langchain-community==0.0.13
```

### 3. Install Additional LangChain Components

```bash
# Embeddings and vector stores
pip install langchain-pinecone==0.0.3
pip install sentence-transformers==2.2.2

# Document loaders and text splitters
pip install langchain-text-splitters==0.0.1

# Output parsers
pip install pydantic==2.5.0
```

### 4. Install LangSmith for Observability

```bash
pip install langsmith==0.0.70
```

### 5. Set Environment Variables

Create a `.env` file:

```bash
# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# LangSmith (for tracing)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ops-copilot
LANGCHAIN_API_KEY=ls__...

# Vector Database
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-west1-gcp

# Other services
ELASTICSEARCH_URL=https://elasticsearch:9200
PROMETHEUS_URL=http://prometheus:9090
```

### 6. Verify Installation

```bash
python -c "import langchain; import langgraph; print('LangChain:', langchain.__version__); print('LangGraph:', langgraph.__version__)"
```

Expected output:
```
LangChain: 0.1.0
LangGraph: 0.0.20
```

### 7. Test LangGraph Setup

Create `test_langgraph.py`:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    message: str

def node1(state: State) -> State:
    state["message"] = "Hello from LangGraph!"
    return state

# Create graph
workflow = StateGraph(State)
workflow.add_node("greet", node1)
workflow.set_entry_point("greet")
workflow.add_edge("greet", END)

app = workflow.compile()

# Test
result = app.invoke({"message": ""})
print(result)  # Should print: {'message': 'Hello from LangGraph!'}
```

Run test:
```bash
python test_langgraph.py
```

### 8. Install Development Tools

```bash
# Testing
pip install pytest==7.4.3
pip install pytest-asyncio==0.21.1

# Code quality
pip install black==23.12.0
pip install isort==5.13.2
pip install mypy==1.7.1

# Debugging
pip install ipython==8.18.1
pip install ipdb==0.13.13
```

### 9. Create requirements.txt

```bash
pip freeze > requirements.txt
```

## Verification Checklist

- [ ] Virtual environment activated
- [ ] LangChain and LangGraph installed
- [ ] Environment variables configured
- [ ] Test script runs successfully
- [ ] LangSmith tracing enabled (check https://smith.langchain.com)

## Troubleshooting

**Issue: Import errors**
- Solution: Ensure virtual environment is activated
- Run: `pip install --upgrade langchain langgraph`

**Issue: LangSmith not tracing**
- Solution: Check LANGCHAIN_API_KEY is set correctly
- Verify at: https://smith.langchain.com

**Issue: Pinecone connection fails**
- Solution: Verify PINECONE_API_KEY and PINECONE_ENVIRONMENT
- Test connection: `python -c "import pinecone; pinecone.init(...)"`

## Next Steps

After setup is complete:
1. Review `LANGGRAPH_ARCHITECTURE.md` for system design
2. Follow `LANGGRAPH_IMPLEMENTATION.md` for implementation details
3. Run the development server: `python -m opscopilot.api`
