---
description: Debug and trace LangGraph execution flows
---

# Debug LangGraph Workflows

This workflow helps you debug and trace LangGraph execution flows in the Ops Copilot system.

## Prerequisites

- LangChain and LangGraph installed
- LangSmith account configured
- Ops Copilot development environment running

## Debugging Tools

### 1. Enable Verbose Logging

```python
import logging

# Set logging level
logging.basicConfig(level=logging.DEBUG)

# LangChain specific logging
logging.getLogger("langchain").setLevel(logging.DEBUG)
logging.getLogger("langgraph").setLevel(logging.DEBUG)
```

### 2. Use LangSmith Tracing

```python
import os

# Enable tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ops-copilot-debug"
os.environ["LANGCHAIN_API_KEY"] = "your_api_key"

# All executions are now traced
app = create_ops_copilot_graph()
result = app.invoke(initial_state)

# View trace at: https://smith.langchain.com
```

### 3. Print State at Each Node

```python
def debug_node(state: AgentState) -> AgentState:
    """Wrapper to print state"""
    print(f"\n{'='*50}")
    print(f"Node: {inspect.currentframe().f_code.co_name}")
    print(f"Iteration: {state.get('iteration_count', 0)}")
    print(f"Tools used: {state.get('tools_used', [])}")
    print(f"Confidence: {state.get('confidence', 'N/A')}")
    print(f"{'='*50}\n")
    return state

# Add to graph
workflow.add_node("debug", debug_node)
workflow.add_edge("reasoning", "debug")
workflow.add_edge("debug", "response_generation")
```

### 4. Visualize Graph Structure

```python
from IPython.display import Image, display

# Get graph visualization
graph_image = app.get_graph().draw_mermaid_png()

# Save to file
with open("graph_structure.png", "wb") as f:
    f.write(graph_image)

# Or display in Jupyter
display(Image(graph_image))
```

### 5. Step-by-Step Execution

```python
# Execute one node at a time
config = {"configurable": {"thread_id": "debug-123"}}

# Step 1
state = app.invoke(initial_state, config)
print("After parse_input:", state)

# Step 2
state = app.invoke(None, config)  # Continue from checkpoint
print("After planning:", state)

# And so on...
```

### 6. Inspect Checkpoints

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Get all checkpoints
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# List checkpoints for a thread
checkpoints = checkpointer.list(config)
for checkpoint in checkpoints:
    print(f"Checkpoint: {checkpoint.id}")
    print(f"  State: {checkpoint.state}")
```

## Common Debugging Scenarios

### Scenario 1: Tool Execution Fails

**Symptoms:**
- Tool returns error
- State shows `tool_results` with `success: False`

**Debug Steps:**

1. Check tool logs:
```python
if not state["tool_results"]["log_search"]["success"]:
    error = state["tool_results"]["log_search"]["error"]
    print(f"Log search failed: {error}")
```

2. Test tool independently:
```python
from opscopilot.tools import LogSearchTool

tool = LogSearchTool()
result = await tool.execute({
    "query": "test",
    "time_range": "last_1h"
})
print(result)
```

3. Check API connectivity:
```bash
curl -X GET "http://elasticsearch:9200/_cluster/health"
```

### Scenario 2: Infinite Loop

**Symptoms:**
- Graph doesn't terminate
- `iteration_count` keeps increasing

**Debug Steps:**

1. Add iteration limit check:
```python
def should_continue(state: AgentState) -> str:
    if state["iteration_count"] >= 5:
        print(f"WARNING: Max iterations reached!")
        return "generate"
    # ... rest of logic
```

2. Check decision logic:
```python
# Add logging to decision node
def should_continue(state: AgentState) -> str:
    decision = "continue" if state["confidence"] < 0.8 else "generate"
    print(f"Decision: {decision} (confidence: {state['confidence']})")
    return decision
```

3. Review state evolution:
```python
# Track state changes
previous_state = None
for event in app.stream(initial_state):
    current_state = list(event.values())[0]
    if previous_state:
        print(f"State changed: {diff(previous_state, current_state)}")
    previous_state = current_state
```

### Scenario 3: LLM Returns Invalid JSON

**Symptoms:**
- JSON parsing error
- Pydantic validation fails

**Debug Steps:**

1. Print raw LLM output:
```python
response = await llm.ainvoke(messages)
print(f"Raw LLM output:\n{response.content}")

try:
    parsed = json.loads(response.content)
except json.JSONDecodeError as e:
    print(f"JSON parse error: {e}")
    print(f"Problematic content: {response.content[max(0, e.pos-50):e.pos+50]}")
```

2. Use structured output:
```python
from langchain.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=YourModel)
structured_llm = llm.with_structured_output(YourModel)

# This enforces schema
result = structured_llm.invoke(prompt)
```

3. Add retry logic:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def call_llm_with_retry(messages):
    response = await llm.ainvoke(messages)
    # Validate before returning
    parsed = json.loads(response.content)
    return parsed
```

### Scenario 4: Slow Performance

**Symptoms:**
- Request takes > 10s
- Timeout errors

**Debug Steps:**

1. Profile execution time:
```python
import time

class TimingNode:
    def __init__(self, node_func):
        self.node_func = node_func
        self.name = node_func.__name__
    
    def __call__(self, state):
        start = time.time()
        result = self.node_func(state)
        elapsed = time.time() - start
        print(f"{self.name}: {elapsed:.2f}s")
        return result

# Wrap nodes
workflow.add_node("planning", TimingNode(planning_node))
```

2. Check parallel execution:
```python
# Ensure tools run in parallel
async def tool_execution_node(state):
    tasks = [tool1(state), tool2(state), tool3(state)]
    results = await asyncio.gather(*tasks)  # Parallel
    # NOT: for task in tasks: await task  # Sequential!
```

3. Optimize LLM calls:
```python
# Reduce max_tokens
llm = ChatOpenAI(model="gpt-4-turbo-preview", max_tokens=1000)

# Use faster model for planning
planning_llm = ChatOpenAI(model="gpt-3.5-turbo")
reasoning_llm = ChatOpenAI(model="gpt-4-turbo-preview")
```

## Advanced Debugging

### Custom Callbacks

```python
from langchain.callbacks.base import BaseCallbackHandler

class DebugCallback(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"\n🔵 Chain started: {serialized.get('name')}")
        print(f"   Inputs: {inputs}")
    
    def on_chain_end(self, outputs, **kwargs):
        print(f"✅ Chain ended")
        print(f"   Outputs: {outputs}\n")
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"🤖 LLM call started")
        print(f"   Prompt: {prompts[0][:100]}...")
    
    def on_llm_end(self, response, **kwargs):
        print(f"✅ LLM call ended")
        print(f"   Response: {response.generations[0][0].text[:100]}...")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"🔧 Tool started: {serialized.get('name')}")
    
    def on_tool_end(self, output, **kwargs):
        print(f"✅ Tool ended: {output[:100]}...")

# Use callback
app.invoke(initial_state, config={"callbacks": [DebugCallback()]})
```

### Breakpoint Debugging

```python
import ipdb

def reasoning_node(state: AgentState) -> AgentState:
    # Set breakpoint
    ipdb.set_trace()
    
    # Inspect state
    # > print(state["evidence"])
    # > print(state["confidence"])
    
    # Continue execution
    # > c
    
    # ... rest of node logic
```

### Memory Profiling

```python
import tracemalloc

tracemalloc.start()

# Execute graph
result = app.invoke(initial_state)

# Get memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory: {current / 1024 / 1024:.2f} MB")
print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")

tracemalloc.stop()
```

## Debugging Checklist

- [ ] LangSmith tracing enabled
- [ ] Verbose logging configured
- [ ] State printed at each node
- [ ] Graph structure visualized
- [ ] Tool execution tested independently
- [ ] Iteration limits enforced
- [ ] LLM output validated
- [ ] Performance profiled
- [ ] Checkpoints inspected

## Useful Commands

```bash
# View LangSmith traces
open https://smith.langchain.com/projects/ops-copilot-debug

# Check graph structure
python -c "from opscopilot.graph import create_ops_copilot_graph; app = create_ops_copilot_graph(); print(app.get_graph().draw_ascii())"

# Run with debug logging
LANGCHAIN_VERBOSE=true python -m opscopilot.api

# Profile execution
python -m cProfile -o profile.stats -m opscopilot.api
python -m pstats profile.stats
```

## Next Steps

- Review execution traces in LangSmith
- Optimize slow nodes
- Add error handling
- Implement retry logic
- Set up monitoring alerts
