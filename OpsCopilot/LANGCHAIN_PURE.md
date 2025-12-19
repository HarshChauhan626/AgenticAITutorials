# Pure LangChain Implementation (Without LangGraph)

## Overview

This document shows how to implement the Ops Copilot system using **pure LangChain** without LangGraph. This approach uses chains, agents, and sequential execution instead of state machines.

---

## 1. Architecture Comparison

### LangGraph Approach (State Machine)
```
Input → State → Node1 → Node2 → Node3 → ... → Output
         ↑_____________________________↓ (loops possible)
```

### Pure LangChain Approach (Sequential Chains)
```
Input → Chain1 → Chain2 → Chain3 → ... → Output
        (linear flow with conditional logic)
```

---

## 2. Core Components

### 2.1 Using LangChain Agents

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define tools
tools = [
    Tool(
        name="log_search",
        func=log_search_function,
        description="Search application logs for errors and patterns"
    ),
    Tool(
        name="metrics_query",
        func=metrics_query_function,
        description="Query time-series metrics from Prometheus"
    ),
    Tool(
        name="deploy_history",
        func=deploy_history_function,
        description="Get recent deployment history"
    ),
    Tool(
        name="runbook_search",
        func=runbook_search_function,
        description="Search runbook corpus for remediation steps"
    )
]

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert SRE assistant analyzing production incidents.
    
    Your goal is to:
    1. Use available tools to gather evidence
    2. Form a hypothesis about the root cause
    3. Suggest actionable remediation steps
    
    Available tools: {tools}
    Tool names: {tool_names}
    """),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create agent
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.1)
agent = create_openai_tools_agent(llm, tools, prompt)

# Create executor with iteration limit
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,  # Prevent infinite loops
    max_execution_time=30,  # 30 second timeout
    verbose=True,
    return_intermediate_steps=True
)

# Execute
result = agent_executor.invoke({
    "input": "API gateway returning 500 errors since 2pm"
})
```

---

## 3. Sequential Chain Approach

### 3.1 Define Individual Chains

```python
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

# Chain 1: Planning - Decide which tools to use
planning_prompt = PromptTemplate(
    input_variables=["incident_description", "service"],
    template="""
    Incident: {incident_description}
    Service: {service}
    
    Available tools:
    1. log_search - Search logs
    2. metrics_query - Query metrics
    3. deploy_history - Get deployments
    4. runbook_search - Search runbooks
    
    Which tools should we use? List them as comma-separated values.
    Tools to use:
    """
)

planning_chain = LLMChain(
    llm=llm,
    prompt=planning_prompt,
    output_key="tools_to_use"
)

# Chain 2: Tool Execution (custom chain)
class ToolExecutionChain(Chain):
    """Custom chain to execute tools"""
    
    input_key: str = "tools_to_use"
    output_key: str = "tool_results"
    
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        tools_list = inputs[self.input_key].split(",")
        results = {}
        
        for tool_name in tools_list:
            tool_name = tool_name.strip()
            if tool_name == "log_search":
                results["logs"] = self.execute_log_search(inputs)
            elif tool_name == "metrics_query":
                results["metrics"] = self.execute_metrics_query(inputs)
            elif tool_name == "deploy_history":
                results["deployments"] = self.execute_deploy_history(inputs)
            elif tool_name == "runbook_search":
                results["runbooks"] = self.execute_runbook_search(inputs)
        
        return {self.output_key: results}
    
    @property
    def _chain_type(self) -> str:
        return "tool_execution"

tool_chain = ToolExecutionChain()

# Chain 3: Evidence Analysis
analysis_prompt = PromptTemplate(
    input_variables=["incident_description", "tool_results"],
    template="""
    Incident: {incident_description}
    
    Evidence gathered:
    {tool_results}
    
    Analyze the evidence and form a hypothesis about the root cause.
    
    Hypothesis:
    """
)

analysis_chain = LLMChain(
    llm=llm,
    prompt=analysis_prompt,
    output_key="hypothesis"
)

# Chain 4: Action Generation
action_prompt = PromptTemplate(
    input_variables=["hypothesis", "tool_results"],
    template="""
    Root cause hypothesis: {hypothesis}
    
    Evidence: {tool_results}
    
    Suggest 3-5 prioritized next actions to resolve this incident.
    Format as JSON array.
    
    Actions:
    """
)

action_chain = LLMChain(
    llm=llm,
    prompt=action_prompt,
    output_key="next_actions"
)

# Combine into sequential chain
overall_chain = SequentialChain(
    chains=[planning_chain, tool_chain, analysis_chain, action_chain],
    input_variables=["incident_description", "service"],
    output_variables=["tools_to_use", "tool_results", "hypothesis", "next_actions"],
    verbose=True
)

# Execute
result = overall_chain({
    "incident_description": "API gateway returning 500 errors",
    "service": "api-gateway"
})
```

---

## 4. RAG Implementation with LangChain

### 4.1 Hybrid Retrieval

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Vector store retriever
vector_store = Pinecone.from_existing_index(
    index_name="runbooks",
    embedding=embeddings
)
vector_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 20}
)

# Keyword retriever
documents = load_runbook_documents()  # Load from Elasticsearch
keyword_retriever = BM25Retriever.from_documents(documents, k=20)

# Ensemble retriever (Reciprocal Rank Fusion)
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, keyword_retriever],
    weights=[0.5, 0.5]
)

# Add reranking
compressor = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
    top_n=5
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever
)

# Use in chain
docs = compression_retriever.get_relevant_documents(
    "API gateway returning 500 errors"
)
```

### 4.2 RAG Chain

```python
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Create RAG prompt
rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Use the following runbooks to help diagnose the incident:
    
    {context}
    
    Incident: {question}
    
    Based on the runbooks, what are the likely causes and remediation steps?
    
    Answer:
    """
)

# Create RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever,
    chain_type_kwargs={"prompt": rag_prompt},
    return_source_documents=True
)

# Execute
result = rag_chain({
    "query": "API gateway returning 500 errors since 2pm"
})

print(result["result"])  # Answer
print(result["source_documents"])  # Retrieved runbooks
```

---

## 5. Complete Implementation

### 5.1 Main Orchestration Class

```python
from typing import Dict, Any, List
import asyncio
from langchain.callbacks import get_openai_callback

class OpscopilotChain:
    """Main orchestration using pure LangChain"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.1)
        self.embeddings = OpenAIEmbeddings()
        self.setup_retrievers()
        self.setup_tools()
        self.setup_chains()
    
    def setup_retrievers(self):
        """Setup RAG retrievers"""
        # Vector store
        vector_store = Pinecone.from_existing_index(
            index_name="runbooks",
            embedding=self.embeddings
        )
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
        
        # Keyword retriever
        documents = self.load_runbooks()
        keyword_retriever = BM25Retriever.from_documents(documents, k=20)
        
        # Ensemble
        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, keyword_retriever],
            weights=[0.5, 0.5]
        )
        
        # Reranking
        compressor = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
            top_n=5
        )
        
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble
        )
    
    def setup_tools(self):
        """Setup tool functions"""
        self.tools = {
            "log_search": self.log_search_tool,
            "metrics_query": self.metrics_query_tool,
            "deploy_history": self.deploy_history_tool,
        }
    
    async def log_search_tool(self, query: str, service: str) -> Dict[str, Any]:
        """Search logs"""
        # Implementation from DATAFLOW.md
        result = await elasticsearch_client.search(
            index="logs",
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"message": query}},
                            {"term": {"service": service}}
                        ]
                    }
                },
                "size": 100
            }
        )
        return {
            "logs": result["hits"]["hits"],
            "error_pattern": self.extract_error_pattern(result)
        }
    
    async def metrics_query_tool(self, service: str) -> Dict[str, Any]:
        """Query metrics"""
        # Implementation from DATAFLOW.md
        promql = f'rate(http_requests_total{{service="{service}", status="500"}}[5m])'
        result = await prometheus_client.query(promql)
        return {
            "data_points": result["data"]["result"],
            "spike_detected": self.detect_spike(result)
        }
    
    async def deploy_history_tool(self, service: str) -> Dict[str, Any]:
        """Get deployment history"""
        result = await deploy_api.get_deployments(service=service, limit=10)
        return {
            "deployments": result["deployments"],
            "correlation": self.correlate_with_incident(result)
        }
    
    def setup_chains(self):
        """Setup LangChain chains"""
        
        # Planning chain
        planning_prompt = PromptTemplate(
            input_variables=["incident_description", "service"],
            template="""
            Analyze this incident and decide which tools to use:
            
            Incident: {incident_description}
            Service: {service}
            
            Available tools: log_search, metrics_query, deploy_history
            
            Return a comma-separated list of tools to use.
            Tools:
            """
        )
        self.planning_chain = LLMChain(
            llm=self.llm,
            prompt=planning_prompt,
            output_key="tools_to_use"
        )
        
        # Analysis chain
        analysis_prompt = PromptTemplate(
            input_variables=["incident_description", "evidence", "runbooks"],
            template="""
            Analyze this incident:
            
            INCIDENT: {incident_description}
            
            EVIDENCE:
            {evidence}
            
            RUNBOOKS:
            {runbooks}
            
            Provide:
            1. Root cause hypothesis
            2. Confidence score (0-1)
            3. Reasoning
            
            Response (JSON format):
            """
        )
        self.analysis_chain = LLMChain(
            llm=self.llm,
            prompt=analysis_prompt,
            output_key="analysis"
        )
        
        # Action generation chain
        action_prompt = PromptTemplate(
            input_variables=["hypothesis", "evidence"],
            template="""
            Based on this hypothesis: {hypothesis}
            
            And this evidence: {evidence}
            
            Suggest 3-5 prioritized next actions with:
            - Action description
            - Priority (high/medium/low)
            - Estimated time
            - Rationale
            
            Format as JSON array.
            Actions:
            """
        )
        self.action_chain = LLMChain(
            llm=self.llm,
            prompt=action_prompt,
            output_key="next_actions"
        )
    
    async def analyze_incident(
        self,
        incident_description: str,
        context: Dict[str, str]
    ) -> Dict[str, Any]:
        """Main analysis method"""
        
        service = context.get("service", "unknown")
        
        with get_openai_callback() as cb:
            # Step 1: Planning
            planning_result = self.planning_chain({
                "incident_description": incident_description,
                "service": service
            })
            
            tools_to_use = [
                t.strip() 
                for t in planning_result["tools_to_use"].split(",")
            ]
            
            # Step 2: Execute tools in parallel
            tool_tasks = []
            for tool_name in tools_to_use:
                if tool_name in self.tools:
                    tool_tasks.append(self.tools[tool_name](
                        query=incident_description,
                        service=service
                    ))
            
            tool_results = await asyncio.gather(*tool_tasks)
            
            # Step 3: Retrieve runbooks
            runbooks = self.retriever.get_relevant_documents(incident_description)
            
            # Step 4: Aggregate evidence
            evidence = self.aggregate_evidence(tool_results)
            
            # Step 5: Analysis
            analysis_result = self.analysis_chain({
                "incident_description": incident_description,
                "evidence": self.format_evidence(evidence),
                "runbooks": self.format_runbooks(runbooks)
            })
            
            # Parse analysis (assuming JSON response)
            import json
            analysis = json.loads(analysis_result["analysis"])
            
            # Step 6: Generate actions
            action_result = self.action_chain({
                "hypothesis": analysis["hypothesis"],
                "evidence": self.format_evidence(evidence)
            })
            
            # Parse actions
            next_actions = json.loads(action_result["next_actions"])
            
            # Step 7: Build final response
            final_response = {
                "hypothesis": analysis["hypothesis"],
                "confidence": analysis["confidence"],
                "reasoning": analysis["reasoning"],
                "next_actions": next_actions,
                "citations": self.build_citations(evidence, runbooks),
                "metadata": {
                    "tools_used": tools_to_use,
                    "runbooks_retrieved": len(runbooks),
                    "total_tokens": cb.total_tokens,
                    "total_cost": cb.total_cost
                }
            }
            
            return final_response
    
    def aggregate_evidence(self, tool_results: List[Dict]) -> List[Dict]:
        """Aggregate evidence from tool results"""
        evidence = []
        
        for result in tool_results:
            if "logs" in result:
                evidence.append({
                    "source": "logs",
                    "finding": result.get("error_pattern", ""),
                    "count": len(result["logs"])
                })
            
            if "data_points" in result:
                evidence.append({
                    "source": "metrics",
                    "finding": "Error rate spike detected" if result.get("spike_detected") else "Normal metrics",
                    "details": result.get("spike_details", {})
                })
            
            if "deployments" in result:
                evidence.append({
                    "source": "deployment",
                    "finding": f"Recent deployment: {result['deployments'][0]['version']}",
                    "correlation": result.get("correlation", {})
                })
        
        return evidence
    
    def format_evidence(self, evidence: List[Dict]) -> str:
        """Format evidence for prompt"""
        return "\n".join([
            f"- [{e['source']}] {e['finding']}"
            for e in evidence
        ])
    
    def format_runbooks(self, runbooks: List) -> str:
        """Format runbooks for prompt"""
        return "\n\n".join([
            f"[{i+1}] {doc.metadata['title']}\n{doc.page_content[:300]}..."
            for i, doc in enumerate(runbooks)
        ])
    
    def build_citations(self, evidence: List[Dict], runbooks: List) -> List[Dict]:
        """Build citations from evidence and runbooks"""
        citations = []
        
        for e in evidence:
            citations.append({
                "source": e["source"],
                "reference": f"{e['source']} data",
                "excerpt": e["finding"],
                "timestamp": e.get("timestamp")
            })
        
        for doc in runbooks:
            citations.append({
                "source": "runbook",
                "reference": doc.metadata.get("id", ""),
                "excerpt": doc.page_content[:200],
                "timestamp": None
            })
        
        return citations

# Usage
async def main():
    copilot = OpscopilotChain()
    
    result = await copilot.analyze_incident(
        incident_description="API gateway returning 500 errors since 2pm",
        context={"service": "api-gateway", "environment": "production"}
    )
    
    print(json.dumps(result, indent=2))

# Run
asyncio.run(main())
```

---

## 6. Comparison: LangGraph vs Pure LangChain

| Feature | LangGraph | Pure LangChain |
|---------|-----------|----------------|
| **State Management** | Built-in state object | Manual state passing |
| **Loops** | Native support with conditions | Manual loop implementation |
| **Iteration Limits** | Built-in | Manual counter |
| **Debugging** | Graph visualization | Chain tracing |
| **Complexity** | Higher learning curve | Simpler, more familiar |
| **Flexibility** | Very flexible routing | Linear with conditions |
| **Checkpointing** | Built-in | Manual implementation |
| **Best For** | Complex workflows, loops | Simple sequential flows |

---

## 7. When to Use Each Approach

### Use LangGraph When:
- ✅ You need complex conditional routing
- ✅ Loops and iterations are required
- ✅ State persistence is important
- ✅ You want built-in cycle detection
- ✅ Multiple paths through the workflow

### Use Pure LangChain When:
- ✅ Simple sequential processing
- ✅ No loops required
- ✅ Faster development time
- ✅ Team familiar with chains
- ✅ Simpler debugging needs

---

## 8. Migration Path

### From Pure LangChain to LangGraph

```python
# Before (Pure LangChain)
chain1 = LLMChain(...)
chain2 = LLMChain(...)
overall = SequentialChain(chains=[chain1, chain2])
result = overall(inputs)

# After (LangGraph)
from langgraph.graph import StateGraph

workflow = StateGraph(State)
workflow.add_node("step1", chain1_as_node)
workflow.add_node("step2", chain2_as_node)
workflow.add_edge("step1", "step2")
app = workflow.compile()
result = app.invoke(inputs)
```

---

This document provides a complete pure LangChain implementation as an alternative to LangGraph for the Ops Copilot system!
