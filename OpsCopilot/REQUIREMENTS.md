# Ops Copilot - Requirements Document

## 1. Executive Summary

**Ops Copilot** is an AI-powered incident-response assistant designed to help operations teams quickly diagnose and resolve production incidents. The system answers three critical questions:
- **What's broken?** - Identify the failing component/service
- **Why is it broken?** - Root cause analysis using logs, metrics, and deployment history
- **What should I do next?** - Actionable remediation steps from runbooks

---

## 2. Functional Requirements

### 2.1 Core Capabilities

#### FR-1: Incident Analysis
- **FR-1.1**: Accept natural language incident descriptions or incident IDs
- **FR-1.2**: Perform hybrid retrieval across runbooks and historical incidents
- **FR-1.3**: Execute log searches to identify error patterns
- **FR-1.4**: Query metrics APIs to detect anomalies
- **FR-1.5**: Retrieve recent deployment history to correlate with incidents
- **FR-1.6**: Generate hypotheses about root causes

#### FR-2: Runbook Integration (RAG)
- **FR-2.1**: Vector search across runbook corpus for semantic similarity
- **FR-2.2**: Exact match search for incident IDs and error codes
- **FR-2.3**: Hybrid retrieval combining vector + keyword search
- **FR-2.4**: Rerank top-k results for relevance
- **FR-2.5**: Deduplicate evidence from multiple sources

#### FR-3: Tool Orchestration
The system must integrate with the following tools:

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| **Log Search** | Query application/system logs | Query string, time range, service filter | Log entries with timestamps |
| **Metrics Query** | Retrieve time-series metrics | Metric name, time range, dimensions | Metric data points |
| **Deploy History** | Get recent deployments | Service name, time range | Deployment records |
| **Runbook Vector Search** | Semantic search in runbooks | Query embedding | Top-k runbook sections |
| **Ticketing API** | Create/update incident tickets | Incident details, priority | Ticket ID |

#### FR-4: Structured Output
The system **must** return responses in the following JSON schema:

```json
{
  "hypothesis": "string - Root cause hypothesis",
  "confidence": "float - 0.0 to 1.0",
  "next_actions": [
    {
      "action": "string - Description of action",
      "priority": "string - high|medium|low",
      "estimated_time": "string - e.g., '5 minutes'"
    }
  ],
  "commands": [
    {
      "description": "string - What this command does",
      "command": "string - Executable command",
      "safe_to_run": "boolean - Auto-executable flag"
    }
  ],
  "citations": [
    {
      "source": "string - runbook|logs|metrics|deployment",
      "reference": "string - Document ID or log query",
      "excerpt": "string - Relevant snippet",
      "timestamp": "string - ISO 8601 timestamp"
    }
  ],
  "related_incidents": [
    {
      "incident_id": "string",
      "similarity_score": "float",
      "resolution": "string"
    }
  ]
}
```

---

## 3. Non-Functional Requirements

### 3.1 Performance Requirements

| Metric | Requirement | Rationale |
|--------|-------------|-----------|
| **Throughput** | 10,000 requests/day | ~7 requests/minute average, peak 50 req/min |
| **Latency (p95)** | < 10 seconds | Incident response is time-critical |
| **Latency (p50)** | < 5 seconds | Majority of queries should be fast |
| **Latency (p99)** | < 20 seconds | Acceptable for complex queries |

### 3.2 Reliability Requirements

- **NFR-1**: 99.9% uptime (max 43 minutes downtime/month)
- **NFR-2**: Graceful degradation when external APIs are unavailable
- **NFR-3**: No infinite loops - strict iteration budgets on all agent loops
- **NFR-4**: Timeout protection on all tool calls (max 30s per tool)
- **NFR-5**: Circuit breakers for external service failures

### 3.3 Scalability Requirements

- **NFR-6**: Horizontal scaling to handle 10x traffic spikes
- **NFR-7**: Stateless design for easy replication
- **NFR-8**: Async processing for non-blocking operations
- **NFR-9**: Caching layer for frequently accessed runbooks

### 3.4 Security Requirements

- **NFR-10**: Authentication via API keys or OAuth 2.0
- **NFR-11**: Authorization checks before executing commands
- **NFR-12**: Audit logging of all actions taken
- **NFR-13**: PII redaction in logs and responses
- **NFR-14**: Rate limiting per user/team (100 req/hour)

### 3.5 Observability Requirements

- **NFR-15**: Distributed tracing for all requests
- **NFR-16**: Metrics on latency, error rates, tool usage
- **NFR-17**: Structured logging with correlation IDs
- **NFR-18**: Real-time dashboards for system health

---

## 4. Iteration Budget Constraints

To prevent infinite loops and ensure predictable performance:

| Component | Max Iterations | Timeout |
|-----------|----------------|---------|
| **Agent reasoning loop** | 5 iterations | 30s total |
| **Tool retry logic** | 3 attempts | 10s per attempt |
| **Runbook retrieval** | 1 pass (no re-ranking loops) | 5s |
| **Log search pagination** | Max 3 pages | 15s total |
| **Metrics query** | Single query (no refinement) | 5s |

---

## 5. Data Requirements

### 5.1 Runbook Corpus
- **Format**: Markdown or structured JSON
- **Size**: ~1,000 runbooks (avg 2KB each)
- **Update frequency**: Daily sync
- **Metadata**: Tags, service mappings, last updated timestamp

### 5.2 Historical Incidents
- **Retention**: 90 days of incident data
- **Fields**: Incident ID, description, resolution, duration, affected services
- **Index**: Vector embeddings + keyword index

### 5.3 Logs
- **Sources**: Application logs, system logs, audit logs
- **Retention**: 30 days
- **Volume**: ~1TB/day
- **Access**: Via log aggregation API (e.g., Elasticsearch, Splunk)

### 5.4 Metrics
- **Types**: CPU, memory, request rate, error rate, latency
- **Granularity**: 1-minute intervals
- **Retention**: 30 days high-res, 1 year downsampled
- **Access**: Via metrics API (e.g., Prometheus, Datadog)

---

## 6. Evaluation Criteria

### 6.1 LangSmith Datasets
Create evaluation datasets with:
- **50 historical incidents** with known root causes
- **20 synthetic incidents** for edge cases
- **10 adversarial cases** (misleading symptoms)

### 6.2 Evaluators

| Evaluator | Metric | Target |
|-----------|--------|--------|
| **Action Correctness** | % of correct next actions | > 85% |
| **Groundedness** | % of claims with citations | > 95% |
| **Time-to-Mitigation Proxy** | Avg steps to resolution | < 3 steps |
| **Hallucination Rate** | % of unsupported claims | < 5% |
| **Tool Usage Efficiency** | Avg tools called per query | < 4 tools |

### 6.3 Success Metrics
- **Resolution Rate**: % of incidents resolved without human escalation (target: 60%)
- **MTTR Reduction**: Mean time to resolution improvement (target: 30% faster)
- **User Satisfaction**: NPS score from on-call engineers (target: > 50)

---

## 7. Constraints and Assumptions

### 7.1 Constraints
- **C-1**: Must use existing log/metrics infrastructure (no new data pipelines)
- **C-2**: Cannot execute commands automatically (requires human approval)
- **C-3**: Budget: $500/month for LLM API costs
- **C-4**: Team size: 2 engineers for initial build

### 7.2 Assumptions
- **A-1**: Runbooks are well-maintained and up-to-date
- **A-2**: Log/metrics APIs have < 2s response time
- **A-3**: Incident descriptions are in English
- **A-4**: Users have basic command-line knowledge

---

## 8. Out of Scope (v1)

- ❌ Automatic command execution (safety-critical)
- ❌ Multi-language support (English only)
- ❌ Predictive incident detection (reactive only)
- ❌ Integration with ChatOps platforms (Slack/Teams)
- ❌ Custom runbook authoring UI
- ❌ Mobile application

---

## 9. Future Enhancements (v2+)

- 🔮 Proactive anomaly detection
- 🔮 Auto-remediation for known issues
- 🔮 Slack/Teams bot interface
- 🔮 Runbook generation from resolved incidents
- 🔮 Multi-cloud support (AWS, GCP, Azure)
- 🔮 Custom alerting rules

---

## 10. Acceptance Criteria

The system is considered complete when:

1. ✅ All functional requirements (FR-1 to FR-4) are implemented
2. ✅ Performance targets (10k req/day, p95 < 10s) are met in load testing
3. ✅ Evaluation metrics exceed targets (85% action correctness, 95% groundedness)
4. ✅ No infinite loops observed in 1000-query stress test
5. ✅ Documentation complete (API docs, runbook authoring guide, deployment guide)
6. ✅ Successfully resolves 10 real production incidents in pilot phase
