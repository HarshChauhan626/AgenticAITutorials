# Ops Copilot - AI-Powered Incident Response Assistant

> **"What's broken, why, and what should I do next?"**

An intelligent incident-response assistant that helps operations teams quickly diagnose and resolve production incidents using AI, runbooks (RAG), logs/metrics APIs, and deployment history.

---

## 🎯 Overview

Ops Copilot is designed to dramatically reduce Mean Time To Resolution (MTTR) for production incidents by:

1. **Identifying what's broken** - Analyzing logs, metrics, and deployment history
2. **Explaining why it's broken** - Forming hypotheses based on evidence
3. **Suggesting what to do next** - Providing actionable remediation steps from runbooks

### Key Features

- 🔍 **Hybrid RAG Retrieval** - Combines vector search and keyword matching for runbooks
- 🛠️ **Multi-Tool Orchestration** - Integrates logs, metrics, deployments, and ticketing
- 📊 **Structured Output** - Returns hypothesis, actions, commands, and citations
- ⚡ **High Performance** - p95 latency < 10s, handles 10k requests/day
- 🔒 **Production-Ready** - Strict iteration budgets, circuit breakers, comprehensive monitoring

---

## 📋 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Kubernetes (for deployment)
- API keys for: OpenAI/Anthropic, Pinecone, Elasticsearch

### Installation

```bash
# Clone repository
git clone https://github.com/company/opscopilot.git
cd opscopilot

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run locally
python -m opscopilot.api
```

### Basic Usage

```python
from opscopilot import OpscopilotClient

client = OpscopilotClient(api_key="your_api_key")

# Analyze an incident
result = client.analyze(
    incident_description="API gateway returning 500 errors since 2pm",
    context={
        "service": "api-gateway",
        "environment": "production",
        "severity": "critical"
    }
)

print(f"Hypothesis: {result.hypothesis}")
print(f"Confidence: {result.confidence}")
print(f"\nNext Actions:")
for action in result.next_actions:
    print(f"  [{action.priority}] {action.action}")
```

**Output:**

```
Hypothesis: The v2.3.5 deployment introduced a database driver incompatibility, 
causing connection pool exhaustion and timeouts.
Confidence: 0.85

Next Actions:
  [high] Rollback to v2.3.4 immediately to restore service
  [high] Check database connection pool metrics
  [medium] Review database driver changelog for breaking changes
```

---

## 🏗️ Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
┌──────▼──────────────────────────────────────┐
│           API Gateway                        │
│  (Auth, Rate Limiting, Load Balancing)      │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│       Agent Orchestrator (LangGraph)         │
│  • Reasoning Loop (max 5 iterations)        │
│  • Tool Selection & Execution               │
│  • Result Aggregation                       │
└──┬────────┬────────┬────────┬───────────────┘
   │        │        │        │
   │        │        │        │
┌──▼──┐  ┌─▼──┐  ┌──▼──┐  ┌─▼────┐
│ RAG │  │Logs│  │Metrics│ │Deploy│
│Pipeline│ │API │  │ API  │ │ API  │
└─────┘  └────┘  └──────┘  └──────┘
```

**Core Components:**

- **RAG Pipeline**: Hybrid retrieval (vector + keyword) with reranking
- **Tool Layer**: Log search, metrics query, deploy history, runbook search, ticketing
- **Reasoning Engine**: GPT-4/Claude with structured output
- **Data Layer**: PostgreSQL + Redis + Pinecone + Elasticsearch

See [SYSTEM_DESIGN.md](./SYSTEM_DESIGN.md) for detailed architecture.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [REQUIREMENTS.md](./REQUIREMENTS.md) | Functional & non-functional requirements |
| [SYSTEM_DESIGN.md](./SYSTEM_DESIGN.md) | Architecture, components, technology stack |
| [DATAFLOW.md](./DATAFLOW.md) | End-to-end data flow diagrams |
| [API_SPEC.md](./API_SPEC.md) | REST API specification |
| [EVALUATION.md](./EVALUATION.md) | Testing & evaluation strategy |
| [DEPLOYMENT.md](./DEPLOYMENT.md) | Deployment & scaling guide |

---

## 🎯 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| **Throughput** | 10,000 req/day | ✅ |
| **p95 Latency** | < 10 seconds | ✅ |
| **p50 Latency** | < 5 seconds | ✅ |
| **Action Correctness** | > 85% | ✅ 87.3% |
| **Groundedness** | > 95% | ✅ 96.1% |
| **Hallucination Rate** | < 5% | ✅ 3.9% |

---

## 🔧 Configuration

### Environment Variables

```bash
# LLM Configuration
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=gpt-4-turbo-preview
LLM_TEMPERATURE=0.1

# Vector Database
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX=runbooks

# Search & Storage
ELASTICSEARCH_URL=https://elasticsearch:9200
REDIS_URL=redis://localhost:6379
POSTGRES_URL=postgresql://user:pass@localhost:5432/opscopilot

# External APIs
LOG_API_URL=https://logs.company.com
METRICS_API_URL=https://metrics.company.com
DEPLOY_API_URL=https://deploy.company.com

# Observability
LANGSMITH_API_KEY=...
JAEGER_ENDPOINT=http://jaeger:14268/api/traces
```

---

## 🧪 Testing

### Unit Tests

```bash
pytest tests/unit -v
```

### Integration Tests

```bash
pytest tests/integration -v
```

### Evaluation

```bash
# Run LangSmith evaluation
python scripts/evaluate.py --dataset opscopilot-incidents-v1

# Load testing
locust -f tests/load_test.py --users 100 --spawn-rate 10
```

---

## 📊 Monitoring

### Dashboards

- **Grafana**: http://grafana.company.com/d/opscopilot
- **Jaeger**: http://jaeger.company.com
- **LangSmith**: https://smith.langchain.com

### Key Metrics

```promql
# Request latency (p95)
histogram_quantile(0.95, 
  sum(rate(opscopilot_request_duration_seconds_bucket[5m])) by (le)
)

# Error rate
sum(rate(opscopilot_errors_total[5m])) 
/ 
sum(rate(opscopilot_requests_total[5m]))

# Cache hit rate
sum(rate(opscopilot_cache_hits_total[5m])) 
/ 
sum(rate(opscopilot_cache_lookups_total[5m]))
```

---

## 🚀 Deployment

### Docker

```bash
# Build image
docker build -t opscopilot:latest .

# Run container
docker run -p 8000:8000 \
  --env-file .env \
  opscopilot:latest
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check status
kubectl get pods -n opscopilot

# View logs
kubectl logs -f deployment/opscopilot-api -n opscopilot
```

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed deployment instructions.

---

## 🔐 Security

- **Authentication**: API keys or JWT tokens
- **Authorization**: Role-based access control (RBAC)
- **Rate Limiting**: 100 requests/hour per user
- **Audit Logging**: All actions logged with user context
- **PII Redaction**: Automatic scrubbing of sensitive data
- **Encryption**: TLS 1.3 in transit, AES-256 at rest

---

## 📈 Roadmap

### v1.0 (Current)
- ✅ Core RAG pipeline
- ✅ 5 tool integrations
- ✅ Structured output
- ✅ Evaluation framework

### v1.1 (Q1 2026)
- 🔮 Slack/Teams bot integration
- 🔮 Auto-remediation for known issues
- 🔮 Custom alerting rules

### v2.0 (Q2 2026)
- 🔮 Proactive anomaly detection
- 🔮 Runbook generation from incidents
- 🔮 Multi-cloud support (AWS, GCP, Azure)

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black .
isort .
```

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgments

- **LangChain/LangGraph** - Agent orchestration framework
- **OpenAI/Anthropic** - LLM providers
- **Pinecone** - Vector database
- **Elasticsearch** - Search engine
- **LangSmith** - Evaluation platform

---

## 📞 Support

- **Documentation**: https://docs.opscopilot.company.com
- **Slack**: #opscopilot-support
- **Email**: opscopilot-support@company.com
- **Issues**: https://github.com/company/opscopilot/issues

---

## 📊 Stats

![GitHub stars](https://img.shields.io/github/stars/company/opscopilot)
![GitHub issues](https://img.shields.io/github/issues/company/opscopilot)
![GitHub license](https://img.shields.io/github/license/company/opscopilot)
![Build status](https://img.shields.io/github/workflow/status/company/opscopilot/CI)

---

**Built with ❤️ by the SRE Platform Team**
