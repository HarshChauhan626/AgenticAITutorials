# Ops Copilot - API Specification

## Overview

This document provides the complete API specification for the Ops Copilot incident-response assistant.

**Base URL:** `https://api.opscopilot.company.com/v1`

**Authentication:** Bearer token (JWT) or API Key

---

## Authentication

### API Key Authentication

Include API key in request header:

```bash
curl -H "X-API-Key: your_api_key_here" \
  https://api.opscopilot.company.com/v1/analyze
```

### JWT Authentication

```bash
curl -H "Authorization: Bearer your_jwt_token" \
  https://api.opscopilot.company.com/v1/analyze
```

---

## Endpoints

### 1. Analyze Incident

**Endpoint:** `POST /api/v1/analyze`

**Description:** Analyze an incident and get AI-powered recommendations.

**Rate Limit:** 100 requests/hour per user

**Request Body:**

```json
{
  "incident_description": "string (required, 10-5000 chars)",
  "incident_id": "string (optional, format: INC-\\d+)",
  "context": {
    "service": "string (optional)",
    "environment": "string (optional, e.g., production, staging)",
    "region": "string (optional)",
    "severity": "string (optional, critical|high|medium|low)"
  },
  "options": {
    "include_commands": "boolean (default: true)",
    "max_citations": "integer (default: 5, max: 10)",
    "include_related_incidents": "boolean (default: true)"
  }
}
```

**Example Request:**

```bash
curl -X POST https://api.opscopilot.company.com/v1/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "incident_description": "API gateway returning 500 errors since 2pm, affecting checkout flow",
    "context": {
      "service": "api-gateway",
      "environment": "production",
      "severity": "critical"
    },
    "options": {
      "include_commands": true,
      "max_citations": 5
    }
  }'
```

**Response (200 OK):**

```json
{
  "request_id": "req_abc123xyz",
  "timestamp": "2025-12-20T00:57:50Z",
  "latency_ms": 4523,
  "result": {
    "hypothesis": "The v2.3.5 deployment introduced a database driver incompatibility...",
    "confidence": 0.85,
    "next_actions": [
      {
        "action": "Rollback to v2.3.4 immediately to restore service",
        "priority": "high",
        "estimated_time": "5 minutes",
        "rationale": "Fastest path to mitigation given strong deployment correlation"
      }
    ],
    "commands": [
      {
        "description": "Rollback API gateway to previous version",
        "command": "kubectl rollout undo deployment/api-gateway -n production",
        "safe_to_run": false
      }
    ],
    "citations": [
      {
        "source": "logs",
        "reference": "Elasticsearch query: status:500 AND service:api-gateway",
        "excerpt": "Database connection timeout (78% of 247 errors)",
        "timestamp": "2025-12-20T00:45:00Z"
      }
    ],
    "related_incidents": [
      {
        "incident_id": "INC-11234",
        "similarity_score": 0.78,
        "resolution": "Rolled back deployment, increased connection pool"
      }
    ]
  },
  "metadata": {
    "tools_used": ["log_search", "metrics_query", "deploy_history", "runbook_search"],
    "runbooks_retrieved": 5,
    "cache_hit": false
  }
}
```

**Error Responses:**

```json
// 400 Bad Request
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "incident_description must be between 10 and 5000 characters",
    "details": {
      "field": "incident_description",
      "provided_length": 5
    }
  }
}

// 401 Unauthorized
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or missing API key"
  }
}

// 429 Too Many Requests
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit of 100 requests/hour exceeded",
    "retry_after": 1800
  }
}

// 504 Gateway Timeout
{
  "error": {
    "code": "TIMEOUT",
    "message": "Request exceeded 30 second timeout",
    "partial_results": {
      "tools_completed": ["log_search", "runbook_search"],
      "tools_failed": ["metrics_query"]
    }
  }
}
```

---

### 2. Get Incident Details

**Endpoint:** `GET /api/v1/incidents/{incident_id}`

**Description:** Retrieve details of a previously analyzed incident.

**Path Parameters:**
- `incident_id` (string, required): Incident ID (e.g., INC-12345)

**Example Request:**

```bash
curl -X GET https://api.opscopilot.company.com/v1/incidents/INC-12345 \
  -H "X-API-Key: your_api_key"
```

**Response (200 OK):**

```json
{
  "incident_id": "INC-12345",
  "description": "API gateway returning 500 errors...",
  "status": "resolved",
  "severity": "critical",
  "created_at": "2025-12-20T00:35:00Z",
  "resolved_at": "2025-12-20T01:15:00Z",
  "duration_seconds": 2400,
  "hypothesis": "Database driver incompatibility...",
  "resolution": "Rolled back to v2.3.4",
  "affected_services": ["api-gateway"],
  "analysis": {
    "next_actions": [...],
    "commands": [...],
    "citations": [...]
  }
}
```

---

### 3. Search Runbooks

**Endpoint:** `POST /api/v1/runbooks/search`

**Description:** Search runbook corpus directly.

**Request Body:**

```json
{
  "query": "string (required)",
  "service": "string (optional)",
  "top_k": "integer (default: 5, max: 20)"
}
```

**Example Request:**

```bash
curl -X POST https://api.opscopilot.company.com/v1/runbooks/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "query": "database connection timeout",
    "service": "api-gateway",
    "top_k": 5
  }'
```

**Response (200 OK):**

```json
{
  "results": [
    {
      "id": "rb_001",
      "title": "Debugging Database Connection Timeouts",
      "score": 0.94,
      "service": "api-gateway",
      "excerpt": "When experiencing database connection timeouts...",
      "url": "https://runbooks.company.com/rb_001",
      "last_updated": "2025-11-15T10:30:00Z"
    }
  ],
  "total_results": 5,
  "query_time_ms": 234
}
```

---

### 4. List Recent Incidents

**Endpoint:** `GET /api/v1/incidents`

**Description:** List recent incidents with optional filtering.

**Query Parameters:**
- `service` (string, optional): Filter by service name
- `status` (string, optional): Filter by status (open, investigating, resolved)
- `severity` (string, optional): Filter by severity
- `limit` (integer, default: 20, max: 100)
- `offset` (integer, default: 0)

**Example Request:**

```bash
curl -X GET "https://api.opscopilot.company.com/v1/incidents?service=api-gateway&status=resolved&limit=10" \
  -H "X-API-Key: your_api_key"
```

**Response (200 OK):**

```json
{
  "incidents": [
    {
      "incident_id": "INC-12345",
      "description": "API gateway returning 500 errors...",
      "status": "resolved",
      "severity": "critical",
      "created_at": "2025-12-20T00:35:00Z",
      "resolved_at": "2025-12-20T01:15:00Z"
    }
  ],
  "total": 47,
  "limit": 10,
  "offset": 0
}
```

---

### 5. Create Incident Ticket

**Endpoint:** `POST /api/v1/tickets`

**Description:** Create a ticket in the integrated ticketing system.

**Request Body:**

```json
{
  "incident_id": "string (optional)",
  "title": "string (required)",
  "description": "string (required)",
  "priority": "string (required, critical|high|medium|low)",
  "assignee": "string (optional)"
}
```

**Response (201 Created):**

```json
{
  "ticket_id": "TICKET-789",
  "ticket_url": "https://jira.company.com/browse/TICKET-789",
  "created_at": "2025-12-20T00:57:50Z"
}
```

---

### 6. Health Check

**Endpoint:** `GET /api/v1/health`

**Description:** Check API health status.

**Response (200 OK):**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-12-20T00:57:50Z",
  "dependencies": {
    "database": "healthy",
    "redis": "healthy",
    "vector_db": "healthy",
    "elasticsearch": "healthy",
    "llm_api": "healthy"
  }
}
```

---

## Rate Limiting

**Limits:**
- 100 requests/hour per user (analyze endpoint)
- 500 requests/hour per user (other endpoints)

**Headers:**

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1703034470
```

---

## Webhooks (Optional)

**Endpoint:** `POST /api/v1/webhooks`

**Description:** Register a webhook for incident notifications.

**Request Body:**

```json
{
  "url": "https://your-service.com/webhook",
  "events": ["incident.created", "incident.resolved"],
  "secret": "your_webhook_secret"
}
```

**Webhook Payload:**

```json
{
  "event": "incident.resolved",
  "timestamp": "2025-12-20T01:15:00Z",
  "data": {
    "incident_id": "INC-12345",
    "resolution": "Rolled back to v2.3.4",
    "duration_seconds": 2400
  }
}
```

---

## SDKs

### Python SDK

```python
from opscopilot import OpscopilotClient

client = OpscopilotClient(api_key="your_api_key")

# Analyze incident
result = client.analyze(
    incident_description="API returning 500 errors",
    context={"service": "api-gateway", "environment": "production"}
)

print(result.hypothesis)
print(result.next_actions)
```

### Node.js SDK

```javascript
const { OpscopilotClient } = require('@company/opscopilot');

const client = new OpscopilotClient({ apiKey: 'your_api_key' });

// Analyze incident
const result = await client.analyze({
  incidentDescription: 'API returning 500 errors',
  context: { service: 'api-gateway', environment: 'production' }
});

console.log(result.hypothesis);
```

---

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Malformed request body |
| `UNAUTHORIZED` | 401 | Invalid or missing credentials |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |
| `SERVICE_UNAVAILABLE` | 503 | Temporary service disruption |
| `TIMEOUT` | 504 | Request timeout |

---

## Changelog

### v1.0.0 (2025-12-20)
- Initial release
- Core incident analysis endpoint
- Runbook search
- Incident history
