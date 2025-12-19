# Ops Copilot - Deployment Guide

## 1. Overview

This guide covers deploying Ops Copilot to production, including infrastructure setup, configuration, scaling, and monitoring.

---

## 2. Infrastructure Requirements

### 2.1 Compute Resources

| Environment | Component | Specification | Count |
|-------------|-----------|---------------|-------|
| **Production** | API Pods | 2 vCPU, 8GB RAM | 3-10 (auto-scale) |
| **Production** | Worker Pods | 2 vCPU, 4GB RAM | 2-5 (auto-scale) |
| **Staging** | API Pods | 1 vCPU, 4GB RAM | 2 |
| **Development** | API Pods | 1 vCPU, 2GB RAM | 1 |

### 2.2 Managed Services

| Service | Provider | Specification |
|---------|----------|---------------|
| **PostgreSQL** | AWS RDS | db.t3.medium (2 vCPU, 4GB) |
| **Redis** | AWS ElastiCache | cache.t3.medium (2 vCPU, 3.2GB) |
| **Pinecone** | Pinecone Cloud | 1M vectors, Standard plan |
| **Elasticsearch** | AWS OpenSearch | 3-node cluster (t3.small) |

### 2.3 Network Requirements

- **Ingress**: HTTPS (443) from internet
- **Egress**: HTTPS (443) to external APIs
- **Internal**: PostgreSQL (5432), Redis (6379), Elasticsearch (9200)

---

## 3. Pre-Deployment Checklist

### 3.1 Prerequisites

- [ ] Kubernetes cluster (v1.24+)
- [ ] kubectl configured
- [ ] Helm 3 installed
- [ ] Docker registry access
- [ ] API keys for all external services
- [ ] SSL/TLS certificates

### 3.2 External Service Setup

#### Pinecone Vector Database

```bash
# Create Pinecone index
curl -X POST "https://controller.us-west1-gcp.pinecone.io/databases" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "runbooks",
    "dimension": 1536,
    "metric": "cosine",
    "pods": 1,
    "pod_type": "p1.x1"
  }'
```

#### PostgreSQL Database

```sql
-- Create database
CREATE DATABASE opscopilot;

-- Create user
CREATE USER opscopilot_user WITH PASSWORD 'secure_password';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE opscopilot TO opscopilot_user;

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
```

#### Elasticsearch Index

```bash
# Create runbooks index
curl -X PUT "https://elasticsearch:9200/runbooks" \
  -H "Content-Type: application/json" \
  -d '{
    "settings": {
      "number_of_shards": 3,
      "number_of_replicas": 2,
      "analysis": {
        "analyzer": {
          "custom_analyzer": {
            "type": "custom",
            "tokenizer": "standard",
            "filter": ["lowercase", "stop", "snowball"]
          }
        }
      }
    },
    "mappings": {
      "properties": {
        "title": {"type": "text", "analyzer": "custom_analyzer"},
        "content": {"type": "text", "analyzer": "custom_analyzer"},
        "service": {"type": "keyword"},
        "tags": {"type": "keyword"},
        "last_updated": {"type": "date"}
      }
    }
  }'
```

---

## 4. Docker Build & Push

### 4.1 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 opscopilot && \
    chown -R opscopilot:opscopilot /app
USER opscopilot

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "opscopilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4.2 Build & Push

```bash
# Build image
docker build -t opscopilot:v1.0.0 .

# Tag for registry
docker tag opscopilot:v1.0.0 registry.company.com/opscopilot:v1.0.0
docker tag opscopilot:v1.0.0 registry.company.com/opscopilot:latest

# Push to registry
docker push registry.company.com/opscopilot:v1.0.0
docker push registry.company.com/opscopilot:latest
```

---

## 5. Kubernetes Deployment

### 5.1 Namespace

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: opscopilot
  labels:
    name: opscopilot
```

### 5.2 Secrets

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: opscopilot-secrets
  namespace: opscopilot
type: Opaque
stringData:
  OPENAI_API_KEY: "sk-..."
  PINECONE_API_KEY: "..."
  POSTGRES_URL: "postgresql://user:pass@postgres:5432/opscopilot"
  REDIS_URL: "redis://redis:6379"
  ELASTICSEARCH_URL: "https://elasticsearch:9200"
```

### 5.3 ConfigMap

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: opscopilot-config
  namespace: opscopilot
data:
  LLM_MODEL: "gpt-4-turbo-preview"
  LLM_TEMPERATURE: "0.1"
  MAX_REASONING_LOOPS: "5"
  MAX_TOOL_RETRIES: "3"
  CACHE_TTL: "3600"
  LOG_LEVEL: "INFO"
```

### 5.4 Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: opscopilot-api
  namespace: opscopilot
  labels:
    app: opscopilot-api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: opscopilot-api
  template:
    metadata:
      labels:
        app: opscopilot-api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: api
        image: registry.company.com/opscopilot:v1.0.0
        ports:
        - containerPort: 8000
          name: http
        envFrom:
        - configMapRef:
            name: opscopilot-config
        - secretRef:
            name: opscopilot-secrets
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - opscopilot-api
              topologyKey: kubernetes.io/hostname
```

### 5.5 Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: opscopilot-api
  namespace: opscopilot
  labels:
    app: opscopilot-api
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: opscopilot-api
```

### 5.6 Ingress

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: opscopilot-ingress
  namespace: opscopilot
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.opscopilot.company.com
    secretName: opscopilot-tls
  rules:
  - host: api.opscopilot.company.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: opscopilot-api
            port:
              number: 80
```

### 5.7 HorizontalPodAutoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: opscopilot-api-hpa
  namespace: opscopilot
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: opscopilot-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 25
        periodSeconds: 60
```

---

## 6. Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

# Verify deployment
kubectl get pods -n opscopilot
kubectl get svc -n opscopilot
kubectl get ingress -n opscopilot

# Check logs
kubectl logs -f deployment/opscopilot-api -n opscopilot

# Check health
curl https://api.opscopilot.company.com/health
```

---

## 7. Database Migrations

```bash
# Run migrations
kubectl exec -it deployment/opscopilot-api -n opscopilot -- \
  python -m alembic upgrade head

# Verify schema
kubectl exec -it deployment/opscopilot-api -n opscopilot -- \
  python -m alembic current
```

---

## 8. Data Ingestion

### 8.1 Load Runbooks

```bash
# Upload runbooks to vector DB and Elasticsearch
kubectl exec -it deployment/opscopilot-api -n opscopilot -- \
  python scripts/ingest_runbooks.py \
    --source /data/runbooks \
    --batch-size 100
```

### 8.2 Backfill Historical Incidents

```bash
# Import historical incidents for similarity search
kubectl exec -it deployment/opscopilot-api -n opscopilot -- \
  python scripts/backfill_incidents.py \
    --source /data/incidents.json \
    --days 90
```

---

## 9. Monitoring Setup

### 9.1 Prometheus ServiceMonitor

```yaml
# k8s/servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: opscopilot-api
  namespace: opscopilot
spec:
  selector:
    matchLabels:
      app: opscopilot-api
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

### 9.2 Grafana Dashboard

```bash
# Import dashboard
kubectl create configmap opscopilot-dashboard \
  --from-file=dashboards/opscopilot.json \
  -n monitoring
```

---

## 10. Backup & Disaster Recovery

### 10.1 Database Backups

```bash
# Automated daily backups (RDS)
aws rds create-db-snapshot \
  --db-instance-identifier opscopilot-prod \
  --db-snapshot-identifier opscopilot-$(date +%Y%m%d)

# Retention: 30 days
```

### 10.2 Vector DB Backups

```bash
# Backup Pinecone index
python scripts/backup_pinecone.py \
  --index runbooks \
  --output s3://backups/pinecone/$(date +%Y%m%d)
```

### 10.3 Disaster Recovery Plan

1. **RTO (Recovery Time Objective)**: 1 hour
2. **RPO (Recovery Point Objective)**: 24 hours

**Recovery Steps:**

```bash
# 1. Restore database from snapshot
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier opscopilot-prod-restored \
  --db-snapshot-identifier opscopilot-20251220

# 2. Restore Pinecone index
python scripts/restore_pinecone.py \
  --backup s3://backups/pinecone/20251220 \
  --index runbooks

# 3. Redeploy application
kubectl apply -f k8s/
```

---

## 11. Rollback Procedure

```bash
# Rollback to previous version
kubectl rollout undo deployment/opscopilot-api -n opscopilot

# Rollback to specific revision
kubectl rollout undo deployment/opscopilot-api -n opscopilot --to-revision=3

# Check rollout status
kubectl rollout status deployment/opscopilot-api -n opscopilot

# View rollout history
kubectl rollout history deployment/opscopilot-api -n opscopilot
```

---

## 12. Scaling Strategies

### 12.1 Vertical Scaling

```bash
# Increase resource limits
kubectl set resources deployment/opscopilot-api -n opscopilot \
  --limits=cpu=4000m,memory=8Gi \
  --requests=cpu=1000m,memory=2Gi
```

### 12.2 Horizontal Scaling

```bash
# Manual scaling
kubectl scale deployment/opscopilot-api -n opscopilot --replicas=5

# Auto-scaling (via HPA)
# Already configured in k8s/hpa.yaml
```

---

## 13. Security Hardening

### 13.1 Network Policies

```yaml
# k8s/networkpolicy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: opscopilot-api-netpol
  namespace: opscopilot
spec:
  podSelector:
    matchLabels:
      app: opscopilot-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 443  # HTTPS
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
```

### 13.2 Pod Security Policy

```yaml
# k8s/psp.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: opscopilot-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
  - ALL
  runAsUser:
    rule: MustRunAsNonRoot
  seLinux:
    rule: RunAsAny
  fsGroup:
    rule: RunAsAny
  volumes:
  - configMap
  - secret
  - emptyDir
```

---

## 14. Cost Optimization

### 14.1 Use Spot Instances (Workers)

```yaml
# k8s/worker-deployment.yaml
spec:
  template:
    spec:
      nodeSelector:
        node.kubernetes.io/instance-type: spot
      tolerations:
      - key: "spot"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
```

### 14.2 Cache Optimization

- Increase cache TTL for frequently accessed runbooks
- Implement query result caching (1 hour TTL)
- Use CDN for static runbook content

**Expected Savings:** ~$200/month

---

## 15. Post-Deployment Validation

```bash
# 1. Health check
curl https://api.opscopilot.company.com/health

# 2. Test analysis endpoint
curl -X POST https://api.opscopilot.company.com/v1/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test_key" \
  -d '{
    "incident_description": "Test incident",
    "context": {"service": "test"}
  }'

# 3. Check metrics
curl https://api.opscopilot.company.com/metrics

# 4. Verify logs
kubectl logs -f deployment/opscopilot-api -n opscopilot

# 5. Run smoke tests
pytest tests/smoke -v
```

---

## 16. Troubleshooting

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| **High latency** | p95 > 10s | Check external API timeouts, increase cache TTL |
| **OOM errors** | Pods restarting | Increase memory limits, optimize embedding cache |
| **Rate limiting** | 429 errors | Increase rate limits or scale horizontally |
| **LLM timeouts** | 504 errors | Reduce max_tokens, implement retry logic |

### Debug Commands

```bash
# Check pod status
kubectl describe pod <pod-name> -n opscopilot

# View recent events
kubectl get events -n opscopilot --sort-by='.lastTimestamp'

# Check resource usage
kubectl top pods -n opscopilot

# Exec into pod
kubectl exec -it <pod-name> -n opscopilot -- /bin/bash
```

---

This deployment guide provides a comprehensive roadmap for deploying Ops Copilot to production with high availability, security, and observability.
