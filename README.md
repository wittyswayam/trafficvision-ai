# TrafficVision-AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange) ![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green) ![Docker](https://img.shields.io/badge/Docker-Compose-2496ED) ![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-326CE5) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Enterprise-grade road traffic sign and light recognition platform — built on CNN transfer learning with a production FastAPI serving layer, multi-GPU training, MLflow experiment tracking, DVC-versioned data pipelines, and full-stack observability.**

---

## Project Overview

TrafficVision-AI is a complete computer vision system for detecting and localising road traffic signs and signals in still images. The platform covers the full ML lifecycle: from raw image ingestion through a validated ETL pipeline, two-phase CNN training with ImageNet-pretrained backbones, Bayesian hyperparameter optimisation, versioned model registry with promotion gates, and a production REST API capable of serving single-image and batch inference requests with Redis caching and Prometheus metrics.

**Real-world problems it addresses:**

- Automated traffic sign inventory and compliance verification for road authorities
- Dataset generation for autonomous vehicle perception model validation
- Infrastructure anomaly detection (missing or obscured signage)
- High-throughput offline analysis of dashcam or traffic camera footage

**Target users:** Computer vision engineers, autonomous systems teams, smart city operations, and ML practitioners who need a fully wired training-to-serving pipeline rather than a bare model notebook.

**Why this architecture:** The system is intentionally split into independent, replaceable layers — data pipeline, model, registry, API, monitoring — each with clean interfaces. Transfer learning from ImageNet dramatically reduces the data volume needed to reach production-grade bounding box precision. The two-phase training strategy (frozen backbone → selective fine-tuning) is a deliberate engineering choice that keeps early training stable while allowing end-to-end gradient flow in the second stage. The FastAPI + Redis + Prometheus stack was chosen for its async-native characteristics and out-of-the-box compatibility with Kubernetes liveness/readiness probes and Grafana dashboards.

---

## Core Features

### Multi-Backbone CNN Architecture

The model factory (`src/ml/model.py`) supports three ImageNet-pretrained backbones selectable at training or inference time:

| Backbone | Parameters | Best Use Case |
|---|---|---|
| `resnet50` | ~25M | Baseline, well-studied, easy to debug |
| `efficientnetb3` | ~12M | Accuracy/efficiency tradeoff, default config |
| `mobilenetv3` | ~5.4M | Edge or embedded deployment |

All backbones feed into a shared detection head: `GlobalAveragePooling2D → BatchNormalization → Dense(1024, relu) → Dropout → Dense(512, relu) → Dense(4, sigmoid)`. The sigmoid output produces normalised bounding box coordinates `[x_min, y_min, x_max, y_max]` in the range `[0, 1]`.

### Combined IoU + MSE Loss

Standard MSE penalises coordinate errors uniformly regardless of box size. TrafficVision-AI implements a `combined_loss` that blends MSE with a differentiable IoU term (`α × MSE + (1-α) × IoU_loss`, default α=0.7). This incentivises the model to maximise spatial overlap rather than minimise absolute coordinate offsets, which translates directly to better mean IoU at evaluation time.

### Two-Phase Training with Keras Callbacks

`TrainingOrchestrator` (`src/ml/trainer.py`) runs two sequential phases. Phase 1 trains only the detection head (frozen backbone) for up to 30 epochs with `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`, `TensorBoard`, and `CSVLogger`. Phase 2 unfreezes the top 20 backbone layers via `model.fine_tune(unfreeze_from=-20)`, re-compiles at one-tenth the learning rate, and halves the batch size for gradient stability. Post-training, the orchestrator computes mean IoU on the held-out test split, persists `metrics.json`, and optionally logs the full run to MLflow.

### Weighted Ensemble Inference

`EnsembleDetector` (`src/ml/model.py`) instantiates multiple backbone models, runs forward passes independently, and returns a weighted average of predictions. It also exposes per-coordinate prediction variance as an uncertainty estimate — useful for human-in-the-loop review queues where the system should flag low-confidence detections.

### Bayesian Hyperparameter Optimisation

`src/ml/hpo.py` drives an Optuna TPE-sampler search over backbone, learning rate (log-uniform 1e-5 to 1e-3), batch size, dense units, dropout rate, loss alpha, and number of layers to unfreeze. Each trial runs for at most 15 epochs; Optuna's MedianPruner terminates underperformers early. The winning trial is promoted to a full two-phase run and logged to MLflow.

### Production REST API

The FastAPI application (`src/api/app.py`) exposes two inference endpoints and a full operations suite:

- `POST /v2/detect` — single-image upload (JPEG/PNG, ≤10 MB), returns normalised bounding box and confidence
- `POST /v2/detect/batch` — up to 32 base64-encoded images per request
- `GET /health` — liveness probe with model load status and uptime
- `GET /ready` — returns 503 until model singleton is initialised
- `GET /metrics/summary` — total requests, errors, average latency, cache hit rate

An HTTP middleware injects a UUID per request, tracks end-to-end latency in milliseconds, and appends `X-Request-ID` and `X-Latency-MS` to every response header.

### SHA-256 Inference Cache

Every inference result is stored against a SHA-256 hash of the raw image bytes in an in-process LRU cache (capacity: 512 entries). Identical uploads from the same or different clients are served from cache without invoking the model. In the Docker Compose stack, this is backed by Redis 7 configured with `allkeys-lru` eviction and 512 MB memory ceiling — providing shared cache state across multiple API worker processes.

### Versioned Model Registry with Promotion Gates

`src/services/model_registry.py` implements a local-filesystem registry with a stage lifecycle: `experimental → staging → production → archived`. Promotion to staging requires a minimum mean IoU of 0.70; promotion to production requires 0.73 (configurable in `configs/train_config.yaml` under `quality_gates`). The `latest` symlink always points to the active production model, which the API loads on startup via `lifespan()`.

### ETL Pipeline

`src/data/etl_pipeline.py` implements four stages: **Extract** (discover images from local paths or cloud sources), **Validate** (checksum verification, corruption detection, SHA-256 deduplication), **Transform** (resize, normalise, YOLO annotation parsing), **Load** (write NumPy shards and a DVC-tracked manifest). Each stage maps cleanly to an Airflow task for scheduled execution.

### Data Augmentation

`src/data/preprocessing.py` builds a `DatasetBuilder` that applies training-time augmentation: horizontal flip, rotation (±15°), brightness/contrast jitter, and Gaussian noise. The default `augmentation_factor=3` triples the effective training set size. All transformations preserve bounding box normalisation.

### Model Drift Monitoring

`src/monitoring/drift.py` implements Population Stability Index (PSI) and KL-divergence tracking against a baseline prediction distribution captured at deployment time. PSI thresholds follow the standard classification: below 0.1 (stable), 0.1–0.2 (monitor), above 0.2 (retrain). Sliding-window statistics are maintained in a `deque` for memory-efficient streaming evaluation.

### JWT + API Key Authentication

`src/api/auth.py` implements RBAC with three roles — `viewer` (health/metrics read), `inference` (adds detection endpoints), `admin` (full access plus model management). Both JWT Bearer (HS256/RS256) and hashed API keys are supported. A sliding-window counter rate-limits failed attempts per IP, and every auth event is written to the audit log.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                         │
│         REST clients · dashcam pipelines · CI scripts       │
└───────────────────┬─────────────────────────────────────────┘
                    │ HTTP/HTTPS
┌───────────────────▼─────────────────────────────────────────┐
│                   FastAPI Inference API                      │
│  /v2/detect  /v2/detect/batch  /health  /ready  /metrics    │
│  Auth middleware → Request-ID injection → Latency tracking  │
└──────┬─────────────────────┬───────────────────────────────-┘
       │ Cache miss          │ Cache hit
┌──────▼──────┐      ┌───────▼──────────────────────────────-┐
│  ML Model   │      │        Redis / In-process LRU Cache    │
│ (singleton) │      │   SHA-256 keyed · TTL · allkeys-lru   │
│ ResNet50 /  │      └───────────────────────────────────────-┘
│ EfficientNet│
│ MobileNetV3 │
└──────┬──────┘
       │
┌──────▼──────────────────────────────────────────────────────┐
│                  Model Registry (filesystem)                 │
│  models/registry/  ·  v1.0.0/ v2.0.0/ latest→v2.0.0/       │
│  Stages: experimental → staging → production → archived      │
└──────┬──────────────────────────────────────────────────────┘
       │
┌──────▼────────────────────────────────────────────────────-─┐
│              Training Pipeline (offline)                     │
│  ETL → Preprocessing → Phase-1 Train → Phase-2 Fine-tune    │
│  Callbacks: EarlyStopping · ModelCheckpoint · TensorBoard    │
│  MLflow experiment tracking · DVC data versioning            │
└──────┬──────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────┐
│                  Observability Stack                         │
│  Prometheus (metrics scrape) · Grafana (dashboards)          │
│  Alert rules: API down · p95 latency · error rate · drift    │
└─────────────────────────────────────────────────────────────┘
```

Request flow for `POST /v2/detect`:
1. Middleware assigns UUID and records start timestamp.
2. Auth middleware validates JWT/API key and enforces `inference` role.
3. Image bytes are read, size-checked, and SHA-256 hashed.
4. Cache hit → return immediately with `cached: true`.
5. Cache miss → `cv2.imdecode`, resize to 224×224, normalise, expand to `(1, 224, 224, 3)`.
6. Model singleton runs `predict(batch)`, returning a `(4,)` float32 bbox array.
7. Response is stored in cache and returned; middleware appends latency headers.

**Scalability:** The Kubernetes manifest deploys 3 API replicas with a `RollingUpdate` strategy (`maxUnavailable: 1`, `maxSurge: 1`). Prometheus annotations on pod templates enable automatic metrics scraping. Redis provides shared cache state across all replicas.

---

## Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| Deep Learning | TensorFlow 2.15 / Keras | Mature transfer learning API; SavedModel / `.keras` serialisation |
| Computer Vision | OpenCV 4.9, Pillow 10 | Image decode, resize, colour conversion |
| API Framework | FastAPI 0.111 | Async-native, auto-generated OpenAPI docs, Pydantic validation |
| API Server | Uvicorn + Gunicorn | ASGI; multi-worker production serving |
| Caching | Redis 7 | Shared LRU cache across API replicas; `allkeys-lru` eviction |
| Experiment Tracking | MLflow 2.12 | Run comparison, artifact registry, Keras model logging |
| Data Versioning | DVC | Reproducible pipelines; stage dependency graph |
| HPO | Optuna | TPE sampler, MedianPruner; MLflow integration |
| Containerisation | Docker (multi-stage) | ~300 MB runtime image vs ~1.5 GB naive build |
| Orchestration | Kubernetes | 3-replica deployment, ConfigMaps, Secrets, Ingress |
| Metrics | Prometheus 2.51 | 10s scrape interval; alert rules for latency, errors, drift |
| Dashboards | Grafana 10.4 | Grafana piechart-panel plugin; provisioned datasources |
| IaC | Terraform | Cloud infrastructure provisioning |
| Code Quality | Ruff, Black, mypy | Lint, format, type-check enforced in CI |
| Testing | pytest, pytest-asyncio, httpx | Unit + async integration tests; 80% coverage gate |
| Security | Bandit, Safety | Static security analysis; dependency vulnerability scanning |
| Explainability | Grad-CAM, LIME, SHAP | Saliency maps per bounding box coordinate; LIME surrogate models |
| Numerical | NumPy 1.26, SciPy, scikit-learn | Vectorised IoU, dataset splits, statistical drift metrics |

---

## Repository Structure

```
trafficvision-ai/
├── src/                        # Application source code
│   ├── api/
│   │   ├── app.py              # FastAPI application factory, all endpoints
│   │   └── auth.py             # JWT + API key auth, RBAC roles
│   ├── core/
│   │   ├── config.py           # Centralised dataclass config (ModelConfig, APIConfig, …)
│   │   ├── exceptions.py       # Domain exception hierarchy
│   │   └── logging.py          # Structured JSON logging configuration
│   ├── data/
│   │   ├── etl_pipeline.py     # 4-stage ETL: extract → validate → transform → load
│   │   └── preprocessing.py    # DatasetBuilder: resize, normalise, augment, YOLO parse
│   ├── ml/
│   │   ├── model.py            # BackboneFactory, TrafficDetectionModel, EnsembleDetector
│   │   ├── trainer.py          # TrainingOrchestrator: 2-phase training, MLflow logging
│   │   ├── distributed_training.py  # MirroredStrategy / MultiWorker / TPU wrappers
│   │   ├── batch_inference.py  # High-throughput offline inference pipeline
│   │   ├── evaluation.py       # Eval metrics: IoU, precision@IoU, latency percentiles
│   │   ├── explainability.py   # GradCAM, GuidedBackprop, LIMEExplainer
│   │   └── hpo.py              # Optuna HPO loop with MLflow trial logging
│   ├── monitoring/
│   │   └── drift.py            # PSI, KL-divergence, sliding-window prediction drift
│   ├── services/
│   │   └── model_registry.py   # Versioned registry with stage promotion gates
│   └── utils/
│       └── caching.py          # Redis + in-process LRU cache abstraction
├── configs/
│   ├── train_config.yaml       # Model, data, ensemble, MLflow, HPO, quality gate config
│   └── uvicorn_log.json        # Uvicorn structured log configuration
├── scripts/
│   ├── train.py                # CLI entrypoint: build → train → evaluate → register
│   ├── evaluate.py             # Standalone evaluation with benchmark latency runs
│   └── load_test.py            # HTTP load testing against the inference API
├── deploy/
│   ├── docker/
│   │   ├── Dockerfile          # Multi-stage build (builder → runtime, ~300 MB)
│   │   └── docker-compose.yml  # Full local stack: api, worker, redis, prometheus, grafana, mlflow, postgres
│   └── k8s/
│       ├── deployment.yaml     # Namespace, ConfigMap, Deployment (3 replicas), Service
│       └── ingress.yaml        # Ingress rules
├── monitoring/
│   └── prometheus/
│       ├── prometheus.yml      # Scrape configs: api, redis-exporter, node-exporter
│       └── alert_rules.yml     # Alerts: APIDown, HighP95Latency, HighErrorRate, drift
├── infrastructure/
│   └── terraform/
│       └── main.tf             # Cloud infrastructure as code
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_model_architecture.ipynb
│   ├── 03_mlops_pipeline.ipynb
│   ├── 04_explainable_ai.ipynb
│   └── 05_benchmarking_and_optimization.ipynb
├── tests/
│   ├── unit/test_core.py       # Config, exceptions, caching unit tests
│   └── integration/test_api.py # Async API integration tests via httpx
├── dvc.yaml                    # DVC pipeline: etl → train → evaluate → hpo
├── pyproject.toml              # Build config, tool settings (ruff, black, mypy, pytest)
├── requirements.txt            # Runtime dependencies
└── requirements-dev.txt        # Dev + test dependencies
```

---

## Installation & Environment Setup

### Prerequisites

- Python 3.10 or 3.11
- Docker + Docker Compose (for the full stack)
- kubectl + a running cluster (for Kubernetes deployment)

### Local Development Setup

```bash
git clone https://github.com/wittyswayam/trafficvision-ai.git
cd trafficvision-ai

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements-dev.txt
```

Create a `.env` file in the project root:

```bash
APP_ENV=development
SECRET_KEY=dev-secret-change-in-prod
REDIS_HOST=localhost
REDIS_PORT=6379
MLFLOW_TRACKING_URI=http://localhost:5000
POSTGRES_PASSWORD=mlflow_dev
GRAFANA_PASSWORD=admin
API_PORT=8000
API_WORKERS=2
```

### Docker Setup (Full Stack)

```bash
docker compose -f deploy/docker/docker-compose.yml up -d

# Verify all services are healthy
docker compose -f deploy/docker/docker-compose.yml ps

# Stream API logs
docker compose -f deploy/docker/docker-compose.yml logs -f api
```

Services available after startup:

| Service | URL |
|---|---|
| TrafficVision API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

### Kubernetes Setup

```bash
# Build and push image to your registry
docker build -t your-registry/trafficvision-ai:2.0.0 .
docker push your-registry/trafficvision-ai:2.0.0

# Update image reference in deploy/k8s/deployment.yaml, then apply
kubectl apply -f deploy/k8s/

# Verify rollout
kubectl rollout status deployment/trafficvision-api -n trafficvision
kubectl get pods -n trafficvision
```

### Environment Variable Reference

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | Runtime profile: `development`, `staging`, `production` |
| `SECRET_KEY` | — | JWT signing key; must be set in production |
| `REDIS_HOST` | `redis` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `API_PORT` | `8000` | Uvicorn bind port |
| `API_WORKERS` | `4` | Number of Uvicorn worker processes |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URI |
| `POSTGRES_PASSWORD` | `mlflow_dev` | PostgreSQL password for MLflow backend |
| `GRAFANA_PASSWORD` | `admin` | Grafana admin password |

**Common setup issues:**

- *Model not loaded on startup:* The API logs a warning and returns HTTP 503 on inference requests until `models/registry/latest/model.keras` exists. Run training first or mount a pre-trained model volume.
- *OpenCV import error on slim images:* Ensure `libgl1-mesa-glx` is installed — the Dockerfile handles this in both build stages.
- *Redis connection refused in local dev:* Start Redis separately (`docker run -p 6379:6379 redis:7-alpine`) or run the full Docker Compose stack.

---

## Usage Guide

### Training a Model

```bash
# Basic training with default ResNet50 backbone
python scripts/train.py \
  --backbone resnet50 \
  --image-dir data/raw/train/images \
  --label-dir data/raw/train/labels \
  --output-dir models/registry/latest

# EfficientNetB3 with MLflow tracking
python scripts/train.py \
  --backbone efficientnetb3 \
  --epochs-phase1 30 \
  --epochs-phase2 20 \
  --batch-size 32 \
  --mlflow \
  --mlflow-uri http://localhost:5000

# DVC-managed full pipeline (ETL → train → evaluate)
dvc repro
```

### Running the API

```bash
# Development server (auto-reload)
uvicorn src.api.app:app --reload --port 8000

# Production (4 workers, structured logs)
uvicorn src.api.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-config configs/uvicorn_log.json
```

### Inference

Single image upload:

```bash
curl -X POST http://localhost:8000/v2/detect \
  -H "Authorization: Bearer <token>" \
  -F "file=@/path/to/image.jpg"
```

Batch inference (base64):

```bash
python -c "
import base64, json, requests
with open('image.jpg','rb') as f:
    b64 = base64.b64encode(f.read()).decode()
resp = requests.post(
    'http://localhost:8000/v2/detect/batch',
    json={'images_b64': [b64]},
    headers={'Authorization': 'Bearer <token>'}
)
print(json.dumps(resp.json(), indent=2))
"
```

### Batch Offline Inference

```bash
python -m scripts.batch_infer \
  --input-dir /data/dashcam_images \
  --output-dir /data/results \
  --model-path models/registry/latest \
  --batch-size 64 \
  --workers 8
```

### Hyperparameter Optimisation

```bash
python scripts/run_hpo.py \
  --n-trials 30 \
  --output-dir models/hpo
```

---

## API Documentation

### `POST /v2/detect`

**Request:** `multipart/form-data`; field `file` — JPEG or PNG, max 10 MB.

**Response:**
```json
{
  "request_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "model_version": "2.0.0",
  "inference_latency_ms": 42.17,
  "cached": false,
  "detections": [
    {
      "x_min": 0.21,
      "y_min": 0.34,
      "x_max": 0.58,
      "y_max": 0.71,
      "confidence": 0.87
    }
  ]
}
```

All bounding box coordinates are normalised to `[0, 1]` relative to image dimensions.

### `POST /v2/detect/batch`

**Request body:**
```json
{
  "images_b64": ["<base64_image_1>", "<base64_image_2>"]
}
```
Maximum 32 images per request. Returns an array of `DetectionResponse` objects in the same order as the input list.

### `GET /health`

Returns model load status, API version, and uptime in seconds. Safe to call without authentication.

### `GET /metrics/summary`

```json
{
  "total_requests": 14823,
  "total_errors": 12,
  "avg_latency_ms": 38.4,
  "cache_hit_rate": 0.61
}
```

---

## Development Workflow

### Running Tests

```bash
# Full test suite with coverage report
pytest --cov=src --cov-report=term-missing

# Unit tests only
pytest tests/unit/ -v

# Integration tests (requires API to be running or uses httpx async client)
pytest tests/integration/ -v
```

Coverage gate is set at 80% in `pyproject.toml` (`fail_under = 80`).

### Linting and Formatting

```bash
# Format
black src/ scripts/ tests/

# Lint
ruff check src/ scripts/ tests/

# Type check
mypy src/

# Security scan
bandit -r src/
safety check -r requirements.txt
```

### CI/CD

The project is structured for a standard GitHub Actions CI workflow:

1. **Lint & type-check** — Ruff, Black (check mode), mypy, Bandit
2. **Unit tests** — pytest with coverage gate
3. **Integration tests** — pytest with httpx async client
4. **Docker build** — multi-stage build verification
5. **DVC repro** (optional, gated) — full pipeline reproduction on GPU runner

### Branching Conventions

- `main` — production-ready, protected
- `develop` — integration branch
- `feature/<short-description>` — feature branches, PR into `develop`
- `hotfix/<issue>` — critical fixes, PR directly into `main`

Commit messages follow Conventional Commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`.

---

## Contributing Guidelines

1. Fork the repository and create a `feature/<description>` branch from `develop`.
2. Implement changes with corresponding tests. Do not lower the 80% coverage threshold.
3. Run the full local check suite (Black, Ruff, mypy, Bandit, pytest) before opening a PR.
4. Open a PR into `develop` with a description covering: what changed, why, and how to test it.
5. Expect at least one review cycle. The reviewer will check: correctness, test coverage, type annotations, and whether any new dependency is justified.
6. Squash-merge into `develop`; linear history is maintained.

**Do not** commit model weights, datasets, `.env` files, or secrets. Use DVC for data and the model registry for weights.

---

## Performance & Scalability

**Inference caching:** SHA-256-keyed cache eliminates redundant model calls for repeated images. In production (Redis, `allkeys-lru`, 512 MB), cache hit rates above 60% are typical in traffic camera deployments where the same intersection frame recurs across time windows.

**Multi-stage Docker build:** The builder stage compiles all Python C extensions and writes them to `/opt/venv`. The runtime stage copies only the venv — the resulting image is approximately 300 MB versus over 1.5 GB for a naive single-stage build.

**Two-phase training:** Freezing the backbone in Phase 1 allows large batch sizes and fast iteration on the detection head. Phase 2 reduces batch size by half to keep the smaller effective learning rate stable during backbone fine-tuning.

**Mixed precision (distributed training):** `DistributedConfig` enables `mixed_precision=True` by default. With `MirroredStrategy`, TF computes in float16 and maintains float32 master weights, roughly doubling throughput on Ampere-class GPUs.

**Batch offline inference:** `batch_inference.py` uses a `ThreadPoolExecutor` for parallel image loading, decoupled from the GPU inference loop. Images that fail to decode are skipped with logged errors rather than aborting the full run.

**DVC pipeline caching:** DVC caches intermediate outputs (processed shards, trained model) by content hash. Unchanged upstream stages are skipped on `dvc repro`, making iterative experimentation significantly faster.

---

## Security & Reliability

**Authentication:** JWT Bearer tokens (HS256 or RS256) and hashed API keys. Role permissions are enforced at the FastAPI dependency layer — not in business logic — making them hard to accidentally bypass.

**Brute-force protection:** A sliding-window counter in `auth.py` rate-limits failed authentication attempts per IP.

**Non-root container:** The production Dockerfile creates a dedicated `appuser` and switches to it before the entrypoint. The model and log directories are the only writable paths at runtime.

**Health and readiness probes:** The `/health` endpoint is always reachable (returns 200 even if the model is not loaded). The `/ready` endpoint returns 503 until the model singleton is initialised — Kubernetes uses this to gate traffic routing.

**Prometheus alert rules** cover four failure modes: API instance down (1-minute window), p95 latency above 500ms (warning) and 1000ms (critical), HTTP 5xx error rate above 5%, and prediction drift (PSI > 0.2).

**Structured logging:** `src/core/logging.py` emits JSON-structured logs tagged with request ID, enabling correlation across distributed traces. Uvicorn access logs use the same format via `configs/uvicorn_log.json`.

**Model promotion gates:** The registry enforces minimum mean IoU thresholds before a model can advance to staging or production, preventing accidental deployment of regressed models.

---

## Roadmap

- **YOLO-format multi-class detection:** The current bounding box head regresses a single box per image. Extending to per-class multi-box outputs (anchor-based or anchor-free) is the highest-priority capability gap for real-world traffic analysis.
- **Streaming inference endpoint:** A WebSocket or Server-Sent Events endpoint for live camera feeds without the overhead of repeated HTTP handshakes.
- **Cloud model registry backend:** The current registry uses local filesystem paths. Swapping `_save`/`_load` in `ModelRegistry` to S3 or GCS is noted in the code and is a near-term target.
- **ONNX export and TensorRT optimisation:** Converting trained `.keras` models to ONNX and quantising with TensorRT would reduce p99 inference latency for GPU-constrained edge deployments.
- **Airflow DAG integration:** The ETL pipeline is structured for Airflow — formalising the DAG definition and scheduling retraining on data drift signals is planned.
- **Known limitations:** The ensemble uncertainty estimate is based on prediction variance across backbones, not a calibrated probability. Confidence scores on the single-model path are a geometric proxy rather than a calibrated output.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for full terms.

---

*Built by [wittyswayam](https://github.com/wittyswayam)*
