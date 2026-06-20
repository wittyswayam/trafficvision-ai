# TrafficVision-AI

CNN-based road traffic sign and light bounding-box detection, built on transfer learning, with a FastAPI inference service and the surrounding MLOps tooling (training pipeline, model registry, drift monitoring, deployment manifests) wrapped around it.

The project started as a single Kaggle notebook (`CNN-based-Road-Traffic-Sign-and-Light-Recognition.ipynb`, still in the repo root) and was restructured into a proper `src/` package with a training CLI, evaluation framework, and a production-shaped API on top of the same core model logic.

## Overview

The model takes an image and predicts a single bounding box (`x_min, y_min, x_max, y_max`, normalized 0–1) for the dominant traffic sign or light in the frame. It's a regression head on top of a frozen, ImageNet-pretrained CNN backbone — not a multi-object detector like YOLO or Faster R-CNN. That tradeoff is intentional: with a modest labeled dataset, transfer learning with a frozen backbone converges fast and avoids the data appetite of training a detector from scratch.

On top of the model itself, the repo includes the kind of scaffolding you'd want before putting something like this in front of real traffic — a versioned model registry, drift detection on incoming predictions, an evaluation framework with latency percentiles and IoU thresholds, Grad-CAM/LIME explainability, and Docker/Kubernetes/Terraform definitions for running it somewhere other than a laptop.

## Features

- **Bounding-box regression** via three swappable CNN backbones: ResNet50, EfficientNetB3, MobileNetV3Large
- **Two-phase training**: Phase 1 trains a frozen-backbone head, Phase 2 fine-tunes the top backbone layers at a reduced learning rate
- **Combined loss function**: weighted MSE + IoU loss (`α·MSE + (1-α)·IoU_loss`), tunable via config
- **Ensemble inference** (`EnsembleDetector`) — weighted-average predictions across multiple backbones with variance-based uncertainty estimates
- **FastAPI inference service** with single-image and batch (up to 32 images) detection endpoints, request ID tracking, latency headers, and an in-process SHA-256-keyed inference cache
- **JWT + API key authentication module** with role-based permissions (viewer / inference / admin) and a sliding-window rate limiter — implemented as a standalone module (see [Known Gaps](#known-gaps) for wiring status)
- **Explainability**: Grad-CAM with auto-detected target conv layer, Guided Backprop, and a LIME-based local explainer
- **Drift monitoring**: Population Stability Index (PSI) and KL divergence over prediction/feature distributions, with configurable alert thresholds
- **Model registry**: filesystem-based versioning with experimental → staging → production → archived promotion gates and a `latest` pointer
- **Hyperparameter search space** defined for Optuna (TPE sampler, median pruner) covering backbone, learning rate, batch size, dropout, loss weighting, and fine-tune depth
- **Data pipeline**: YOLO-format annotation parsing, image validation/dedup, augmentation, and an ETL flow designed to plug into Airflow or DVC
- **Batch inference CLI** for offline scoring of image directories with parallel loading and JSONL output
- **Load testing script** with sequential, concurrent (ThreadPoolExecutor), and Locust-compatible modes
- **Structured JSON logging** with request/trace ID propagation, suitable for ELK/Datadog/CloudWatch ingestion
- **Full deployment stack**: multi-stage Dockerfile, docker-compose (API, Redis, Prometheus, Grafana, MLflow, Postgres), Kubernetes manifests, and Terraform for AWS (EKS, ElastiCache, RDS, S3, ECR)

## Tech Stack

**Model / ML**
- TensorFlow / Keras 2.15
- OpenCV (`opencv-python-headless`) for image decode/resize
- scikit-learn (train/val/test splitting)
- Optuna search space (HPO module references it; not pinned in `requirements.txt` — see gaps)
- MLflow for experiment tracking (dev dependency)
- SHAP listed as a dev dependency alongside the custom Grad-CAM/LIME implementation

**API / Service**
- FastAPI + Pydantic v2
- Uvicorn (ASGI server, multi-worker)
- Redis (caching layer abstraction; in-process LRU cache is the actual fallback used in `app.py`)
- Prometheus client (metrics exposition target)

**Data / Pipeline**
- DVC pipeline definition (`dvc.yaml`) covering ETL → train → evaluate → HPO stages
- pandas, NumPy, SciPy

**Infra / Ops**
- Docker (multi-stage build), Docker Compose
- Kubernetes manifests (Deployment, ConfigMap, Ingress)
- Terraform for AWS (EKS, ElastiCache, RDS Aurora, S3, ECR, IAM)
- Prometheus + Grafana for metrics/dashboards

**Testing / Quality**
- pytest, pytest-asyncio, httpx (async API testing)
- ruff, black, mypy, bandit, safety

## Project Structure

```
trafficvision-ai-main/
├── src/
│   ├── api/
│   │   ├── app.py              # FastAPI app, routes, caching, metrics
│   │   └── auth.py             # JWT + API key auth, RBAC (standalone module)
│   ├── core/
│   │   ├── config.py           # Dataclass-based config (model/data/env)
│   │   ├── exceptions.py       # Domain exception hierarchy → HTTP status mapping
│   │   └── logging.py          # JSON structured logging, context propagation
│   ├── data/
│   │   ├── etl_pipeline.py     # Extract/validate/transform/load stages
│   │   └── preprocessing.py    # BoundingBox, YOLO parsing, augmentation, DatasetBuilder
│   ├── ml/
│   │   ├── model.py            # Backbone factory, TrafficDetectionModel, EnsembleDetector
│   │   ├── trainer.py          # Two-phase training orchestrator
│   │   ├── evaluation.py       # IoU/latency/calibration evaluation framework
│   │   ├── explainability.py   # Grad-CAM, Guided Backprop, LIME
│   │   ├── hpo.py              # Optuna search space + trial result tracking
│   │   ├── distributed_training.py  # Multi-GPU/TPU strategy selection
│   │   └── batch_inference.py  # Offline batch scoring pipeline
│   ├── services/
│   │   └── model_registry.py   # Versioned model storage with promotion gates
│   ├── monitoring/
│   │   └── drift.py            # PSI, KL divergence, alert thresholds
│   └── utils/
│       └── caching.py          # Cache backend abstraction (Redis / in-memory LRU)
├── scripts/
│   ├── train.py                 # Training CLI entry point
│   ├── evaluate.py              # Evaluation/benchmark CLI
│   └── load_test.py             # API load testing (sequential/concurrent/Locust)
├── notebooks/                    # Pipeline, architecture, MLOps, XAI, benchmarking notebooks
├── configs/
│   ├── train_config.yaml        # Model/data/quality-gate config used by training & DVC
│   └── uvicorn_log.json         # Uvicorn logging config used in the Dockerfile
├── deploy/
│   ├── docker/                  # Dockerfile + docker-compose.yml
│   └── k8s/                     # Deployment, Ingress manifests
├── infrastructure/terraform/     # AWS infra (EKS, ElastiCache, RDS, S3, ECR)
├── monitoring/prometheus/        # Scrape config + alert rules
├── tests/
│   ├── unit/                    # BoundingBox, core component tests
│   └── integration/             # FastAPI endpoint tests (httpx/TestClient)
├── docs/architecture/ARCHITECTURE.md  # ADRs explaining key design decisions
├── dvc.yaml                      # ETL → train → evaluate → HPO pipeline definition
├── CNN-based-Road-Traffic-Sign-and-Light-Recognition.ipynb  # Original Kaggle notebook
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

## Installation

Requires Python 3.10 or 3.11.

```bash
git clone <github.com/wittyswayam/trafficvision-ai>
cd trafficvision-ai-main

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
# for development (tests, linting, MLflow, notebooks, SHAP, profiling):
pip install -r requirements-dev.txt
```

OpenCV needs a few system libraries if you're not using the Docker image — `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender-dev`, `libgl1-mesa-glx` on Debian/Ubuntu (the Dockerfile installs these for you).

## Configuration

Configuration is centralized in `src/core/config.py` as a set of dataclasses (`ModelConfig`, `DataConfig`, etc.), with `configs/train_config.yaml` overriding training defaults and the DVC pipeline reading from the same file.

Runtime environment variables used across the Docker/Compose/Kubernetes setup:

```bash
# .env (example)
APP_ENV=development                 # development | staging | production
SECRET_KEY=change-me-to-32-chars-minimum   # required by JWTHandler (min 32 chars)

# API
API_PORT=8000
API_WORKERS=4

# Redis (caching layer)
REDIS_HOST=redis
REDIS_PORT=6379

# MLflow (experiment tracking, optional)
MLFLOW_TRACKING_URI=http://localhost:5000

# Postgres (MLflow backend store, only used via docker-compose)
POSTGRES_PASSWORD=mlflow_dev

# Grafana (only used via docker-compose)
GRAFANA_PASSWORD=admin
```

Training-specific values (backbone choice, learning rate, loss weighting, quality gates for staging/production promotion) live in `configs/train_config.yaml` rather than environment variables.

## Running the Project

### Train a model

```bash
python scripts/train.py \
  --backbone efficientnetb3 \
  --image-dir data/raw/train/images \
  --label-dir data/raw/train/labels \
  --epochs-phase1 30 \
  --epochs-phase2 20 \
  --output-dir models/registry/latest
```

Labels are expected in YOLO `.txt` format alongside a matching image directory. The script does a 70/15/15 train/val/test split, runs Phase 1 (frozen backbone) and Phase 2 (fine-tuning) sequentially, and writes the trained model + metrics to `--output-dir`. Pass `--mlflow` to log the run to an MLflow tracking server.

### Evaluate a trained model

```bash
python scripts/evaluate.py \
  --model-path models/registry/latest \
  --test-data data/processed \
  --benchmark-runs 100
```

Produces IoU metrics at multiple thresholds, latency percentiles (p50/p95/p99), and writes an evaluation report to `models/evaluation/`.

### Run the inference API locally

```bash
uvicorn src.api.app:app --reload --port 8000
```

The app loads a model from `models/registry/latest` at startup if one exists; if not, it starts anyway and inference endpoints return `503` until a model is trained and placed there. Interactive docs are at `http://localhost:8000/docs`.

### Run the full local stack (API + Redis + Prometheus + Grafana + MLflow + Postgres)

```bash
cd deploy/docker
docker compose up -d
docker compose logs -f api
```

This brings up the API on `:8000`, Redis on `:6379`, Prometheus on `:9090`, Grafana on `:3000`, and MLflow on `:5000`. The compose file also defines a `worker` service (`python -m scripts.worker`) for background/batch processing — that script doesn't exist in the repo yet, so the worker container will fail to start until it's added.

### Deploy to Kubernetes

```bash
kubectl apply -f deploy/k8s/
```

Applies the namespace, ConfigMap, and Deployment (3 replicas, rolling update strategy, Prometheus scrape annotations).

### Provision AWS infrastructure

```bash
cd infrastructure/terraform
terraform init
terraform plan -var-file=environments/production.tfvars
terraform apply
```

Provisions an EKS cluster, ElastiCache Redis, an S3-backed model registry, RDS Aurora for MLflow, an ECR repository, and supporting IAM roles. State is stored remotely in S3 with DynamoDB locking.

## API Endpoints

All routes are defined in `src/api/app.py`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — returns status, model version, whether a model is loaded, uptime |
| `GET` | `/ready` | Readiness probe — `503` until a model is loaded |
| `GET` | `/metrics/summary` | Request count, error count, average latency, cache hit rate |
| `POST` | `/v2/detect` | Single-image detection. Multipart upload, JPEG/PNG, 10MB max |
| `POST` | `/v2/detect/batch` | Batch detection for up to 32 base64-encoded images in one request |
| `GET` | `/docs`, `/redoc` | Auto-generated OpenAPI documentation |

`POST /v2/detect` response shape:

```json
{
  "request_id": "f3a2...",
  "model_version": "2.0.0",
  "inference_latency_ms": 42.18,
  "detections": [
    {
      "x_min": 0.21,
      "y_min": 0.34,
      "x_max": 0.58,
      "y_max": 0.71,
      "confidence": 0.93
    }
  ],
  "cached": false
}
```

`detections` is always a single-element list — the model predicts one bounding box per image, and `confidence` here is derived from predicted box area rather than a learned classification score.

**Auth note:** `src/api/auth.py` implements JWT + API key authentication with RBAC and rate limiting, but `create_app()` in `app.py` doesn't currently apply `make_auth_dependency(...)` to any route — as shipped, the API is unauthenticated. Wiring it in is a matter of adding the dependency to the relevant route decorators.

## Usage Examples

**Single image, curl:**

```bash
curl -X POST http://localhost:8000/v2/detect \
  -F "file=@dashcam_frame.jpg"
```

**Batch detection, Python:**

```python
import base64
import requests

images_b64 = [
    base64.b64encode(open(f"frame_{i}.jpg", "rb").read()).decode()
    for i in range(5)
]

resp = requests.post(
    "http://localhost:8000/v2/detect/batch",
    json={"images_b64": images_b64},
)
for r in resp.json():
    print(r["detections"][0])
```

**Offline batch inference over a directory:**

```bash
python -m src.ml.batch_inference \
  --input-dir /data/dashcam_images \
  --output-dir /data/results \
  --model-path models/registry/latest \
  --batch-size 64 \
  --workers 8
```

**Load testing the running API:**

```bash
python scripts/load_test.py --mode concurrent --max-workers 50 --n-requests 2000
```

## Screenshots

_No screenshots included in this repository. The `/docs` route (Swagger UI) and Grafana dashboards (once provisioned via `docker compose up`) are the most useful things to capture here._



## Future Improvements

Based on where the code currently stops short:

- Wire `auth.py` into the API routes and add an actual API key / user store (it's in-memory today)
- Add the missing `serve.py`, `worker.py`, `run_etl.py`, and `run_hpo.py` entry points referenced by `pyproject.toml`, `docker-compose.yml`, and `dvc.yaml`
- Move from single-box regression toward true multi-object detection if the use case requires recognizing more than one sign/light per frame
- Replace the in-process LRU cache in `app.py` with the Redis backend already implemented in `src/utils/caching.py`
- Add a CI workflow that runs the existing pytest suite, ruff/mypy, and bandit on every PR

