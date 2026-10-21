# IntelliForge AI — Model-Agnostic Intelligence & Evaluation Platform

An open-source reference platform for evaluating, adapting, routing, and governing model calls on synthetic engineering workloads.

Model-agnostic AI platform for open-weight LLM serving, LoRA adaptation, model routing, reproducible evaluations, knowledge retrieval, governed agents, and inference cost optimization.

This is a reference implementation. The equipment records are synthetic (`SYN-` identifiers, `real_equipment: 0`). It does not record production manufacturing experience.

The default path is a CPU mock: a FastAPI gateway, tenant retrieval, a NumPy LoRA head, a SciPy tool, and an approval gate. Qwen 2.5 3B on vLLM is the open-weight slot. It was not started. No GPU is required to run the tests.

**Deployment decision: NO-GO.** See [benchmarks/results/benchmark_report.md](benchmarks/results/benchmark_report.md) and [docs/benchmark-methodology.md](docs/benchmark-methodology.md).

| | |
| --- | --- |
| Package | `intelliforge` 0.1.0 |
| Gateway | FastAPI, tenant taken from the bearer token |
| Local head | NumPy LoRA, rank 2, seed 7 |
| Open-weight slot | `Qwen/Qwen2.5-3B-Instruct` through an OpenAI-compatible vLLM URL. Unset URL stops the run |
| Commercial slot | `grok-4.7` at `https://api.x.ai/v1/chat/completions`. Unset `XAI_API_KEY` stops the run |
| Knowledge | In-process documents and graph. Neo4j and pgvector are compose profiles |
| Tool | SciPy bounded scalar optimization |
| Governance | Reader, engineer, approver. Apply waits for a different approver |
| Decision | NO-GO until a GPU serving run and a hosted call are measured |

## Architecture

An engineer asks why synthetic process `SYN-PROC-ETCH-01` is hot. The gateway copies the tenant from the token, retrieves that tenant's note, walks the equipment graph, and routes the case. The quality policy uses the LoRA head. The cost policy uses the zero head. Either remote policy stops before an HTTP call when its setting is absent. A diagnosis cites evidence and a recommended coolant flow, and it states that the setpoint was not applied.

```mermaid
flowchart TD
  engineer["Engineer with a bearer token"] --> api["FastAPI /v1"]
  api --> retrieval["Tenant documents and graph"]
  retrieval --> router["Policy router"]
  router --> local["Local baseline or LoRA head"]
  local --> scipy["SciPy setpoint tool"]
  scipy --> audit["Audit: evidence, model version, tokens, latency, cost"]
  router --> slot["vLLM Qwen or grok-4.7 slot"]
  slot --> halt["STOPPED, completion_claimed false"]
```

Applying a setpoint is a separate action.

```mermaid
stateDiagram-v2
  [*] --> RUNNING
  RUNNING --> COMPLETED: diagnosis checks pass
  RUNNING --> STOPPED: head and document disagree, or the remote slot is unset
  RUNNING --> AWAITING_APPROVAL: engineer requests apply
  AWAITING_APPROVAL --> REJECTED: approver rejects
  AWAITING_APPROVAL --> EXPIRED: approval is past expiry
  AWAITING_APPROVAL --> UNVERIFIED: recomputed flow does not match
  AWAITING_APPROVAL --> APPLIED: different approver and the tool reread matches
```

| Layer | What the demo runs | Adapter slot |
| --- | --- | --- |
| Gateway | FastAPI and Pydantic bodies that forbid extra fields | — |
| Routing | Quality and cost policies on the local heads | vLLM and xAI, fail closed |
| Adaptation | NumPy LoRA trained in this repository | Hugging Face PEFT, not imported |
| Evaluation | Qualification cases and a local JSON tracking record | MLflow client, not executed |
| Knowledge | Overlap retrieval and a three-hop graph walk | Neo4j and pgvector |
| Quantization | Int8 round-trip error on the local matrices | bitsandbytes |
| Science tool | SciPy `minimize_scalar` | A plant simulator |
| Governance | RBAC, audit rows, approval reread | An external identity provider |

The longer write-up is [docs/architecture.md](docs/architecture.md). Threats and the benchmark method are in [docs/](docs/).

## How this sits with the other references

| | [H100 GPU factory](https://github.com/JatinF/h100-ai-factory-reference-architecture) | [Healthcare AI factory](https://github.com/JatinF/healthcare-ai-factory-platform) | [IdentityGuard](https://github.com/JatinF/identityguard-agent-platform) | [OrchestraScale](https://github.com/JatinF/orchestrascale-airflow-platform) | This repository |
| --- | --- | --- | --- | --- | --- |
| Focus | GPU infrastructure acceptance | Imaging and private clinical inference | Identity-aware agent control | Airflow scheduling | Model routing, local adaptation, and governed tools |
| Outcome | Design and gates for the cluster | Synthetic imaging service path | Synthetic investigation with a durable pause | Measured workstation batch of 5,000 scheduled runs | Measured local head, retrieval, and tool path |

Each repository stands on its own evidence. A green local head here does not measure a GPU, a clinical model, or a cluster scheduler.

## Evidence

| Tag | Meaning in this repo |
| --- | --- |
| `EXAMPLE` | Synthetic cases, documents, and the sensor encoder |
| `ASSUMPTION` | 0.00015 USD per 1,000 whitespace tokens, dated 2026-10-21 |
| `DERIVED` | That price times measured whitespace tokens |
| `SPEC` | grok-4.7 list price from the vendor page on 2026-10-21. Zero calls |
| `GATE` | Deployment `NO-GO`. vLLM, PEFT, bitsandbytes, GPU memory, Neo4j, pgvector, MLflow, and the hosted call are `not-run` |
| `MEASURED` | Rows in [benchmarks/results/benchmark_report.md](benchmarks/results/benchmark_report.md) |

## This workstation run

The figures below are copied from that report. Machine line: Windows 11 AMD64, `processor=Intel64 Family 6 Model 140 Stepping 1, GenuineIntel`, `cpu_count=8`, Python 3.12.10.

| Measurement | Value |
| --- | --- |
| LoRA held-out accuracy | 1.0 |
| Baseline held-out accuracy | 0.333333 |
| Qualification cases | 4 |
| LoRA-route task completion | 1.0 |
| Baseline-route task completion | 0.25 |
| Local diagnose P95, 100 calls | 2.667 ms |
| Whitespace tokens per second | 73436.525774 |
| Int8 max absolute error | 0.00665971 |
| Derived mean cost | 1.56e-05 USD |
| GPU memory, vLLM, Qwen weights, PEFT | not-run |

The token rate counts whitespace tokens in the mock prompt and answer across those 100 calls. It is a local-mock figure.

## Run the tests

Python 3.12 or newer.

```bash
python -m pip install -e ".[dev]"
python benchmarks/run_latency_eval.py
python -m pytest
uvicorn intelliforge.gateway.app:create_app --factory
```

`run_latency_eval.py` retrains the adapter, scores the qualification cases, times 100 local diagnoses, and rewrites `benchmarks/results/`. Pytest does not call AWS and does not call a model API. The workflow in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) installs the package on Ubuntu 24.04 and runs pytest.

Docker Compose profiles `app`, `knowledge`, and `gpu` are in the tree. They were not part of the committed measurement. `local-demo-only` is a fixture password.

## API

Dev tokens are fixtures.

| Token | Tenant | Role |
| --- | --- | --- |
| `dev-reader-a` | `tenant-a` | reader |
| `dev-engineer-a` | `tenant-a` | engineer |
| `dev-approver-a` | `tenant-a` | approver |
| `dev-engineer-b` | `tenant-b` | engineer |

`GET /healthz` reports `real_equipment: 0`. `POST /v1/investigations` with `{"case_id": "SYN-CASE-001"}` runs the quality route. `action` `apply_setpoint` returns `AWAITING_APPROVAL`. `POST /v1/approvals/{id}/decision` accepts `approve` or `reject`. A missing token returns 401. Another tenant's case returns 404.

## Layout

```
src/intelliforge/     gateway, routing, serving, governance, evaluation, knowledge, simulation, lora
training/             lora_train.py, compare_adapters.py, adapter artifact
benchmarks/           quality eval, latency eval, committed results
datasets/synthetic_equipment/   cases, documents, graph
infrastructure/       unapplied Terraform sketch
docs/                 architecture, threats, method
tests/
```
