# Threat model

The records in this repository are synthetic. Identifiers use a `SYN-` prefix. `real_equipment` is 0. There is no customer dataset, no production tool history, and no API key.

## Boundaries the code enforces

| Boundary | Behavior |
| --- | --- |
| Principal | The bearer token supplies tenant, role, and subject. Fixture tokens are `dev-reader-a`, `dev-engineer-a`, `dev-approver-a`, and `dev-engineer-b` |
| Case lookup | A case whose tenant differs from the caller returns 404 |
| Documents | Retrieval drops other-tenant documents before scoring |
| Injected labels | The request body forbids extra fields, including `expected_fault` |
| Remote routes | `open-weight` stops when `VLLM_BASE_URL` is unset. `commercial` stops when `XAI_API_KEY` is unset. The local path does not post to either endpoint |
| Apply | Only an engineer can request `apply_setpoint`. A reader receives 403 |
| Approval | The approver must be a different subject in the same tenant. Rejected, expired, and unverified decisions leave `setpoint_applied` false and do not claim completion |
| Tool reread | An approval stores the flow. Accepting it recomputes the SciPy result and stops when the values differ |
| Audit | `GET /v1/audit/{id}` returns a row only for the caller's tenant |

## Credentials in the tree

| Value | Where | Class |
| --- | --- | --- |
| `local-demo-only` | Compose for Postgres and Neo4j | Fixture password. Those services were not started |
| `dev-engineer-a` and the other `dev-*` tokens | `governance/auth.py` | Fixture map. A deployment replaces it |

## Out of scope for the current evidence

vLLM process isolation, GPU driver access, Neo4j authentication, and a live commercial account were not exercised. Publishing this repository does not review a change to vLLM, PEFT, or Qwen.
