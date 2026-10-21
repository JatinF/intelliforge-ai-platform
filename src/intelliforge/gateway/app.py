"""FastAPI gateway. The tenant and role come from the bearer token."""

from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict

from intelliforge import __version__
from intelliforge.governance.auth import PlatformStore
from intelliforge.investigate import decide_approval, investigate


class InvestigateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    policy: str = "quality"
    action: str = "diagnose"


class DecisionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: str


def _bearer(authorization: str | None, store: PlatformStore) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="unauthorized")
    token = authorization.split(" ", 1)[1].strip()
    principal = store.principal(token)
    if principal is None:
        raise HTTPException(status_code=401, detail="unauthorized")
    return principal


def create_app() -> FastAPI:
    app = FastAPI(title="IntelliForge", version=__version__)
    app.state.store = PlatformStore()

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "real_customer_data": 0, "real_equipment": 0, "mode": "local-mock"}

    @app.post("/v1/investigations")
    def start(body: InvestigateBody, authorization: str | None = Header(default=None)) -> dict:
        principal = _bearer(authorization, app.state.store)
        try:
            return investigate(principal, body.case_id, body.policy, body.action, app.state.store)
        except PermissionError:
            raise HTTPException(status_code=403, detail="forbidden") from None
        except KeyError:
            raise HTTPException(status_code=404, detail="case not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.post("/v1/approvals/{approval_id}/decision")
    def decision(approval_id: str, body: DecisionBody, authorization: str | None = Header(default=None)) -> dict:
        principal = _bearer(authorization, app.state.store)
        try:
            return decide_approval(principal, approval_id, body.decision, app.state.store)
        except PermissionError:
            raise HTTPException(status_code=403, detail="forbidden") from None
        except KeyError:
            raise HTTPException(status_code=404, detail="approval not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.get("/v1/audit/{audit_id}")
    def audit(audit_id: str, authorization: str | None = Header(default=None)) -> dict:
        principal = _bearer(authorization, app.state.store)
        for record in app.state.store.audits:
            if record["audit_id"] == audit_id and record["tenant"] == principal["tenant"]:
                return record
        raise HTTPException(status_code=404, detail="audit not found")

    return app
