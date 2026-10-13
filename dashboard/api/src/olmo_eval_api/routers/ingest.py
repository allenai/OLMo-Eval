"""Ingest routes (/v1/*), used by the olmo-eval upload client.

Every route except /health requires a verified Google access token (auth/google_token.py).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Path, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from olmo_eval_api.auth.google_token import Principal, require_principal
from olmo_eval_api.db.engine import get_session
from olmo_eval_api.schemas import ingest as s
from olmo_eval_api.services import ingest as svc
from olmo_eval_api.settings import Settings
from olmo_eval_api.storage.base import Storage

PROTOCOL_VERSIONS = [1]

router = APIRouter(prefix="/v1", tags=["ingest"])

RunIdPath = Annotated[str, Path(pattern=s.RUN_ID_PATTERN)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
PrincipalDep = Annotated[Principal, Depends(require_principal)]
# The run's write secret (services.ingest.can_write). Clients send it on every run write.
RunSecretDep = Annotated[str | None, Header(alias="X-Olmo-Eval-Run-Secret", max_length=256)]


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _storage(request: Request) -> Storage:
    return request.app.state.storage


@router.get("/whoami", response_model=s.WhoAmIResponse)
async def whoami(principal: PrincipalDep) -> s.WhoAmIResponse:
    return s.WhoAmIResponse(
        email=principal.email,
        principal_type=principal.principal_type,
        expires_in=principal.expires_in,
        protocol_versions=PROTOCOL_VERSIONS,
    )


@router.put("/runs/{run_id}", response_model=s.RunUpsertResponse)
async def put_run(
    run_id: RunIdPath,
    body: s.RunUpsertRequest,
    request: Request,
    session: SessionDep,
    principal: PrincipalDep,
    run_secret: RunSecretDep = None,
) -> s.RunUpsertResponse:
    return await svc.upsert_run(session, _settings(request), run_id, body, principal, run_secret)


@router.delete("/runs/{run_id}", status_code=204)
async def delete_run(
    run_id: RunIdPath, request: Request, session: SessionDep, principal: PrincipalDep
) -> Response:
    await svc.delete_run(session, _storage(request), _settings(request), run_id, principal)
    return Response(status_code=204)


@router.post("/runs/{run_id}/artifacts:sign", response_model=s.SignArtifactsResponse)
async def sign_artifacts(
    run_id: RunIdPath,
    body: s.SignArtifactsRequest,
    request: Request,
    session: SessionDep,
    principal: PrincipalDep,
    run_secret: RunSecretDep = None,
) -> s.SignArtifactsResponse:
    await svc.require_write(session, run_id, principal, run_secret)
    return await svc.sign_artifacts(session, _storage(request), _settings(request), run_id, body)


@router.post("/runs/{run_id}/task-results", response_model=s.TaskResultUpsertResponse)
async def post_task_result(
    run_id: RunIdPath,
    body: s.TaskResultIn,
    session: SessionDep,
    principal: PrincipalDep,
    run_secret: RunSecretDep = None,
) -> s.TaskResultUpsertResponse:
    await svc.require_write(session, run_id, principal, run_secret)
    return await svc.upsert_task_result(session, run_id, body)


@router.post("/task-results/{task_result_id}/instances", response_model=s.InstanceBatchResponse)
async def post_instances(
    task_result_id: int,
    body: s.InstanceBatchRequest,
    session: SessionDep,
    principal: PrincipalDep,
    run_secret: RunSecretDep = None,
) -> s.InstanceBatchResponse:
    await svc.require_task_result_write(session, task_result_id, principal, run_secret)
    return await svc.add_instances(session, task_result_id, body)


@router.put("/runs/{run_id}/inference", response_model=s.InferenceUploadResponse)
async def put_inference(
    run_id: RunIdPath,
    body: s.InferenceUploadRequest,
    session: SessionDep,
    principal: PrincipalDep,
    run_secret: RunSecretDep = None,
) -> s.InferenceUploadResponse:
    await svc.require_write(session, run_id, principal, run_secret)
    return await svc.put_inference(session, run_id, body)


@router.post("/runs/{run_id}/complete", response_model=s.CompleteResponse)
async def complete(
    run_id: RunIdPath,
    body: s.CompleteRequest,
    request: Request,
    session: SessionDep,
    principal: PrincipalDep,
    run_secret: RunSecretDep = None,
) -> s.CompleteResponse:
    await svc.require_write(session, run_id, principal, run_secret)
    return await svc.complete_run(session, _storage(request), _settings(request), run_id, body)
