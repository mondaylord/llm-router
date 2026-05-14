"""FastAPI service exposing the router over HTTP.

Production-friendly defaults:
- async route handler so a slow embedding call doesn't block the loop;
- structured logging configured on startup;
- config path read from env `LLM_ROUTER_CONFIG` so the same image can
  serve different tenants by mounting different files.

For higher concurrency, run with `uvicorn --workers N`. The Router and
its loaded classifier are constructed once per process.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from llm_router.core.config import RouterConfig
from llm_router.core.decision import RoutingRequest
from llm_router.core.router import Router
from llm_router.observability.logger import configure_logging, get_logger
from llm_router.server.schemas import HealthResponse, RouteRequestBody, RouteResponseBody

log = get_logger(__name__)


def _build_router_from_env() -> Router:
    cfg_path = os.environ.get("LLM_ROUTER_CONFIG")
    config = RouterConfig.load(cfg_path) if cfg_path else RouterConfig.default()
    configure_logging(level=config.logging.level, json=config.logging.json_format)
    log.info("router_starting", config_path=cfg_path)
    return Router.from_config(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.router = _build_router_from_env()
    yield


app = FastAPI(
    title="llm-router",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    router: Router = app.state.router
    return HealthResponse(
        status="ok",
        classifier_loaded=router.classifier is not None,
        rules_count=len(router.rules),
    )


@app.post("/route", response_model=RouteResponseBody)
async def route(body: RouteRequestBody) -> RouteResponseBody:
    if not body.prompt and not body.messages:
        raise HTTPException(
            status_code=400, detail="either `prompt` or `messages` is required"
        )
    router: Router = app.state.router
    decision = router.route(
        RoutingRequest(
            prompt=body.prompt,
            messages=body.messages,
            session_id=body.session_id,
            tenant_id=body.tenant_id,
            agent_step_type=body.agent_step_type,
            available_tools=body.available_tools,
            planned_tool=body.planned_tool,
            last_tool_called=body.last_tool_called,
            recent_outcomes=body.recent_outcomes,
            total_context_tokens=body.total_context_tokens,
            history_turns=body.history_turns,
            metadata=body.metadata,
        )
    )
    return RouteResponseBody.from_decision(decision)


def main() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "llm_router.server.app:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        workers=int(os.environ.get("WORKERS", "1")),
        log_config=None,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
