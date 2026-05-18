# TypeScript / Bun client for llm-router

A zero-dependency client for calling llm-router's HTTP API from TS+Bun
agent frameworks. Used by [reel-agent](https://github.com/mondaylord/reel-agent)
(or wherever your agent lives).

## Files

- [llm_router_client.ts](llm_router_client.ts) — the client + types.
- [agent_loop_example.ts](agent_loop_example.ts) — end-to-end demo of
  a full agent loop with planning, tool calls, failure escalation.

## Quick start

```bash
# 1. Run the router service.
cd ../../..
.venv/bin/uvicorn llm_router.server.app:app --port 8000

# 2. Run the demo in another shell.
bun run examples/clients/typescript/agent_loop_example.ts
```

## Importing into your project

Copy `llm_router_client.ts` into your project, or publish llm-router-ts
as a separate package later. Public API is intentionally tiny:

```ts
import { LlmRouterClient, DEFAULT_MODEL_FOR_TIER } from "./llm_router_client";

const router = new LlmRouterClient({
  baseUrl: "http://llm-router.internal:8000",
  timeoutMs: 1500,        // optional; default 1500ms
  fallbackTier: "strong", // optional; what to use if router is unreachable
});

const decision = await router.route({
  prompt: userMessage,
  session_id: tapeId,
  tenant_id: tenantKey,
  agent_step_type: "tool_call",
  planned_tool: "edit_file",
  available_tools: ["read_file", "grep", "edit_file"],
  recent_outcomes: priorFailures,
});

const model = DEFAULT_MODEL_FOR_TIER[decision.tier];
// → invoke `model` via your provider SDK
```

## Failure handling

The client defaults to **fail-safe**: if the router is unreachable or
returns an error, it returns a synthetic decision pinned to the
`fallbackTier` (default `"strong"`). This means a router outage causes
slightly higher cost but never a quality regression.

Pass `{ throwOnError: true }` to `route()` to handle failures yourself.

## Outcome / cascade pipeline

The cascade only works if your agent loop feeds outcomes back. Wire your
verifier's failure verdicts to `OutcomeKind` values:

| Verifier event                          | OutcomeKind                |
| --------------------------------------- | -------------------------- |
| Tool args fail JSON-schema validation   | `tool_schema_error`        |
| Tool returned non-zero / threw          | `tool_execution_error`     |
| Output failed to parse (JSON / diff)    | `parse_error`              |
| Tests / lint / types failed after edit  | `validation_error`         |
| User clicked Retry                      | `retry_attempt`            |
| User clicked Thumbs-Down                | `user_negative_feedback`   |
| Anything else flagged                   | `generic_failure`          |

Keep a rolling window of the last 1-3 outcomes per session and pass them
into the next `route()` call. Older outcomes are stale.

## Schema compatibility

This client is hand-aligned with the FastAPI service in
`src/llm_router/server/schemas.py`. If you change the server schema,
update this client. (TODO: generate from OpenAPI to remove drift —
backlog item.)
