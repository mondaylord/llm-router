/**
 * End-to-end example of how reel-agent (or any TS+Bun agent) drives a
 * full agent loop with llm-router as the routing brain.
 *
 * Run after `uvicorn llm_router.server.app:app --reload` is up on :8000.
 *
 *   bun run examples/clients/typescript/agent_loop_example.ts
 *
 * What it shows:
 *   - explicit step labeling at each loop turn
 *   - outcome feedback after a simulated tool failure
 *   - tier → model resolution on the caller side
 */

import {
  DEFAULT_MODEL_FOR_TIER,
  LlmRouterClient,
  type Outcome,
  type RouteResponse,
} from "./llm_router_client.ts";

const client = new LlmRouterClient({
  baseUrl: process.env.LLM_ROUTER_URL ?? "http://localhost:8000",
});

const SESSION = "demo-tape-001";
const TENANT = "reel-internal";

function logDecision(label: string, d: RouteResponse) {
  const model = DEFAULT_MODEL_FOR_TIER[d.tier];
  console.log(
    `[${d.tier.padStart(6)}] [${d.layer.padStart(18)}] ` +
      `step=${(d.inferred_step_type ?? "-").padEnd(12)} ` +
      `model=${model.padEnd(22)} ${d.reason}`,
  );
  console.log(`           -> ${label}\n`);
}

async function main() {
  // 1. Planning step — long-form intent, expect STRONG.
  logDecision(
    "user asks the agent to plan a refactor",
    await client.route({
      prompt: "Plan how to refactor the auth middleware into smaller modules.",
      session_id: SESSION,
      tenant_id: TENANT,
      agent_step_type: "planning",
    }),
  );

  // 2. Tool call — read_file is in the safe list, expect WEAK.
  logDecision(
    "agent decides to read auth.py",
    await client.route({
      prompt: "Read the contents of auth.py.",
      session_id: SESSION,
      tenant_id: TENANT,
      agent_step_type: "tool_call",
      planned_tool: "read_file",
      available_tools: ["read_file", "grep", "edit_file"],
    }),
  );

  // 3. Tool result interpretation — small result, expect WEAK.
  logDecision(
    "agent interprets a small read_file result",
    await client.route({
      messages: [
        { role: "user", content: "What does this file do?" },
        {
          role: "assistant",
          tool_calls: [{ name: "read_file", arguments: { path: "auth.py" } }],
        },
        {
          role: "tool",
          name: "read_file",
          content: "def authenticate(req): ...",
        },
      ],
      session_id: SESSION,
      tenant_id: TENANT,
      last_tool_called: "read_file",
    }),
  );

  // 4. Tool call — edit_file is high-stakes, expect STRONG.
  logDecision(
    "agent calls edit_file (requires strong)",
    await client.route({
      prompt: "Edit auth.py to extract the validation logic into a helper.",
      session_id: SESSION,
      tenant_id: TENANT,
      agent_step_type: "tool_call",
      planned_tool: "edit_file",
    }),
  );

  // 5. Simulate the previous edit failing schema validation. The next turn
  //    carries that outcome and the router escalates regardless of tool.
  const failure: Outcome = {
    kind: "tool_schema_error",
    tool_name: "edit_file",
    detail: "argument 'edits' was not a list",
  };
  logDecision(
    "after a tool_schema_error, even a normally-safe read goes STRONG",
    await client.route({
      prompt: "Retry: first read the file again to re-verify line numbers.",
      session_id: SESSION,
      tenant_id: TENANT,
      agent_step_type: "tool_call",
      planned_tool: "read_file",
      recent_outcomes: [failure],
    }),
  );

  // 6. New session, fresh decisions. No carry-over.
  logDecision(
    "different session — short greeting routes weak",
    await client.route({
      prompt: "hi",
      session_id: "demo-tape-002",
      tenant_id: TENANT,
    }),
  );
}

main().catch((e) => {
  console.error("fatal:", e);
  process.exit(1);
});
