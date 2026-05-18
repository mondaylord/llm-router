/**
 * Minimal TypeScript client for llm-router's HTTP API.
 *
 * Intended for embedding inside agent frameworks (e.g. reel-agent) running
 * on Node / Bun. Zero runtime dependencies beyond `fetch`.
 *
 * Wire-compatible with the FastAPI service in `src/llm_router/server/app.py`.
 *
 * Example:
 *
 *   const client = new LlmRouterClient({ baseUrl: "http://localhost:8000" });
 *
 *   const d = await client.route({
 *     prompt: "Plan how to refactor auth middleware.",
 *     session_id: "tape-123",
 *     agent_step_type: "planning",
 *   });
 *   const model = MODEL_FOR_TIER[d.tier];
 *   // ... invoke `model` via your provider SDK
 *
 *   // Next turn: report outcome from prior step so cascade can fire.
 *   const d2 = await client.route({
 *     prompt: "Try editing auth.py again.",
 *     session_id: "tape-123",
 *     agent_step_type: "tool_call",
 *     planned_tool: "edit_file",
 *     recent_outcomes: [{ kind: "tool_schema_error", tool_name: "edit_file" }],
 *   });
 */

export type Tier = "weak" | "mid" | "strong";

export type DecisionLayer =
  | "tenant_override"
  | "session_stickiness"
  | "rule"
  | "classifier"
  | "default";

export type AgentStepType =
  | "chat"
  | "planning"
  | "tool_call"
  | "tool_result"
  | "edit"
  | "summarize";

export type OutcomeKind =
  | "tool_schema_error"
  | "tool_execution_error"
  | "parse_error"
  | "validation_error"
  | "user_negative_feedback"
  | "retry_attempt"
  | "generic_failure";

export interface Outcome {
  kind: OutcomeKind;
  detail?: string;
  tool_name?: string;
}

export interface ToolCall {
  id?: string;
  name: string;
  arguments?: Record<string, unknown> | string;
}

export interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  name?: string;
}

export interface RouteRequest {
  /** Either `prompt` or `messages` must be set. */
  prompt?: string;
  messages?: Message[];

  /** Conversation / tape id for stickiness. */
  session_id?: string;

  /** Tenant key for per-customer policy. */
  tenant_id?: string;

  /** Explicit step label; auto-detected if omitted. */
  agent_step_type?: AgentStepType;

  /** Tools the agent may call this turn. */
  available_tools?: string[];

  /** Tool the agent intends to call (if already decided). */
  planned_tool?: string;

  /** Tool that produced the most recent tool message. */
  last_tool_called?: string;

  /** Failure / quality signals from prior turns. Drives cascade. */
  recent_outcomes?: Outcome[];

  /** Caller-known total assembled context size (for long-context rule). */
  total_context_tokens?: number;

  history_turns?: number;
  metadata?: Record<string, unknown>;
}

export interface RouteResponse {
  tier: Tier;
  layer: DecisionLayer;
  reason: string;
  confidence: number;
  classifier_score?: number | null;
  rules_evaluated: string[];
  elapsed_ms: number;
  inferred_step_type?: AgentStepType | null;
}

export interface HealthResponse {
  status: string;
  classifier_loaded: boolean;
  rules_count: number;
}

export interface LlmRouterClientOptions {
  baseUrl: string;
  /** Per-request timeout in ms. Default 1500. The router itself adds <100ms;
   *  this only kicks in for network / cold-start cases. */
  timeoutMs?: number;
  /** If the router fails (network, 5xx, timeout), which tier to use as
   *  safe fallback. Default "strong" — never silently downgrade. */
  fallbackTier?: Tier;
  /** Optional extra headers (e.g. auth). */
  headers?: Record<string, string>;
  /** Custom fetch impl, for tests. */
  fetchImpl?: typeof fetch;
}

export class LlmRouterError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message);
    this.name = "LlmRouterError";
  }
}

export class LlmRouterClient {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly fallbackTier: Tier;
  private readonly headers: Record<string, string>;
  private readonly fetchImpl: typeof fetch;

  constructor(opts: LlmRouterClientOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/+$/, "");
    this.timeoutMs = opts.timeoutMs ?? 1500;
    this.fallbackTier = opts.fallbackTier ?? "strong";
    this.headers = {
      "content-type": "application/json",
      ...(opts.headers ?? {}),
    };
    this.fetchImpl = opts.fetchImpl ?? fetch;
  }

  async health(): Promise<HealthResponse> {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), this.timeoutMs);
    try {
      const res = await this.fetchImpl(`${this.baseUrl}/healthz`, {
        signal: ctl.signal,
        headers: this.headers,
      });
      if (!res.ok) {
        throw new LlmRouterError(`healthz ${res.status}`, res.status);
      }
      return (await res.json()) as HealthResponse;
    } finally {
      clearTimeout(t);
    }
  }

  /**
   * Make a routing decision. On any error (network, 5xx, timeout), returns
   * a synthetic decision pinned to `fallbackTier` so the agent loop keeps
   * moving. The error is also exposed via `decision.reason` for telemetry.
   *
   * Set `throwOnError: true` if you'd rather handle failures yourself.
   */
  async route(
    req: RouteRequest,
    opts: { throwOnError?: boolean } = {},
  ): Promise<RouteResponse> {
    if (!req.prompt && !(req.messages && req.messages.length > 0)) {
      throw new LlmRouterError("either `prompt` or `messages` must be set");
    }

    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), this.timeoutMs);
    try {
      const res = await this.fetchImpl(`${this.baseUrl}/route`, {
        method: "POST",
        headers: this.headers,
        body: JSON.stringify(req),
        signal: ctl.signal,
      });
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new LlmRouterError(
          `route ${res.status}: ${text.slice(0, 200)}`,
          res.status,
        );
      }
      return (await res.json()) as RouteResponse;
    } catch (e) {
      if (opts.throwOnError) throw e;
      const err = e instanceof Error ? e.message : String(e);
      return {
        tier: this.fallbackTier,
        layer: "default",
        reason: `client_fallback:${err}`,
        confidence: 1.0,
        rules_evaluated: [],
        elapsed_ms: 0,
        inferred_step_type: null,
      };
    } finally {
      clearTimeout(t);
    }
  }
}

/**
 * Recommended tier → model mapping for reel-agent.
 * Override per deployment.
 */
export const DEFAULT_MODEL_FOR_TIER: Record<Tier, string> = {
  weak: "claude-haiku-4-5",
  mid: "claude-sonnet-4-6",
  strong: "claude-opus-4-7",
};
