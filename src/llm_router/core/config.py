"""Router configuration. YAML-loadable, schema-validated.

Configuration is split by concern so individual layers can be tuned
without re-deploying the world.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from llm_router.core.decision import Tier


class ClassifierConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True

    embedding_model: str = "intfloat/multilingual-e5-small"
    """HuggingFace model id used by sentence-transformers. Changing this
    requires re-training the classifier head."""

    artifact_path: str | None = None
    """Path to the trained sklearn classifier pickle. If None, the
    classifier layer is disabled at runtime even if `enabled=True`."""

    p_low: float = Field(default=0.30, ge=0.0, le=1.0)
    """Below this `p(strong)`, route to weak. Tune from eval data."""

    p_high: float = Field(default=0.70, ge=0.0, le=1.0)
    """Above this `p(strong)`, route to strong. Between low and high
    is the uncertainty band → fail open to strong."""

    timeout_ms: int = 200
    """If embedding/predict takes longer than this, skip Layer 2 and
    fall through to default (strong)."""


class RuleEngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    enabled_rule_names: list[str] | None = None
    """If set, only these rule names will run. None = all builtin rules."""


class StickinessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    upgrade_only: bool = True
    """If True, once a session lands on a higher tier it stays there.
    The recommended default. Setting False allows downgrades and is
    almost never what you want."""

    ttl_seconds: int = 3600


class TenantPolicyEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forced_tier: Tier | None = None
    """If set, this tenant ALWAYS gets routed to this tier. Short-circuits
    the entire pipeline."""

    blocked_tiers: list[Tier] = Field(default_factory=list)
    """Tiers this tenant must never land on (e.g. for compliance)."""

    latency_strict: bool = False
    """If True, skip Layer 2 (classifier) for this tenant to save the
    embedding latency. Used by tenants with hard SLAs."""

    classifier_p_high_override: float | None = Field(default=None, ge=0.0, le=1.0)
    """Per-tenant override of the upper threshold. Lower threshold can
    be added the same way when needed."""


class TenantConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_policy: TenantPolicyEntry = Field(default_factory=TenantPolicyEntry)
    overrides: dict[str, TenantPolicyEntry] = Field(default_factory=dict)


class GatewayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str = "noop"
    """One of: noop, litellm. Noop = router emits decisions only and
    never invokes a model. Use for embedding deployments where the
    caller invokes the model itself."""

    tier_to_model: dict[Tier, str] = Field(default_factory=dict)
    """Maps logical tiers to concrete provider/model strings. e.g.
    `{Tier.WEAK: "claude-haiku-4-5", Tier.STRONG: "claude-opus-4-7"}`.
    The router does not parse these; the gateway adapter does."""


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"
    json_format: bool = True
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    """Per-decision log sampling. 1.0 = log every decision."""


class RouterConfig(BaseModel):
    """Top-level config object. Treat as immutable post-load."""

    model_config = ConfigDict(extra="forbid")

    default_tier: Tier = Tier.STRONG
    """Tier to use when no layer fires. Strong by default — never
    silently downgrade."""

    rules: RuleEngineConfig = Field(default_factory=RuleEngineConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    stickiness: StickinessConfig = Field(default_factory=StickinessConfig)
    tenants: TenantConfig = Field(default_factory=TenantConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, path: str | Path) -> RouterConfig:
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    @classmethod
    def default(cls) -> RouterConfig:
        return cls()
