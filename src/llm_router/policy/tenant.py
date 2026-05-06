"""Per-tenant policy resolution.

Tenants are looked up by `tenant_id`. Unknown tenants get the default
policy. Resolution is O(1) and runs on every request — keep it cheap.
"""

from __future__ import annotations

from llm_router.core.config import TenantConfig, TenantPolicyEntry


class TenantPolicyResolver:
    def __init__(self, config: TenantConfig) -> None:
        self._default = config.default_policy
        self._overrides = config.overrides

    def resolve(self, tenant_id: str | None) -> TenantPolicyEntry:
        if tenant_id is None:
            return self._default
        return self._overrides.get(tenant_id, self._default)
