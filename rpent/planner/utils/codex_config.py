# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Codex configuration shared by the SDK and CLI drivers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROVIDER_ID = "rpent_proxy"
PROVIDER_ENV_KEY = "RPENT_CODEX_PROVIDER_KEY"
_LOOPBACK_NO_PROXY_HOSTS = ("127.0.0.1", "localhost")


def _codex_environment() -> dict[str, str]:
    """Build the Codex child environment with direct access to its local MCP."""
    env = {**os.environ}
    entries: list[str] = []
    for key in ("NO_PROXY", "no_proxy"):
        entries.extend(item.strip() for item in env.get(key, "").split(","))
    entries = list(
        dict.fromkeys(item for item in (*entries, *_LOOPBACK_NO_PROXY_HOSTS) if item)
    )
    bypass = ",".join(entries)
    env["NO_PROXY"] = bypass
    env["no_proxy"] = bypass
    return env


def _codex_mcp_config_overrides(
    *,
    mcp_url: str | None,
    base_url: str | None,
) -> list[str]:
    config: list[tuple[str, Any]] = []
    if mcp_url:
        config.append(("mcp_servers.rpent.url", mcp_url))
    if base_url:
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized = normalized + "/v1"
        config.extend(
            [
                ("model_provider", PROVIDER_ID),
                (f"model_providers.{PROVIDER_ID}.name", PROVIDER_ID),
                (f"model_providers.{PROVIDER_ID}.base_url", normalized),
                (f"model_providers.{PROVIDER_ID}.wire_api", "responses"),
                (f"model_providers.{PROVIDER_ID}.env_key", PROVIDER_ENV_KEY),
            ]
        )
    model_context_window = os.environ.get("CODEX_MODEL_CONTEXT_WINDOW", None)
    if model_context_window is not None:
        config.append(("model_context_window", int(model_context_window)))
    auto_compact_token_limit = os.environ.get("CODEX_AUTO_COMPACT_TOKEN_LIMIT", None)
    if auto_compact_token_limit is not None:
        config.append(("model_auto_compact_token_limit", int(auto_compact_token_limit)))
    return [f"{key}={json.dumps(value)}" for key, value in config]


def codex_config_overrides(
    *, mcp_url: str | None, base_url: str | None, cwd: str
) -> list[str]:
    """Build endpoint, MCP, and project-context overrides for either driver."""
    config_overrides = _codex_mcp_config_overrides(mcp_url=mcp_url, base_url=base_url)
    # Project development instructions and skills are excluded from planner context.
    config_overrides.append("project_doc_max_bytes=0")
    disabled_skills = [
        f"{{ path = {json.dumps(str(path.resolve()), ensure_ascii=False)}, enabled = false }}"
        for path in sorted((Path(cwd) / ".agents" / "skills").glob("*/SKILL.md"))
        if path.is_file()
    ]
    if disabled_skills:
        config_overrides.append(f"skills.config=[{', '.join(disabled_skills)}]")
    return config_overrides
