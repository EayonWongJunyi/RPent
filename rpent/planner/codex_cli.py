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

"""Non-interactive Codex CLI planner using JSONL events and RPent HTTP MCP."""

from __future__ import annotations

import json
import os
import queue
import selectors
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rpent.dashboard.events import DashboardEventSink
from rpent.dashboard.interaction import DashboardInteractionPort
from rpent.planner.base import (
    REASONING_EFFORTS,
    Planner,
    PlannerResult,
    strip_mcp_prefix,
)
from rpent.planner.utils.codex_config import (
    PROVIDER_ENV_KEY,
    _codex_environment,
    codex_config_overrides,
)
from rpent.planner.utils.http_mcp_server import HttpMcpServer
from rpent.tools.toolkit import Toolkit
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger

logger = get_logger("codex_cli")


def _cli_command(
    *,
    cwd: str,
    model: str | None,
    reasoning_effort: str,
    mcp_url: str | None = None,
    tool_timeout_s: float | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Resolve the CLI executable and shared endpoint/context configuration."""
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"unsupported reasoning effort: {reasoning_effort}")
    env = _codex_environment()
    if api_key := env.get("CODEX_API_KEY"):
        env[PROVIDER_ENV_KEY] = api_key
    command = [
        env.get("CODEX_BIN") or "codex",
        "exec",
        "--json",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--color",
        "never",
    ]
    overrides = codex_config_overrides(
        mcp_url=mcp_url,
        base_url=env.get("CODEX_BASE_URL"),
        cwd=cwd,
    )
    overrides.extend(
        [
            'approval_policy="never"',
            "features.multi_agent=false",
            f"model_reasoning_effort={json.dumps(reasoning_effort)}",
        ]
    )
    if mcp_url:
        overrides.extend(
            [
                "mcp_servers.rpent.enabled=true",
                "mcp_servers.rpent.required=true",
                'mcp_servers.rpent.default_tools_approval_mode="approve"',
            ]
        )
    if mcp_url and tool_timeout_s is not None:
        overrides.append(f"mcp_servers.rpent.tool_timeout_sec={max(1, tool_timeout_s)}")
    if service_tier := env.get("CODEX_SERVICE_TIER"):
        overrides.append(f"service_tier={json.dumps(service_tier)}")
    for override in overrides:
        command.extend(["-c", override])
    if model:
        command.extend(["--model", model])
    command.append("-")
    return command, env


@dataclass
class _CliRecorder:
    """Record CLI items; exec turns do not measure individual model responses."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0
    cli_turns_completed: int = 0
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "total_input_tokens": 0,
            "total_cached_input_tokens": 0,
            "total_output_tokens": 0,
            "total_reasoning_output_tokens": 0,
        }
    )
    final_response: str = ""
    finish_result: dict[str, Any] | None = None
    terminal_error: str | None = None
    last_error: str | None = None
    completed: bool = False
    _seen_items: set[str] = field(default_factory=set)

    def observe(self, event: dict[str, Any]) -> str:
        kind = event.get("type")
        if kind == "error":
            self.last_error = str(event.get("message") or event)
            return f"[codex-retry] {self.last_error}\n"
        if kind == "turn.failed":
            error = event.get("error") or {}
            self.terminal_error = (
                str(error.get("message", error))
                if isinstance(error, dict)
                else str(error)
            )
            return f"[codex-failed] {self.terminal_error}\n"
        if kind == "turn.completed":
            self.completed = True
            self.cli_turns_completed += 1
            usage = event.get("usage") or {}
            for source, target in (
                ("input_tokens", "total_input_tokens"),
                ("cached_input_tokens", "total_cached_input_tokens"),
                ("output_tokens", "total_output_tokens"),
                ("reasoning_output_tokens", "total_reasoning_output_tokens"),
            ):
                value = usage.get(source, 0)
                if type(value) is not int or value < 0:
                    raise ValueError(f"invalid Codex CLI usage: {source}={value!r}")
                self.usage[target] += value
            return f"[codex-result] completed {json.dumps(usage)}\n"
        if kind != "item.completed":
            return ""
        item = event.get("item")
        if not isinstance(item, dict):
            raise ValueError("Codex CLI item.completed has no item object")
        item_id = item.get("id")
        if item_id in self._seen_items:
            return ""
        if isinstance(item_id, str):
            self._seen_items.add(item_id)
        item_type = item.get("type")
        if item_type == "agent_message":
            text = str(item.get("text") or "")
            if not text.strip():
                return ""
            self.final_response = text
            self.messages.append({"role": "assistant", "content": self.final_response})
            return f"[codex] {self.final_response}\n"
        if item_type == "reasoning":
            text = str(item.get("text") or "")
            self.messages.append({"role": "reasoning", "content": text})
            return f"[codex-reasoning] {text}\n"
        if item_type in {
            "mcp_tool_call",
            "command_execution",
            "file_change",
            "web_search",
        }:
            self.tool_calls += 1
            self.messages.append({"role": "tool", "content": item})
            if item_type == "mcp_tool_call":
                self._capture_finish(item)
            return f"[tool<-] {json.dumps(item, ensure_ascii=False)}\n"
        return ""

    def _capture_finish(self, item: dict[str, Any]) -> None:
        if self.finish_result is not None or item.get("server") != "rpent":
            return
        if strip_mcp_prefix(str(item.get("tool", ""))) != "finish":
            return
        if item.get("status") != "completed" or item.get("error"):
            return
        result = item.get("result")
        if not isinstance(result, dict) or result.get("isError") is True:
            return
        for block in result.get("content", []):
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            try:
                payload = json.loads(block.get("text", ""))
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict) and payload.get("_finish") is True:
                self.finish_result = payload
                return

    def stats(self) -> dict[str, Any]:
        return {
            "turns_used": None,
            "cli_turns_completed": self.cli_turns_completed,
            "max_turns_enforced": False,
            "tool_calls": self.tool_calls,
            **self.usage,
        }


def _stop_process(process: subprocess.Popen) -> None:
    """Reap the owned CLI and stop descendants in its process group."""

    def send(sig: int) -> None:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    send(signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        send(signal.SIGKILL)
        process.wait(timeout=5)


def _run_cli(
    *,
    prompt: str,
    cwd: str,
    model: str | None,
    reasoning_effort: str,
    timeout_s: float,
    recorder: _CliRecorder,
    mcp_url: str | None = None,
    emit: Callable[[str, str], None] | None = None,
) -> None:
    """Drain both pipes, parse JSONL, and always reap the launched process."""
    if timeout_s <= 0:
        raise TimeoutError("Codex CLI timed out before launch")
    command, env = _cli_command(
        cwd=cwd,
        model=model,
        reasoning_effort=reasoning_effort,
        mcp_url=mcp_url,
        tool_timeout_s=timeout_s,
    )
    deadline = time.monotonic() + timeout_s
    stderr_tail = ""
    with tempfile.TemporaryFile() as prompt_file:
        prompt_file.write(prompt.encode("utf-8"))
        prompt_file.seek(0)
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=prompt_file,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, "stdout")
                selector.register(process.stderr, selectors.EVENT_READ, "stderr")
                pending = b""

                def consume(line: bytes) -> None:
                    if not line.strip():
                        return
                    raw = line.decode("utf-8")
                    if emit:
                        emit("raw", raw + "\n")
                    try:
                        event = json.loads(raw)
                    except ValueError as exc:
                        raise RuntimeError("invalid JSONL from Codex CLI") from exc
                    if not isinstance(event, dict):
                        raise RuntimeError("Codex CLI event must be a JSON object")
                    rendered = recorder.observe(event)
                    if emit and rendered:
                        emit("text", rendered)

                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"Codex CLI timed out after {timeout_s:g}s")
                    for key, _ in selector.select(timeout=min(remaining, 0.2)):
                        data = os.read(key.fd, 65536)
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        if key.data == "stderr":
                            text = data.decode("utf-8", errors="replace")
                            stderr_tail = (stderr_tail + text)[-8000:]
                            if emit:
                                emit("stderr", text)
                        else:
                            pending += data
                            while b"\n" in pending:
                                line, pending = pending.split(b"\n", 1)
                                consume(line)
                if pending:
                    consume(pending)
            try:
                code = process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f"Codex CLI timed out after {timeout_s:g}s") from exc
            if recorder.terminal_error:
                raise RuntimeError(recorder.terminal_error)
            if code != 0 or not recorder.completed:
                detail = (
                    stderr_tail.strip()
                    or recorder.last_error
                    or "no turn.completed event"
                )
                raise RuntimeError(f"Codex CLI exited with code {code}: {detail}")
        finally:
            _stop_process(process)
            process.stdout.close()
            process.stderr.close()


class CodexCliPlanner(Planner):
    """Run one task through the official CLI; model-response budgets are unavailable."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        dashboard_events: DashboardEventSink,
        repo_root: str | Path | None = None,
        timeout_s: int = 600,
        extra_dirs: list[str] | None = None,
        output_path: str | Path | None = None,
        model: str | None = None,
        reasoning_effort: str = "none",
    ) -> None:
        """Configure the non-interactive driver using the current CLI login."""
        if reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"unsupported reasoning effort: {reasoning_effort}")
        if dashboard_events.enabled:
            raise ValueError("the Codex CLI driver does not support Dashboard")
        self._repo_root = str(repo_root or get_repo_root())
        self._output_path = Path(output_path or Path(output_dir) / "codex_cli.txt")
        self._timeout_s = timeout_s
        self._model = model or os.environ.get("CODEX_MODEL")
        self._reasoning_effort = reasoning_effort
        # Toolkit owns file and artifact access; the CLI shell stays read-only.

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
        dashboard_interaction: DashboardInteractionPort | None = None,
    ) -> PlannerResult:
        """Run to CLI completion or timeout, preserving only returned finish evidence."""
        if input_queue is not None or dashboard_interaction is not None:
            raise ValueError("the Codex CLI driver supports non-interactive tasks only")
        if max_turns < 1 or self._timeout_s <= 0:
            raise ValueError("max_turns and timeout_s must be positive")
        logger.warning(
            "Codex CLI does not enforce max_turns=%d; timeout_s=%s limits the run",
            max_turns,
            self._timeout_s,
        )
        output = self._output_path
        output.parent.mkdir(parents=True, exist_ok=True)
        raw = output.with_suffix(output.suffix + ".stream.jsonl")
        last = output.with_suffix(output.suffix + ".last")
        stderr = output.with_suffix(output.suffix + ".stderr")
        recorder = _CliRecorder()
        if system_prompt:
            recorder.messages.append({"role": "system", "content": system_prompt})
        recorder.messages.append({"role": "user", "content": user_message})
        prompt = f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
        started = time.monotonic()
        server = HttpMcpServer(toolkit)
        error = None
        with (
            output.open("w") as out_f,
            raw.open("w") as raw_f,
            stderr.open("w") as err_f,
        ):

            def emit(kind: str, text: str) -> None:
                target = {"raw": raw_f, "text": out_f, "stderr": err_f}[kind]
                target.write(text)
                target.flush()

            try:
                mcp_url = server.start(ready_timeout_s=min(self._timeout_s, 30))
                _run_cli(
                    prompt=prompt,
                    cwd=self._repo_root,
                    model=self._model,
                    reasoning_effort=self._reasoning_effort,
                    timeout_s=max(0, self._timeout_s - (time.monotonic() - started)),
                    recorder=recorder,
                    mcp_url=mcp_url,
                    emit=emit,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                emit("text", f"[codex-planner] {error}\n")
                logger.warning(error)
            finally:
                try:
                    server.cancel_active_and_wait()
                except Exception as exc:
                    cleanup_error = f"Codex CLI toolkit cancellation failed: {type(exc).__name__}: {exc}"
                    logger.warning(cleanup_error)
                    error = error or cleanup_error
                finally:
                    server.stop()
        last.write_text(recorder.final_response)
        return PlannerResult(
            finish_result=recorder.finish_result,
            messages=recorder.messages,
            stats={
                "backend": "codex_cli",
                "elapsed_s": round(time.monotonic() - started, 1),
                "output_path": str(output),
                "raw_stream_path": str(raw),
                "last_message_path": str(last),
                "stderr_path": str(stderr),
                "output_chars": len(output.read_text()),
                "last_message_chars": len(recorder.final_response),
                **recorder.stats(),
            },
            error=error,
        )


def run_cli_probe(*, prompt: str, model: str | None, timeout_s: int) -> str:
    """Check the same CLI transport without starting an RPent MCP server."""
    recorder = _CliRecorder()
    _run_cli(
        prompt=prompt,
        cwd=str(get_repo_root()),
        model=model,
        reasoning_effort="low",
        timeout_s=timeout_s,
        recorder=recorder,
    )
    return recorder.final_response
