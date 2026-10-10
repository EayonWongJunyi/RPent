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

from __future__ import annotations

import json
import os
import queue
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory.manager import MemoryManager
from rpent.planner.base import build_planner
from rpent.planner.check import LlmCheckRequest, check_llm
from rpent.planner.codex_cli import CodexCliPlanner, _cli_command, _CliRecorder
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult, tool
from rpent.utils import logging as logging_module

# A real child process talks to the real loopback MCP bridge. No model service.
FAKE_CLI = r"""
import json, os, signal, subprocess, sys, time
import httpx

prompt = sys.stdin.read()
mode = os.environ.get("RPENT_TEST_SCENARIO", "success")
config = dict(arg.split("=", 1) for i, arg in enumerate(sys.argv[1:]) if sys.argv[i] == "-c")

def emit(event):
    print(json.dumps(event), flush=True)

if mode == "invalid":
    print("not-json", flush=True)
    sys.exit(0)
if mode == "failure":
    emit({"type": "error", "message": "retrying"})
    emit({"type": "turn.failed", "error": {"message": "403 This account only allows Codex official clients"}})
    sys.exit(1)
if mode == "stderr":
    sys.stderr.write("MCP initialization failed\n" + "x" * 200000)
    sys.exit(1)
if mode == "no_completion":
    emit({"type": "item.completed", "item": {"id": "a", "type": "agent_message", "text": "ok"}})
    sys.exit(0)
if mode == "descendant":
    child = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
    with open(os.environ["RPENT_TEST_CHILD_PID"], "w") as f:
        f.write(str(child.pid))
    time.sleep(60)

emit({"type": "thread.started", "thread_id": "test-thread"})
emit({"type": "turn.started"})
if "mcp_servers.rpent.url" in config:
    assert config["mcp_servers.rpent.required"] == "true"
    assert json.loads(config["mcp_servers.rpent.default_tools_approval_mode"]) == "approve"
    assert float(config["mcp_servers.rpent.tool_timeout_sec"]) > 0
    url = json.loads(config["mcp_servers.rpent.url"])
    client = httpx.Client(trust_env=False, timeout=60)
    def rpc(method, params):
        reply = client.post(url, headers={"Accept": "application/json, text/event-stream"}, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
        reply.raise_for_status()
        return reply.json()["result"]
    rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}})
    result = rpc("tools/call", {"name": "slow" if mode == "slow" else "finish", "arguments": {} if mode == "slow" else {"status": "success", "summary": "tool return"}})
    item = {"id": "f", "type": "mcp_tool_call", "server": "rpent", "tool": "finish", "status": "completed", "arguments": {"status": "failure", "summary": "must not be used"}, "result": result}
    emit({"type": "item.started", "item": {**item, "status": "in_progress"}})
    emit({"type": "item.completed", "item": item})
    emit({"type": "item.completed", "item": item})
    client.close()
emit({"type": "error", "message": "concurrency limit; retrying"})
emit({"type": "error", "message": "concurrency limit; retrying"})
emit({"type": "item.completed", "item": {"id": "a", "type": "agent_message", "text": "ok"}})
emit({"type": "item.completed", "item": {"id": "empty", "type": "agent_message", "text": ""}})
emit({"type": "turn.completed", "usage": {"input_tokens": 12, "cached_input_tokens": 5, "output_tokens": 3}})
"""


class SlowToolkit(Toolkit):
    def __init__(self, root: Path) -> None:
        super().__init__(
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(root / "memory"),
            state=EnvState(root),
        )
        self.slow_started = False
        self.slow_stopped = False
        self.add_tool(self.slow)

    @tool(readonly=True)
    def slow(self) -> ToolResult:
        """Wait until cancellation reaches a safe boundary."""
        import time

        self.slow_started = True
        try:
            while True:
                self.raise_if_cancelled()
                time.sleep(0.01)
        finally:
            self.slow_stopped = True


@pytest.fixture(autouse=True)
def scoped_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide template paths without changing the process-wide logger handlers."""
    monkeypatch.setattr(logging_module, "_output_dir", tmp_path)


@pytest.fixture
def cli_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    binary = tmp_path / "fake-codex"
    binary.write_text(f"#!{sys.executable}\n" + FAKE_CLI)
    binary.chmod(0o700)
    monkeypatch.setenv("CODEX_BIN", str(binary))
    monkeypatch.setenv("CODEX_API_KEY", "offline-test-key")
    for name in (
        "CODEX_BASE_URL",
        "CODEX_MODEL_CONTEXT_WINDOW",
        "CODEX_AUTO_COMPACT_TOKEN_LIMIT",
        "RPENT_TEST_SCENARIO",
    ):
        monkeypatch.delenv(name, raising=False)
    return binary


def planner(tmp_path: Path, *, timeout_s: float = 30) -> CodexCliPlanner:
    return CodexCliPlanner(
        output_dir=tmp_path,
        repo_root=tmp_path,
        timeout_s=timeout_s,
        model="gpt-6.1-sol",
        reasoning_effort="low",
        dashboard_events=NullDashboardEventSink(),
    )


def solve(backend: CodexCliPlanner, toolkit: Toolkit):
    return backend.solve(
        system_prompt="Use RPent tools.",
        user_message="finish",
        toolkit=toolkit,
        max_turns=1,
    )


def test_cli_solve_records_real_mcp_finish_and_retry_success(
    tmp_path: Path, cli_binary: Path
) -> None:
    result = solve(planner(tmp_path), SlowToolkit(tmp_path))
    assert result.error is None
    assert result.messages[:2] == [
        {"role": "system", "content": "Use RPent tools."},
        {"role": "user", "content": "finish"},
    ]
    assert result.finish_result == {
        "_finish": True,
        "status": "success",
        "summary": "tool return",
    }
    assert result.stats["backend"] == "codex_cli"
    assert result.stats["tool_calls"] == 1
    assert result.stats["turns_used"] is None
    assert result.stats["max_turns_enforced"] is False
    assert result.stats["total_input_tokens"] == 12
    assert result.stats["total_cached_input_tokens"] == 5
    assert result.stats["total_output_tokens"] == 3
    assert result.messages[-1] == {"role": "assistant", "content": "ok"}
    events = [
        json.loads(line)
        for line in Path(result.stats["raw_stream_path"]).read_text().splitlines()
    ]
    assert sum(e["type"] == "error" for e in events) == 2
    assert Path(result.stats["last_message_path"]).read_text() == "ok"


@pytest.mark.parametrize("mode", ["failure", "invalid", "stderr", "no_completion"])
def test_cli_failures_return_error_and_close_mcp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cli_binary: Path, mode: str
) -> None:
    from rpent.planner import codex_cli
    from rpent.planner.utils.http_mcp_server import HttpMcpServer

    servers = []

    class TrackingServer(HttpMcpServer):
        def __init__(self, toolkit):
            super().__init__(toolkit)
            servers.append(self)

    monkeypatch.setattr(codex_cli, "HttpMcpServer", TrackingServer)
    monkeypatch.setenv("RPENT_TEST_SCENARIO", mode)
    result = solve(planner(tmp_path), SlowToolkit(tmp_path))
    assert result.error
    assert result.finish_result is None
    assert servers[0]._thread is None
    with httpx.Client(trust_env=False) as client, pytest.raises(httpx.ConnectError):
        client.post(servers[0].url, timeout=1)


def test_timeout_waits_for_active_tool_and_stops_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cli_binary: Path
) -> None:
    monkeypatch.setenv("RPENT_TEST_SCENARIO", "slow")
    from rpent.planner import codex_cli

    toolkit = SlowToolkit(tmp_path)
    # Advance only this driver's clock after the real tool starts. Process
    # startup and filesystem latency cannot consume the cancellation window.
    monkeypatch.setattr(
        codex_cli,
        "time",
        SimpleNamespace(monotonic=lambda: 10 if toolkit.slow_started else 0),
    )
    result = solve(planner(tmp_path, timeout_s=3), toolkit)
    assert "timed out" in result.error
    assert toolkit.slow_started and toolkit.slow_stopped
    assert toolkit._active_operation is None


def test_timeout_terminates_owned_descendants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cli_binary: Path
) -> None:
    pid_file = tmp_path / "child.pid"
    monkeypatch.setenv("RPENT_TEST_SCENARIO", "descendant")
    monkeypatch.setenv("RPENT_TEST_CHILD_PID", str(pid_file))
    from rpent.planner import codex_cli

    monkeypatch.setattr(
        codex_cli,
        "time",
        SimpleNamespace(monotonic=lambda: 10 if pid_file.exists() else 0),
    )
    result = solve(planner(tmp_path, timeout_s=3), SlowToolkit(tmp_path))
    assert "timed out" in result.error
    pid = int(pid_file.read_text())
    # Orphan reaping is controlled by the host; a zombie cannot execute work.
    stat = Path(f"/proc/{pid}/stat")
    assert not stat.exists() or stat.read_text().split()[2] == "Z"


@pytest.mark.parametrize(
    "server,status,error,result",
    [
        (
            "other",
            "completed",
            None,
            {"content": [{"type": "text", "text": '{"_finish": true}'}]},
        ),
        (
            "rpent",
            "failed",
            None,
            {"content": [{"type": "text", "text": '{"_finish": true}'}]},
        ),
        ("rpent", "completed", {"message": "failed"}, {"content": []}),
        (
            "rpent",
            "completed",
            None,
            {
                "isError": True,
                "content": [{"type": "text", "text": '{"_finish": true}'}],
            },
        ),
        ("rpent", "completed", None, None),
        (
            "rpent",
            "completed",
            None,
            {"content": [{"type": "text", "text": '{"status": "success"}'}]},
        ),
    ],
)
def test_finish_requires_successful_rpent_tool_return(
    server: str, status: str, error: Any, result: Any
) -> None:
    recorder = _CliRecorder()
    recorder.observe(
        {
            "type": "item.completed",
            "item": {
                "id": "f",
                "type": "mcp_tool_call",
                "server": server,
                "tool": "finish",
                "status": status,
                "error": error,
                "result": result,
                "arguments": {"_finish": True, "status": "success"},
            },
        }
    )
    assert recorder.finish_result is None


def test_cli_probe_uses_cli_and_classifies_terminal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cli_binary: Path
) -> None:
    result = check_llm(
        LlmCheckRequest(
            planner="codex", codex_driver="cli", model="gpt-6.1-sol", timeout_s=10
        )
    )
    assert result.ok and result.reply == "ok"
    assert result.as_dict()["codex_driver"] == "cli"
    monkeypatch.setenv("RPENT_TEST_SCENARIO", "failure")
    result = check_llm(
        LlmCheckRequest(planner="codex", codex_driver="cli", timeout_s=10)
    )
    assert result.status == "auth_failed"
    assert "only allows Codex official clients" in result.detail


def test_cli_command_preserves_endpoint_login_model_and_required_mcp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_BIN", "/home/junyi/bin/codex")
    monkeypatch.setenv("CODEX_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("CODEX_API_KEY", "test-key")
    monkeypatch.setenv("CODEX_SERVICE_TIER", "fast")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid")
    monkeypatch.setenv("NO_PROXY", "existing.invalid")
    command, env = _cli_command(
        cwd=str(tmp_path),
        model="gpt-6.1-sol",
        reasoning_effort="low",
        mcp_url="http://127.0.0.1:1234/mcp/",
    )
    assert command[0] == "/home/junyi/bin/codex"
    assert command[-3:] == ["--model", "gpt-6.1-sol", "-"]
    assert "mcp_servers.rpent.required=true" in command
    assert 'mcp_servers.rpent.default_tools_approval_mode="approve"' in command
    assert (
        'model_providers.rpent_proxy.base_url="https://example.invalid/v1"' in command
    )
    assert 'service_tier="fast"' in command
    assert "features.multi_agent=false" in command
    assert env["RPENT_CODEX_PROVIDER_KEY"] == "test-key"
    assert "127.0.0.1" in env["NO_PROXY"]
    assert os.environ["NO_PROXY"] == "existing.invalid"


def test_cli_factory_and_interaction_rejection(tmp_path: Path) -> None:
    backend = build_planner(
        "codex",
        codex_driver="cli",
        output_dir=tmp_path,
        recipe_tag="test",
        robot_name="libero",
        dashboard_events=NullDashboardEventSink(),
    )
    assert isinstance(backend, CodexCliPlanner)
    with pytest.raises(ValueError, match="non-interactive"):
        backend.solve(
            system_prompt="",
            user_message="",
            toolkit=None,
            max_turns=1,
            input_queue=queue.Queue(),
        )
    with pytest.raises(ValueError, match="requires"):
        build_planner(
            "api",
            codex_driver="cli",
            output_dir=tmp_path,
            recipe_tag="test",
            robot_name="libero",
            dashboard_events=NullDashboardEventSink(),
        )


def test_keyboard_interrupt_reaps_cli_and_stops_mcp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cli_binary: Path,
) -> None:
    from rpent.planner import codex_cli
    from rpent.planner.utils.http_mcp_server import HttpMcpServer

    processes = []
    servers = []
    real_popen = codex_cli.subprocess.Popen
    real_observe = _CliRecorder.observe

    def popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    def observe(self, event):
        if event.get("type") == "thread.started":
            raise KeyboardInterrupt
        return real_observe(self, event)

    class TrackingServer(HttpMcpServer):
        def __init__(self, toolkit):
            super().__init__(toolkit)
            servers.append(self)

    monkeypatch.setattr(codex_cli.subprocess, "Popen", popen)
    monkeypatch.setattr(codex_cli, "HttpMcpServer", TrackingServer)
    monkeypatch.setattr(_CliRecorder, "observe", observe)
    with pytest.raises(KeyboardInterrupt):
        solve(planner(tmp_path), SlowToolkit(tmp_path))
    assert processes[0].poll() is not None
    assert servers[0]._thread is None


def test_mcp_startup_failure_is_recorded_and_stopped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rpent.planner import codex_cli

    stopped = []

    class FailingServer:
        def __init__(self, toolkit):
            pass

        def start(self, **kwargs):
            raise RuntimeError("MCP startup failed")

        def cancel_active_and_wait(self):
            pass

        def stop(self):
            stopped.append(True)

    monkeypatch.setattr(codex_cli, "HttpMcpServer", FailingServer)
    result = solve(planner(tmp_path), SlowToolkit(tmp_path))
    assert result.error == "RuntimeError: MCP startup failed"
    assert stopped == [True]
