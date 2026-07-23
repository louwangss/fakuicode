"""Fault-isolated execution for validated lifecycle Hook rules."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import RLock, Thread
import time

import httpx

from fakuicode.hooks.models import (
    AgentAction,
    CommandAction,
    HookEvent,
    HookRule,
    HttpAction,
    PromptAction,
)
from fakuicode.hooks.pointers import parse_pointer, resolve_pointer


@dataclass(frozen=True)
class HookDiagnostic:
    hook: str
    source: str
    event: str
    action: str
    category: str
    duration_seconds: float | None = None
    background: bool = False
    status: int | None = None


@dataclass(frozen=True)
class HookDispatchResult:
    denied_reason: str | None = None
    prompts: tuple[str, ...] = ()
    diagnostics: tuple[HookDiagnostic, ...] = ()


@dataclass(frozen=True)
class _ActionResult:
    decision: str | None = None
    reason: str | None = None
    prompt: str | None = None
    diagnostic: HookDiagnostic | None = None


DiagnosticSink = Callable[[HookDiagnostic], None]
_MAX_HTTP_RESPONSE_BYTES = 32 * 1024


class HookEngine:
    """Run matching rules in declaration order while containing every Hook fault."""

    def __init__(
        self,
        rules: tuple[HookRule, ...],
        *,
        diagnostic_sink: DiagnosticSink | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.rules = rules
        self.diagnostic_sink = diagnostic_sink
        self.workspace = (workspace or Path.cwd()).resolve()
        self._once: set[tuple[str, str, str]] = set()
        self._pending_prompts: list[str] = []
        self._scoped_prompts: dict[tuple[str, str, str], tuple[HookEvent, str]] = {}
        self._lock = RLock()

    def dispatch(
        self,
        event: HookEvent,
        payload: Mapping[str, object],
        *,
        plan_mode: bool = False,
    ) -> HookDispatchResult:
        envelope: dict[str, object] = {"event": event.value, **payload}
        prompts: list[str] = []
        pending_prompts: list[str] = []
        diagnostics: list[HookDiagnostic] = []
        denied_reason: str | None = None
        self._end_prompt_scope(event)
        for rule in self.rules:
            if rule.event is not event or (rule.condition is not None and not rule.condition.matches(envelope)):
                continue
            if plan_mode and not isinstance(rule.action, PromptAction):
                continue
            identity = (rule.source.value, rule.name, rule.event.value)
            with self._lock:
                if getattr(rule.action, "once", False) and identity in self._once:
                    continue
                if getattr(rule.action, "once", False):
                    self._once.add(identity)
            if getattr(rule.action, "async_", False):
                Thread(
                    target=self._run_background,
                    args=(rule, envelope),
                    name=f"fakuicode-hook-{rule.name}",
                    daemon=True,
                ).start()
                continue
            result = self._run(rule, envelope)
            if result.prompt:
                prompts.append(result.prompt)
                if event in {
                    HookEvent.APP_START,
                    HookEvent.SESSION_START,
                    HookEvent.TURN_START,
                }:
                    with self._lock:
                        self._scoped_prompts[identity] = (event, result.prompt)
                else:
                    pending_prompts.append(result.prompt)
            if result.decision == "deny" and denied_reason is None:
                denied_reason = result.reason or "Hook 策略拒绝了该操作。"
            if result.diagnostic is not None:
                diagnostics.append(result.diagnostic)
                self._report(result.diagnostic)
        if pending_prompts:
            with self._lock:
                self._pending_prompts.extend(pending_prompts)
        return HookDispatchResult(denied_reason, tuple(prompts), tuple(diagnostics))

    def consume_prompts(self) -> tuple[str, ...]:
        with self._lock:
            prompts = (
                *(content for _, content in self._scoped_prompts.values()),
                *self._pending_prompts,
            )
            self._pending_prompts.clear()
        return prompts

    def peek_prompts(self) -> tuple[str, ...]:
        """Inspect current injections for sizing/internal preparation without consuming them."""
        with self._lock:
            return (
                *(content for _, content in self._scoped_prompts.values()),
                *self._pending_prompts,
            )

    def consume_pending_prompts(self) -> tuple[str, ...]:
        """Consume only one-shot injections created after a request was first built."""
        with self._lock:
            prompts = tuple(self._pending_prompts)
            self._pending_prompts.clear()
        return prompts

    def replace_rules(self, rules: tuple[HookRule, ...]) -> None:
        """Atomically replace declarations without resetting process-local once markers."""
        with self._lock:
            self.rules = rules
            active = {(rule.source.value, rule.name, rule.event.value) for rule in rules}
            self._scoped_prompts = {
                identity: value
                for identity, value in self._scoped_prompts.items()
                if identity in active
            }

    def _end_prompt_scope(self, event: HookEvent) -> None:
        ended = {
            HookEvent.TURN_END: HookEvent.TURN_START,
            HookEvent.SESSION_END: HookEvent.SESSION_START,
            HookEvent.APP_STOP: HookEvent.APP_START,
        }.get(event)
        if ended is None:
            return
        with self._lock:
            self._scoped_prompts = {
                identity: value
                for identity, value in self._scoped_prompts.items()
                if value[0] is not ended
            }

    def _run_background(self, rule: HookRule, payload: Mapping[str, object]) -> None:
        result = self._run(rule, payload, background=True)
        if result.diagnostic is not None:
            self._report(result.diagnostic)

    def _run(
        self, rule: HookRule, payload: Mapping[str, object], *, background: bool = False
    ) -> _ActionResult:
        started = time.monotonic()
        try:
            if isinstance(rule.action, PromptAction):
                return _ActionResult(prompt=rule.action.content)
            if isinstance(rule.action, CommandAction):
                return self._run_command(rule, payload, started, background)
            if isinstance(rule.action, HttpAction):
                return self._run_http(rule, payload, started, background)
            if isinstance(rule.action, AgentAction):
                return _ActionResult(
                    diagnostic=self._diagnostic(
                        rule, "agent", "unsupported_action", started, background
                    )
                )
        except subprocess.TimeoutExpired:
            return _ActionResult(
                diagnostic=self._diagnostic(rule, "command", "timeout", started, background)
            )
        except httpx.HTTPError:
            return _ActionResult(
                diagnostic=self._diagnostic(rule, "http", "network", started, background)
            )
        except Exception:
            return _ActionResult(
                diagnostic=self._diagnostic(
                    rule, _action_name(rule), "internal", started, background
                )
            )
        return _ActionResult()

    def _run_command(
        self,
        rule: HookRule,
        payload: Mapping[str, object],
        started: float,
        background: bool,
    ) -> _ActionResult:
        action = rule.action
        assert isinstance(action, CommandAction)
        command = action.command_windows if sys.platform == "win32" and action.command_windows else action.command
        completed = subprocess.run(
            command,
            shell=True,
            input=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            text=True,
            capture_output=True,
            cwd=self.workspace,
            timeout=action.timeout_seconds,
            check=False,
        )
        if completed.returncode == 2:
            return _ActionResult(decision="deny", reason=_reason(completed.stderr))
        if completed.returncode != 0:
            return _ActionResult(
                diagnostic=self._diagnostic(
                    rule,
                    "command",
                    "command_exit",
                    started,
                    background,
                    status=completed.returncode,
                )
            )
        return self._structured_result(
            rule, "command", completed.stdout, started, background
        )

    def _run_http(
        self,
        rule: HookRule,
        payload: Mapping[str, object],
        started: float,
        background: bool,
    ) -> _ActionResult:
        action = rule.action
        assert isinstance(action, HttpAction)
        headers: dict[str, str] = {}
        allowed = set(action.allowed_env_vars)
        for key, template in action.headers:
            headers[key] = _expand_header(template, allowed)
        included: dict[str, object] = {}
        for pointer in action.include:
            found, value = resolve_pointer(payload, parse_pointer(pointer))
            if found:
                included[pointer] = value
        body: dict[str, object] = {
            "event": rule.event.value,
            "hook": rule.name,
            "source": rule.source.value,
        }
        if included:
            body["included"] = included
        with httpx.stream(
            "POST",
            action.url,
            headers=headers,
            json=body,
            follow_redirects=False,
        ) as response:
            if not 200 <= response.status_code < 300:
                return _ActionResult(
                    diagnostic=self._diagnostic(
                        rule,
                        "http",
                        "http_status",
                        started,
                        background,
                        status=response.status_code,
                    )
                )
            content = bytearray()
            for chunk in response.iter_bytes():
                content.extend(chunk)
                if len(content) > _MAX_HTTP_RESPONSE_BYTES:
                    return _ActionResult(
                        diagnostic=self._diagnostic(
                            rule, "http", "response_limit", started, background
                        )
                    )
            encoding = response.encoding or "utf-8"
            text = bytes(content).decode(encoding, errors="replace")
        return self._structured_result(
            rule, "http", text, started, background
        )

    def _structured_result(
        self,
        rule: HookRule,
        action: str,
        content: str,
        started: float,
        background: bool,
    ) -> _ActionResult:
        if not content.strip():
            return _ActionResult()
        try:
            value = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return _ActionResult(
                diagnostic=self._diagnostic(
                    rule, action, "response_format", started, background
                )
            )
        if not isinstance(value, dict) or set(value) - {"decision", "reason", "additional_context"}:
            return _ActionResult(
                diagnostic=self._diagnostic(
                    rule, action, "response_format", started, background
                )
            )
        decision = value.get("decision")
        reason = value.get("reason")
        prompt = value.get("additional_context")
        if decision not in {None, "allow", "deny"}:
            return _ActionResult(
                diagnostic=self._diagnostic(
                    rule, action, "response_format", started, background
                )
            )
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            return _ActionResult(
                diagnostic=self._diagnostic(
                    rule, action, "response_format", started, background
                )
            )
        if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
            return _ActionResult(
                diagnostic=self._diagnostic(
                    rule, action, "response_format", started, background
                )
            )
        if decision == "deny" and not reason:
            return _ActionResult(
                diagnostic=self._diagnostic(
                    rule, action, "response_format", started, background
                )
            )
        return _ActionResult(decision, _reason(reason) if reason else None, prompt)

    @staticmethod
    def _diagnostic(
        rule: HookRule,
        action: str,
        category: str,
        started: float,
        background: bool,
        *,
        status: int | None = None,
    ) -> HookDiagnostic:
        return HookDiagnostic(
            rule.name,
            rule.source.value,
            rule.event.value,
            action,
            category,
            time.monotonic() - started,
            background,
            status,
        )

    def _report(self, diagnostic: HookDiagnostic) -> None:
        if self.diagnostic_sink is None:
            return
        try:
            self.diagnostic_sink(diagnostic)
        except Exception:
            pass


def _action_name(rule: HookRule) -> str:
    name = type(rule.action).__name__
    return name.removesuffix("Action").lower()


def _reason(value: str | None) -> str:
    if not value or not value.strip():
        return "Hook 策略拒绝了该操作。"
    return " ".join(value.split())


def _expand_header(template: str, allowed: set[str]) -> str:
    result = template
    for name in allowed:
        marker = "${" + name + "}"
        if marker in result:
            value = os.environ.get(name)
            if value is None:
                raise ValueError("allowlisted Hook environment variable is unavailable")
            result = result.replace(marker, value)
    return result
