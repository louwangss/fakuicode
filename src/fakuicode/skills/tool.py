"""Execution adapter for trusted package-local Python Skill tools."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import subprocess
import sys
from threading import Event
from time import monotonic

from jsonschema import Draft202012Validator, ValidationError

from fakuicode.errors import ToolExecutionError
from fakuicode.models import ToolDefinition
from fakuicode.skills.models import SkillDefinition, SkillToolSpec
from fakuicode.skills.parser import fingerprint_package
from fakuicode.tools.base import ToolExecution, ToolPreparation, freeze_arguments
from fakuicode.tools.command import DEFAULT_COMMAND_TIMEOUT_SECONDS


class SkillScriptTool:
    def __init__(
        self,
        workspace: Path,
        skill: SkillDefinition,
        spec: SkillToolSpec,
        *,
        timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ) -> None:
        self.workspace = workspace
        self.skill = skill
        self.spec = spec
        self.timeout_seconds = timeout_seconds
        self._validator = Draft202012Validator(dict(spec.input_schema))

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            f"skill__{self.skill.name}__{self.spec.name}",
            self.spec.description,
            self.spec.input_schema,
        )

    @property
    def read_only(self) -> bool:
        return False

    def prepare(self, arguments: Mapping[str, object]) -> ToolPreparation:
        self._assert_current_fingerprint()
        try:
            self._validator.validate(dict(arguments))
        except ValidationError as error:
            raise ToolExecutionError("Skill tool arguments do not match input_schema.") from error
        target = f"skill:{self.skill.name}:{self.skill.fingerprint}:{self.spec.name}"
        return ToolPreparation(freeze_arguments(arguments), target)

    def execute(self, arguments: Mapping[str, object], *, cancel_event: Event | None = None) -> ToolExecution:
        return self.execute_prepared(self.prepare(arguments).arguments, cancel_event=cancel_event)

    def execute_prepared(
        self, arguments: Mapping[str, object], *, cancel_event: Event | None = None
    ) -> ToolExecution:
        self._assert_current_fingerprint()
        try:
            process = subprocess.Popen(
                [sys.executable, str(self.spec.entrypoint)],
                cwd=self.workspace,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
            )
        except OSError as error:
            raise ToolExecutionError("Unable to start the Skill tool.") from error
        input_text = json.dumps(dict(arguments), ensure_ascii=False, separators=(",", ":"))
        deadline = monotonic() + self.timeout_seconds
        pending_input: str | None = input_text
        while True:
            if cancel_event is not None and cancel_event.is_set():
                process.terminate()
                process.communicate()
                return ToolExecution(False, "Skill tool was cancelled.", "skill tool cancelled")
            remaining = deadline - monotonic()
            if remaining <= 0:
                process.terminate()
                process.communicate()
                raise ToolExecutionError("Skill tool timed out.")
            try:
                stdout, stderr = process.communicate(input=pending_input, timeout=min(0.1, remaining))
                break
            except subprocess.TimeoutExpired:
                pending_input = None
                continue
        if process.returncode != 0:
            detail = stderr.strip() or f"exit {process.returncode}"
            raise ToolExecutionError(f"Skill tool failed: {detail}")
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as error:
            raise ToolExecutionError("Skill tool returned invalid JSON.") from error
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"output", "summary"}
            or not isinstance(payload.get("output"), str)
            or not isinstance(payload.get("summary"), str)
        ):
            raise ToolExecutionError("Skill tool returned an invalid result object.")
        return ToolExecution(True, payload["output"], payload["summary"])

    def _assert_current_fingerprint(self) -> None:
        try:
            current = fingerprint_package(self.skill.package_path)
        except (OSError, ValueError) as error:
            raise ToolExecutionError("Skill package can no longer be verified.") from error
        if current != self.skill.fingerprint:
            raise ToolExecutionError("Skill package changed after activation; reactivate it before running code.")
