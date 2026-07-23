from __future__ import annotations

from pathlib import Path
from threading import Event, Thread

from fakuicode.skills.install import (
    SkillInstallDecision,
    SkillInstallPreset,
    SkillInstallPreview,
    SkillInstallScope,
)
from fakuicode.skills.install_broker import SkillInstallBroker


def _preview() -> SkillInstallPreview:
    return SkillInstallPreview(
        "frontend-design",
        "Build polished interfaces",
        "Apache-2.0",
        "https://www.skills.sh/anthropics/skills/frontend-design",
        "https://github.com/anthropics/skills",
        "a" * 40,
        "skills/frontend-design",
        Path("project/.fakuicode/skills/frontend-design"),
        SkillInstallScope.PROJECT,
        SkillInstallPreset.CODING,
        ("read_file", "write_file"),
        ("LICENSE.txt", "SKILL.md"),
        120,
        False,
        (),
        False,
        (),
    )


def test_broker_returns_selected_preset() -> None:
    broker = SkillInstallBroker(poll_seconds=0.001)
    preview = _preview()
    result: list[SkillInstallDecision] = []
    worker = Thread(target=lambda: result.append(broker.request(preview)))
    worker.start()

    while broker.next_request() is None:
        pass
    assert broker.resolve(preview, SkillInstallDecision(True, SkillInstallPreset.READ_ONLY))
    worker.join(timeout=1)

    assert result == [SkillInstallDecision(True, SkillInstallPreset.READ_ONLY)]


def test_broker_cancellation_is_fail_closed() -> None:
    broker = SkillInstallBroker(poll_seconds=0.001)
    preview = _preview()
    cancelled = Event()
    result: list[SkillInstallDecision] = []
    worker = Thread(target=lambda: result.append(broker.request(preview, cancel_event=cancelled)))
    worker.start()
    cancelled.set()
    worker.join(timeout=1)

    assert result == [SkillInstallDecision(False, SkillInstallPreset.CODING)]
