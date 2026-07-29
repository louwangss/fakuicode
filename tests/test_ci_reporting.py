from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_pytest_failure_reporter_emits_a_located_github_annotation(tmp_path: Path) -> None:
    report = tmp_path / "pytest-results.xml"
    report.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="1" failures="1">
    <testcase classname="tests.tui.test_app" name="test_compact_brand_panel">
      <failure message="assert 15 &lt;= 12">tests/tui/test_app.py:3113: AssertionError</failure>
    </testcase>
  </testsuite>
</testsuites>
""",
        encoding="utf-8",
    )
    reporter = Path(__file__).parents[1] / ".github" / "scripts" / "report_pytest_failures.py"

    completed = subprocess.run(
        [sys.executable, str(reporter), str(report)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == (
        "::error file=tests/tui/test_app.py,line=3113,"
        "title=pytest%3A tests.tui.test_app.test_compact_brand_panel::assert 15 <= 12"
    )
