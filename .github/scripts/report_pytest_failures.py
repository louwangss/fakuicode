from __future__ import annotations

from pathlib import Path
import re
import sys
from xml.etree import ElementTree


_LOCATION_PATTERN = re.compile(r"(?m)^(?P<file>.+?\.py):(?P<line>\d+)(?::|$)")


def _escape_message(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(value: str) -> str:
    return _escape_message(value).replace(":", "%3A").replace(",", "%2C")


def _failure_annotation(testcase: ElementTree.Element) -> str | None:
    failure = testcase.find("failure")
    if failure is None:
        failure = testcase.find("error")
    if failure is None:
        return None

    details = (failure.text or "").strip()
    detail_summary = next((line.strip() for line in details.splitlines() if line.strip()), "")
    message = (failure.get("message") or detail_summary or "pytest failed").strip()
    test_name = ".".join(
        part for part in (testcase.get("classname", ""), testcase.get("name", "")) if part
    )
    properties = [f"title={_escape_property(f'pytest: {test_name}')}" if test_name else "title=pytest"]

    location = _LOCATION_PATTERN.search(details)
    file_name = testcase.get("file") or (location.group("file") if location else "")
    line_number = testcase.get("line") or (location.group("line") if location else "")
    if file_name:
        properties.insert(0, f"file={_escape_property(file_name)}")
    if line_number and line_number.isdigit():
        properties.insert(1 if file_name else 0, f"line={line_number}")

    return f"::error {','.join(properties)}::{_escape_message(message)}"


def main(report_path: str = "pytest-results.xml") -> int:
    path = Path(report_path)
    if not path.is_file():
        print(f"::error title=pytest report missing::{_escape_message(str(path))}")
        return 0

    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as error:
        print(f"::error title=invalid pytest report::{_escape_message(str(error))}")
        return 0

    for testcase in root.iter("testcase"):
        annotation = _failure_annotation(testcase)
        if annotation is not None:
            print(annotation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "pytest-results.xml"))
