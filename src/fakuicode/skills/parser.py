"""Strict parsing and fingerprinting for directory-based Skill packages."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator, SchemaError
import yaml

from fakuicode.context import approximate_token_count
from fakuicode.instructions.models import DEFAULT_INSTRUCTION_LIMITS
from fakuicode.skills.models import (
    SkillDefinition,
    SkillExecution,
    SkillInstallReceipt,
    SkillInvocation,
    SkillSource,
    SkillToolSpec,
)


_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_REQUIRED_TOP_FIELDS = {"name", "description"}
_TOP_FIELDS = {
    "name",
    "description",
    "license",
    "compatibility",
    "metadata",
    "allowed-tools",
    "fakuicode",
}
_EXTENSION_FIELDS = {"invocation", "visible-tools", "execution", "history-turns", "profile"}
_TOOL_FIELDS = {"name", "description", "input_schema", "entrypoint"}
_RECEIPT_FIELDS = {
    "schema-version",
    "requested-url",
    "source-url",
    "revision",
    "skill-path",
    "upstream-fingerprint",
    "fakuicode",
}
_RECEIPT_PATH = PurePosixPath(".fakuicode/install.yaml")


class SkillParseError(ValueError):
    pass


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise SkillParseError("YAML contains a duplicate key")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def parse_skill_package(package: Path, source: SkillSource) -> SkillDefinition:
    _assert_safe_package(package)
    entry = package / "SKILL.md"
    text = _read_bounded_text(entry)
    frontmatter, body = _split_document(text)
    try:
        raw = yaml.load(frontmatter, Loader=_UniqueKeyLoader)
    except (yaml.YAMLError, SkillParseError) as error:
        raise SkillParseError("frontmatter is invalid") from error
    if (
        not isinstance(raw, Mapping)
        or not _REQUIRED_TOP_FIELDS.issubset(raw)
        or not set(raw).issubset(_TOP_FIELDS)
    ):
        raise SkillParseError("frontmatter fields are invalid")

    name = raw.get("name")
    description = raw.get("description")
    extension = raw.get("fakuicode", {})
    if not isinstance(name, str) or len(name) > 64 or _SKILL_NAME.fullmatch(name) is None:
        raise SkillParseError("name is invalid")
    if name != package.name:
        raise SkillParseError("name must match the package directory")
    if not isinstance(description, str) or not description.strip() or len(description) > 1024:
        raise SkillParseError("description is invalid")
    license_value = _optional_string(raw.get("license"), "license")
    compatibility = _optional_string(raw.get("compatibility"), "compatibility", max_length=500)
    metadata = _metadata(raw.get("metadata"))
    allowed_tools = _optional_string(raw.get("allowed-tools"), "allowed-tools")
    receipt, receipt_extension = _parse_install_receipt(package)
    if receipt_extension is not None:
        extension = receipt_extension
    invocation, visible_tools, execution, history_turns, profile = _parse_extension(extension)

    tools = _parse_tools(package)
    own_names = {f"skill__{name}__{tool.name}" for tool in tools}
    if own_names.intersection(visible_tools):
        raise SkillParseError("dedicated tools must not be repeated in visible-tools")
    warnings = ()
    if approximate_token_count(body) > 5_000:
        warnings = ("skill_body_over_5000_tokens",)
    return SkillDefinition(
        name=name,
        description=description.strip(),
        source=source,
        package_path=package,
        body=body.strip(),
        invocation=invocation,
        visible_tools=visible_tools,
        execution=execution,
        history_turns=history_turns,
        profile=profile,
        tools=tools,
        fingerprint=fingerprint_package(package),
        license=license_value,
        compatibility=compatibility,
        metadata=metadata,
        allowed_tools=allowed_tools,
        install_receipt=receipt,
        author_warnings=warnings,
    )


def fingerprint_package(package: Path) -> str:
    return _fingerprint_files(package, _package_files(package))


def fingerprint_upstream(package: Path) -> str:
    files = tuple(
        path
        for path in _package_files(package)
        if PurePosixPath(path.relative_to(package).as_posix()) != _RECEIPT_PATH
    )
    return _fingerprint_files(package, files)


def _fingerprint_files(package: Path, files: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(package).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _parse_install_receipt(
    package: Path,
) -> tuple[SkillInstallReceipt | None, Mapping[str, object] | None]:
    path = package.joinpath(*_RECEIPT_PATH.parts)
    if not path.exists():
        return None, None
    try:
        raw = yaml.load(_read_bounded_text(path), Loader=_UniqueKeyLoader)
    except (yaml.YAMLError, SkillParseError) as error:
        raise SkillParseError("install receipt is invalid") from error
    if not isinstance(raw, Mapping) or set(raw) != _RECEIPT_FIELDS:
        raise SkillParseError("install receipt fields are invalid")
    if raw.get("schema-version") != 1:
        raise SkillParseError("install receipt schema version is unsupported")
    requested_url = _https_url(raw.get("requested-url"), "requested-url", {"skills.sh", "www.skills.sh", "github.com"})
    source_url = _https_url(raw.get("source-url"), "source-url", {"github.com"})
    revision = raw.get("revision")
    skill_path = raw.get("skill-path")
    upstream_fingerprint = raw.get("upstream-fingerprint")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise SkillParseError("install receipt revision is invalid")
    if not isinstance(skill_path, str) or not _safe_relative_posix_path(skill_path):
        raise SkillParseError("install receipt skill path is invalid")
    if not isinstance(upstream_fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", upstream_fingerprint) is None:
        raise SkillParseError("install receipt upstream fingerprint is invalid")
    extension = raw.get("fakuicode")
    _parse_extension(extension)
    actual_fingerprint = fingerprint_upstream(package)
    if actual_fingerprint != upstream_fingerprint:
        raise SkillParseError("install receipt upstream fingerprint does not match package contents")
    return (
        SkillInstallReceipt(
            1,
            requested_url,
            source_url,
            revision,
            skill_path,
            upstream_fingerprint,
        ),
        extension,
    )


def _parse_extension(
    extension: object,
) -> tuple[SkillInvocation, tuple[str, ...], SkillExecution, int, str]:
    if extension is None:
        extension = {}
    if not isinstance(extension, Mapping) or not set(extension).issubset(_EXTENSION_FIELDS):
        raise SkillParseError("fakuicode fields are invalid")
    invocation = _enum_value(SkillInvocation, extension.get("invocation", "auto"), "invocation")
    execution = _enum_value(SkillExecution, extension.get("execution", "shared"), "execution")
    visible_tools = _string_list(extension.get("visible-tools", []), "visible-tools")
    history_turns = extension.get("history-turns", 0)
    profile = extension.get("profile", "inherit")
    if isinstance(history_turns, bool) or not isinstance(history_turns, int) or history_turns < 0:
        raise SkillParseError("history-turns must be a non-negative integer")
    if not isinstance(profile, str) or not profile.strip():
        raise SkillParseError("profile is invalid")
    if execution is SkillExecution.SHARED and (history_turns != 0 or profile != "inherit"):
        raise SkillParseError("shared Skills cannot select history or a profile")
    return invocation, visible_tools, execution, history_turns, profile


def _optional_string(value: object, field: str, *, max_length: int | None = None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or (max_length is not None and len(value) > max_length):
        raise SkillParseError(f"{field} is invalid")
    return value.strip()


def _metadata(value: object) -> Mapping[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()
    ):
        raise SkillParseError("metadata is invalid")
    return dict(value)


def _https_url(value: object, field: str, hosts: set[str]) -> str:
    if not isinstance(value, str):
        raise SkillParseError(f"install receipt {field} is invalid")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise SkillParseError(f"install receipt {field} is invalid") from error
    host = (parsed.hostname or "").casefold()
    parts = tuple(part for part in parsed.path.split("/") if part)
    if (
        parsed.scheme != "https"
        or host not in hosts
        or parsed.username
        or parsed.password
        or port is not None
        or parsed.query
        or parsed.fragment
        or parsed.netloc.casefold() != host
        or parsed.path.endswith("/")
    ):
        raise SkillParseError(f"install receipt {field} is invalid")
    if field == "source-url" and (len(parts) != 2 or parsed.path != f"/{parts[0]}/{parts[1]}"):
        raise SkillParseError(f"install receipt {field} is invalid")
    if field == "requested-url":
        skills_url = host in {"skills.sh", "www.skills.sh"} and len(parts) == 3
        github_url = host == "github.com" and (
            len(parts) == 2 or (len(parts) >= 5 and parts[2] == "tree")
        )
        if not skills_url and not github_url:
            raise SkillParseError(f"install receipt {field} is invalid")
    return value


def _safe_relative_posix_path(value: str) -> bool:
    if value == ".":
        return True
    path = PurePosixPath(value)
    return bool(path.parts) and not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def _parse_tools(package: Path) -> tuple[SkillToolSpec, ...]:
    tools_root = package / "tools"
    if not tools_root.exists():
        return ()
    if not tools_root.is_dir() or _is_reparse(tools_root):
        raise SkillParseError("tools must be a regular package directory")
    parsed: list[SkillToolSpec] = []
    names: set[str] = set()
    for path in sorted(tools_root.iterdir(), key=lambda item: item.name):
        if path.suffix != ".json" or not path.is_file() or _is_reparse(path):
            raise SkillParseError("tools may contain only regular JSON descriptors")
        try:
            raw = json.loads(_read_bounded_text(path), object_pairs_hook=_unique_json_object)
        except json.JSONDecodeError as error:
            raise SkillParseError("tool descriptor JSON is invalid") from error
        if not isinstance(raw, Mapping) or set(raw) != _TOOL_FIELDS:
            raise SkillParseError("tool descriptor fields are invalid")
        name = raw.get("name")
        description = raw.get("description")
        schema = raw.get("input_schema")
        entrypoint = raw.get("entrypoint")
        if not isinstance(name, str) or _TOOL_NAME.fullmatch(name) is None or name in names:
            raise SkillParseError("tool name is invalid or duplicated")
        if path.stem != name:
            raise SkillParseError("tool descriptor filename must match its name")
        if not isinstance(description, str) or not description.strip():
            raise SkillParseError("tool description is invalid")
        if not isinstance(schema, Mapping):
            raise SkillParseError("tool input_schema is invalid")
        try:
            Draft202012Validator.check_schema(dict(schema))
        except SchemaError as error:
            raise SkillParseError("tool input_schema is not valid JSON Schema") from error
        if not isinstance(entrypoint, str):
            raise SkillParseError("tool entrypoint is invalid")
        parts = PurePosixPath(entrypoint).parts
        if len(parts) != 2 or parts[0] != "scripts" or not parts[1].endswith(".py"):
            raise SkillParseError("tool entrypoint must be scripts/*.py")
        script = package.joinpath(*parts)
        if not script.is_file() or _is_reparse(script):
            raise SkillParseError("tool entrypoint does not name a safe regular file")
        names.add(name)
        parsed.append(SkillToolSpec(name, description.strip(), dict(schema), script))
    return tuple(parsed)


def _split_document(text: str) -> tuple[str, str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---\n"):
        raise SkillParseError("SKILL.md must start with YAML frontmatter")
    end = normalized.find("\n---\n", 4)
    if end < 0:
        raise SkillParseError("SKILL.md frontmatter is not closed")
    body = normalized[end + 5 :]
    if not body.strip():
        raise SkillParseError("Skill instructions cannot be empty")
    return normalized[4:end], body


def _read_bounded_text(path: Path) -> str:
    if not path.is_file() or _is_reparse(path):
        raise SkillParseError("required package file is missing or unsafe")
    try:
        with path.open("rb") as handle:
            data = handle.read(DEFAULT_INSTRUCTION_LIMITS.max_read_bytes)
    except OSError as error:
        raise SkillParseError("package file cannot be read") from error
    if len(data) > DEFAULT_INSTRUCTION_LIMITS.max_source_bytes:
        raise SkillParseError("package file is too large")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise SkillParseError("package file is not UTF-8") from error


def _assert_safe_package(package: Path) -> None:
    if not package.is_dir() or _is_reparse(package):
        raise SkillParseError("package path is not a safe directory")
    try:
        resolved = package.resolve(strict=True)
        absolute = package.absolute()
    except OSError as error:
        raise SkillParseError("package path cannot be resolved") from error
    if os.path.normcase(str(resolved)) != os.path.normcase(str(absolute)):
        raise SkillParseError("package path traverses a symbolic link or reparse point")
    # Reuse the instruction loader's established target bound for untrusted packages.
    files = _package_files(package)
    if len(files) > DEFAULT_INSTRUCTION_LIMITS.max_file_targets:
        raise SkillParseError("package contains too many files")
    for path in files:
        if path.stat().st_size > DEFAULT_INSTRUCTION_LIMITS.max_source_bytes:
            raise SkillParseError("package file is too large")


def _package_files(package: Path) -> tuple[Path, ...]:
    result: list[Path] = []
    for root, directories, files in os.walk(package, followlinks=False):
        root_path = Path(root)
        for directory in directories:
            if _is_reparse(root_path / directory):
                raise SkillParseError("package contains a reparse point")
        for filename in files:
            path = root_path / filename
            if _is_reparse(path) or not path.is_file():
                raise SkillParseError("package contains an unsafe file")
            result.append(path)
    return tuple(sorted(result, key=lambda item: item.relative_to(package).as_posix()))


def _is_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        return bool(path.lstat().st_file_attributes & 0x400)
    except (AttributeError, OSError):
        return False


def _enum_value(enum_type, value: object, field: str):
    if not isinstance(value, str):
        raise SkillParseError(f"{field} is invalid")
    try:
        return enum_type(value)
    except ValueError as error:
        raise SkillParseError(f"{field} is invalid") from error


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SkillParseError("tool descriptor JSON contains a duplicate key")
        result[key] = value
    return result


def _string_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise SkillParseError(f"{field} must be a string list")
    if len(set(value)) != len(value):
        raise SkillParseError(f"{field} contains duplicates")
    return tuple(value)
