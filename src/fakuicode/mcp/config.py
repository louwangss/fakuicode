"""Strict, isolated loading for user and project MCP server configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import ipaddress
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

import yaml

from fakuicode.mcp.models import (
    DisabledServerConfig,
    HttpServerConfig,
    McpConfigSnapshot,
    McpConfigSource,
    McpDiagnostic,
    McpFailureCode,
    McpServerConfig,
    ResolvedHttpServerConfig,
    ResolvedServerConfig,
    ResolvedStdioServerConfig,
    SecretText,
    StdioServerConfig,
)


_SERVER_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_VARIABLE_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_COMMON_FIELDS = {"type", "enabled", "enabled_tools", "disabled_tools"}
_STDIO_FIELDS = _COMMON_FIELDS | {"command", "args", "env"}
_HTTP_FIELDS = _COMMON_FIELDS | {"url", "headers"}


@dataclass(frozen=True)
class McpPaths:
    user: Path
    project: Path

    @classmethod
    def for_workspace(cls, workspace: Path, *, home: Path | None = None) -> McpPaths:
        return cls(
            (home or Path.home()) / ".fakuicode" / "mcp.yaml",
            workspace / ".fakuicode" / "mcp.yaml",
        )


@dataclass
class _YamlMap:
    values: dict[Any, Any]
    duplicates: tuple[object, ...] = ()


class _DuplicateAwareLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _DuplicateAwareLoader, node: yaml.MappingNode, deep: bool = False
) -> _YamlMap:
    values: dict[Any, Any] = {}
    duplicates: list[object] = []
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in values
        except TypeError:
            key = None
            duplicate = True
        value = loader.construct_object(value_node, deep=deep)
        if duplicate:
            duplicates.append(key)
        else:
            values[key] = value
    return _YamlMap(values, tuple(duplicates))


_DuplicateAwareLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


class McpConfigRepository:
    """Load two independent layers and merge them by complete server definition."""

    def __init__(self, paths: McpPaths, workspace: Path) -> None:
        self.paths = paths
        self.workspace = workspace.resolve()

    def load(self) -> McpConfigSnapshot:
        user, user_diagnostics, user_present = _load_layer(
            self.paths.user, McpConfigSource.USER
        )
        try:
            _validate_project_path(self.paths.project, self.workspace)
        except ValueError:
            project: dict[str, McpServerConfig] = {}
            project_diagnostics = [
                McpDiagnostic(
                    "项目 MCP 配置路径不安全，已忽略项目层。",
                    source=McpConfigSource.PROJECT,
                )
            ]
            project_present = False
        else:
            project, project_diagnostics, project_present = _load_layer(
                self.paths.project, McpConfigSource.PROJECT
            )

        merged = dict(user)
        merged.update(project)
        return McpConfigSnapshot(
            servers=tuple(merged[name] for name in sorted(merged)),
            diagnostics=tuple(user_diagnostics + project_diagnostics),
            has_configuration=user_present or project_present,
        )


def resolve_server(
    config: McpServerConfig,
    environment: Mapping[str, str],
) -> tuple[ResolvedServerConfig | None, McpDiagnostic | None]:
    """Expand secret-bearing values only and return a sanitized failure when needed."""

    if isinstance(config, DisabledServerConfig):
        return None, None
    templates = config.env_templates if isinstance(config, StdioServerConfig) else config.header_templates
    resolved: dict[str, SecretText] = {}
    missing: set[str] = set()
    malformed = False
    for key, template in templates.items():
        try:
            value, absent = _expand_template(template, environment)
        except ValueError:
            malformed = True
            continue
        missing.update(absent)
        resolved[key] = SecretText(value)
    if malformed:
        return None, McpDiagnostic(
            "环境变量引用格式无效。",
            server_name=config.name,
            source=config.source,
            failure_code=McpFailureCode.MISSING_ENVIRONMENT,
        )
    if missing:
        return None, McpDiagnostic(
            f"缺少环境变量：{', '.join(sorted(missing))}",
            server_name=config.name,
            source=config.source,
            failure_code=McpFailureCode.MISSING_ENVIRONMENT,
        )
    if isinstance(config, StdioServerConfig):
        return ResolvedStdioServerConfig(
            config.name,
            config.command,
            config.args,
            resolved,
            config.enabled_tools,
            config.disabled_tools,
        ), None
    return ResolvedHttpServerConfig(
        config.name,
        config.url,
        resolved,
        config.enabled_tools,
        config.disabled_tools,
    ), None


def referenced_variables(config: McpServerConfig) -> tuple[str, ...]:
    if isinstance(config, DisabledServerConfig):
        return ()
    templates = config.env_templates if isinstance(config, StdioServerConfig) else config.header_templates
    names: set[str] = set()
    for value in templates.values():
        names.update(_VARIABLE_REFERENCE.findall(value))
    return tuple(sorted(names))


def _load_layer(
    path: Path, source: McpConfigSource
) -> tuple[dict[str, McpServerConfig], list[McpDiagnostic], bool]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}, [], False
    except OSError:
        return {}, [McpDiagnostic("MCP 配置文件无法读取。", source=source)], True
    try:
        root = yaml.load(text, Loader=_DuplicateAwareLoader)
    except (yaml.YAMLError, TypeError, ValueError):
        return {}, [McpDiagnostic("MCP 配置 YAML 无法解析，已忽略该层。", source=source)], True
    if root is None:
        return {}, [], False
    if not isinstance(root, _YamlMap):
        return {}, [McpDiagnostic("MCP 配置顶层必须是 mapping，已忽略该层。", source=source)], True
    if root.duplicates or not _string_keys(root) or set(root.values) - {"mcp_servers"}:
        return {}, [McpDiagnostic("MCP 配置顶层字段无效，已忽略该层。", source=source)], True
    raw_servers = root.values.get("mcp_servers")
    if raw_servers is None:
        return {}, [], False
    if not isinstance(raw_servers, _YamlMap) or raw_servers.duplicates or not _string_keys(raw_servers):
        return {}, [McpDiagnostic("mcp_servers 必须是无重复名称的 mapping，已忽略该层。", source=source)], True
    if not raw_servers.values:
        return {}, [], False

    servers: dict[str, McpServerConfig] = {}
    diagnostics: list[McpDiagnostic] = []
    for name, raw in raw_servers.values.items():
        try:
            servers[name] = _parse_server(name, raw, source)
        except ValueError as error:
            diagnostics.append(McpDiagnostic(str(error), name, source))
    return servers, diagnostics, True


def _parse_server(name: str, raw: object, source: McpConfigSource) -> McpServerConfig:
    if not _SERVER_NAME.fullmatch(name):
        raise ValueError("Server 名称必须匹配 ^[a-z][a-z0-9_]{0,31}$。")
    if not isinstance(raw, _YamlMap) or not _string_keys(raw) or _contains_duplicate(raw):
        raise ValueError("Server 定义必须是无重复字符串键的 mapping。")
    values = raw.values
    enabled = values.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("enabled 必须是 boolean。")
    if not enabled and set(values) == {"enabled"}:
        return DisabledServerConfig(name, source)

    transport = values.get("type")
    if transport not in {"stdio", "http"}:
        raise ValueError("type 必须显式设置为 stdio 或 http。")
    allowed = _STDIO_FIELDS if transport == "stdio" else _HTTP_FIELDS
    if set(values) - allowed:
        raise ValueError("Server 定义包含未知或不适用于该传输的字段。")
    enabled_tools = _parse_tool_filter(values.get("enabled_tools"), "enabled_tools", optional=True)
    disabled_tools = _parse_tool_filter(values.get("disabled_tools"), "disabled_tools", optional=False)

    if transport == "stdio":
        command = values.get("command")
        args = values.get("args", [])
        env = values.get("env", _YamlMap({}))
        if not isinstance(command, str) or not command.strip():
            raise ValueError("stdio command 必须是非空字符串。")
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            raise ValueError("stdio args 必须是字符串列表。")
        env_values = _parse_string_mapping(env, "env", _ENVIRONMENT_NAME)
        result: McpServerConfig = StdioServerConfig(
            name, source, command, tuple(args), env_values, enabled_tools, disabled_tools
        )
    else:
        url = values.get("url")
        headers = values.get("headers", _YamlMap({}))
        if not isinstance(url, str) or not url.strip():
            raise ValueError("HTTP url 必须是非空字符串。")
        _validate_http_url(url)
        header_values = _parse_string_mapping(headers, "headers", _HEADER_NAME)
        result = HttpServerConfig(
            name, source, url, header_values, enabled_tools, disabled_tools
        )
    if not enabled:
        return DisabledServerConfig(name, source)
    return result


def _parse_tool_filter(value: object, label: str, *, optional: bool) -> frozenset[str] | None:
    if value is None:
        return None if optional else frozenset()
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{label} 必须是非空字符串组成的列表。")
    return frozenset(value)


def _parse_string_mapping(value: object, label: str, key_pattern: re.Pattern[str]) -> dict[str, str]:
    if not isinstance(value, _YamlMap) or _contains_duplicate(value) or not _string_keys(value):
        raise ValueError(f"{label} 必须是无重复字符串键的 mapping。")
    if not all(key_pattern.fullmatch(key) for key in value.values):
        raise ValueError(f"{label} 包含无效名称。")
    if not all(isinstance(item, str) for item in value.values.values()):
        raise ValueError(f"{label} 的值必须是字符串。")
    return dict(value.values)


def _validate_http_url(value: str) -> None:
    try:
        parsed = urlparse(value)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as error:
        raise ValueError("HTTP url 无效。") from error
    if parsed.scheme not in {"http", "https"} or not hostname or parsed.fragment:
        raise ValueError("HTTP url 必须是无 fragment 的 HTTP(S) 地址。")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("HTTP url 不得包含 user-info。")
    if parsed.scheme == "http" and not _is_literal_loopback(hostname):
        raise ValueError("HTTP 明文地址仅允许 localhost 或字面量 loopback。")


def _is_literal_loopback(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _expand_template(value: str, environment: Mapping[str, str]) -> tuple[str, set[str]]:
    missing: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in environment:
            missing.add(name)
            return ""
        return environment[name]

    expanded = _VARIABLE_REFERENCE.sub(replace, value)
    if "${" in expanded:
        raise ValueError("malformed variable reference")
    return expanded, missing


def _string_keys(value: _YamlMap) -> bool:
    return all(isinstance(key, str) for key in value.values)


def _contains_duplicate(value: object) -> bool:
    if isinstance(value, _YamlMap):
        return bool(value.duplicates) or any(_contains_duplicate(item) for item in value.values.values())
    if isinstance(value, list):
        return any(_contains_duplicate(item) for item in value)
    return False


def _validate_project_path(path: Path, workspace: Path) -> None:
    lexical = Path(os.path.abspath(path))
    try:
        lexical.relative_to(workspace)
    except ValueError as error:
        raise ValueError("outside workspace") from error
    missing: list[str] = []
    ancestor = lexical
    while not ancestor.exists() and ancestor != ancestor.parent:
        missing.append(ancestor.name)
        ancestor = ancestor.parent
    try:
        resolved = ancestor.resolve(strict=True).joinpath(*reversed(missing)).resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ValueError("unsafe project path") from error
    if resolved != lexical:
        raise ValueError("redirected project path")
