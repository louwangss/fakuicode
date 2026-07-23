# MCP 外部工具

fakuiCode 在启动时读取 MCP 配置，完成标准 `initialize` / `notifications/initialized` 握手、分页 `tools/list`，再把可用工具注册到现有 Tool Registry。Agent 看到的名称为 `mcp__<server>__<tool>`；实际调用仍使用 Server 公布的原始工具名。

## 配置位置与合并

- 用户级：`~/.fakuicode/mcp.yaml`
- 项目级：`<workspace>/.fakuicode/mcp.yaml`

两个文件都使用顶层 `mcp_servers` map。先读取用户级，再由项目级按 Server 名称完整覆盖；不会逐字段继承。项目级 `{enabled: false}` 可以屏蔽同名用户 Server。Server 名称必须匹配 `^[a-z][a-z0-9_]{0,31}$`。

用户级示例：

```yaml
mcp_servers:
  local_docs:
    type: stdio
    command: python
    args: ["C:/tools/docs_server.py"]
    env:
      DOCS_TOKEN: "${DOCS_TOKEN}"
    enabled_tools: [search, fetch]
    disabled_tools: [fetch_private]

  team_api:
    type: http
    url: https://mcp.example.com/mcp
    headers:
      Authorization: "Bearer ${MCP_API_TOKEN}"
```

项目级示例：

```yaml
mcp_servers:
  local_docs:
    type: stdio
    command: uvx
    args: [project-docs-server, --root, .]
    env:
      PROJECT_TOKEN: "${PROJECT_TOKEN}"

  team_api:
    enabled: false
```

上例中的项目 `local_docs` 完整替换用户定义；用户层的 `command`、`args`、`env` 和过滤器都不会残留。`disabled_tools` 始终优先于 `enabled_tools`。

## 变量与传输安全

`${VAR}` 只会在 `env` 和 `headers` 的值中展开；Server 名、`command`、`args`、URL 和工具过滤器不会展开。缺少变量或引用格式错误只会跳过对应 Server，诊断仅显示变量名，不显示值。

HTTP Server 必须使用 HTTPS。为本地开发提供的例外仅包括字面量 `localhost`、IPv4 loopback 和 `::1`；不接受远程明文 HTTP、URL user-info 或 fragment。stdio 子进程只继承官方 SDK 的最小系统环境，再加入配置中显式声明的 `env`。

## 信任、权限与生命周期

项目级 Server 在连接前逐个请求信任。确认界面只显示 Server 名、脱敏命令或 URL、参数数量、环境变量名和 Header 名；不显示任何值。批准记录保存到 `~/.fakuicode/mcp-trust.yaml`，只包含工作区哈希、Server 名和配置指纹。安全相关配置一旦变化，旧批准自动失效；拒绝仅影响本次启动。

每个 MCP 工具都按有副作用工具处理：

- `/plan` 中不可调用。
- “仅本次”只允许当前调用。
- “本会话”和“永久”授权的是整个 MCP 工具，对所有参数生效，确认界面会明确提示这一点。
- 参数、Header、环境值和完整结果不会写入权限规则。

多个 Server 并行连接并在应用生命周期内复用；新建会话、恢复会话和切换模型不会重连。单个 Server 连接、协议或工具发现失败不会影响其他 Server。启动发现每个 Server 最多等待 10 秒，单次工具调用最多等待 60 秒，退出时最多等待 5 秒清理。当前版本不做应用级健康检查、自动重连或热更新；收到 `tools/list_changed` 后 `/mcp` 会提示重启生效。

## 状态与结果

存在 MCP 配置时，启动完成会显示：

```text
Connected to 1 MCP server(s), 2 tools registered
```

`/mcp` 可查看每个 Server 的 connected、failed、disabled、pending trust、trust denied 或 restart required 状态。MCP 工具活动各占一条紧凑状态行并显示耗时，不显示参数。文本结果保持顺序；`structuredContent` 使用稳定 JSON；图片、音频和资源只显示类型占位；输出最多 12,000 字符。

当前只支持 MCP 工具能力，不支持 resources、prompts、sampling、roots 管理、Server 健康检查或自动重连。
