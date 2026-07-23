# 生命周期 Hooks

Hooks 用声明式的“事件 + 条件 + 动作”规则，在 Agent 生命周期的固定节点执行自动化。用户规则位于 `~/.fakuicode/hooks.yaml`；项目规则位于 `<workspace>/.fakuicode/hooks.yaml`。两者都只在启动时加载。

项目规则可以执行本地命令和发出网络请求，因此不会复用普通工作区信任。首次发现有效项目规则时，界面会展示路径、规则数和文件 SHA-256 指纹并要求确认；信任保存在 `~/.fakuicode/trusted-hooks.yaml`。文件内容变化后，旧指纹立即失效，项目规则保持禁用直到再次确认。用户规则视为用户自己的本机配置，不再额外询问。

## 基本格式

```yaml
version: 1
hooks:
  - name: format-python
    event: post_tool_use
    if:
      all:
        - field: /tool/name
          exact: write_file
        - field: /tool/arguments/path
          glob: "**/*.py"
        - field: /tool/outcome
          exact: failed
          not: true
    action:
      type: command
      command: python -m ruff format
      command_windows: py -m ruff format
      timeout_seconds: 60
```

`event` 和 `action` 必填，`name` 可省略；显式名称在同一来源内必须唯一。`if` 省略时无条件触发。YAML 使用严格校验：重复键、未知字段、未知事件、混合逻辑或非法动作只会禁用该配置来源并显示诊断，不影响另一个来源和 Agent 主流程。

## 事件

| 层级 | 事件 | 主要载荷 |
| --- | --- | --- |
| 应用 | `app_start`、`app_stop` | `/app/workspace`、`/app/outcome` |
| 会话 | `session_start`、`session_end` | `/session/conversation_id`、`/session/outcome` |
| 轮次 | `turn_start`、`turn_end` | `/turn/message_count`、`/turn/outcome` |
| 消息 | `pre_model_request`、`post_model_response` | `/message/round`、`/message/outcome`；完成响应还可含 `/message/text`、`/message/tool_call_count` |
| 工具 | `pre_tool_use`、`post_tool_use` | `/tool/id`、`/tool/name`、`/tool/arguments`、`/tool/target`、`/tool/read_only`；后置事件另含 `/tool/outcome`、`/tool/summary`、`/tool/duration_seconds` |
| 上下文 | `pre_compact`、`post_compact` | `/compact/trigger`、`/compact/outcome` |
| 上下文 | `context_cleared` | `/context/conversation_id`、`/context/outcome` |

每个载荷顶层都包含 `/event`。失败、拒绝和取消通过 `outcome` 表达，不另造事件名。`context_cleared` 表示 `/clear` 已完成，不会伪装成新会话。

## 条件

`if` 必须且只能选择 `all` 或 `any`，列表不可嵌套。`field` 使用 JSON Pointer；每个谓词必须且只能选择一种匹配器：

```yaml
if:
  any:
    - {field: /tool/name, exact: dangerous_tool}
    - {field: /tool/arguments/command, regex: "(?:rm|del) .+"}
    - {field: /tool/arguments/path, glob: "secrets/**"}
    - {field: /tool/read_only, exact: true, not: true}
```

- `exact`：类型和值都相同才匹配。
- `glob`：对完整字符串匹配 `*`、`?` 和字符类。
- `regex`：对完整字符串匹配正则表达式。
- `not: true`：反转已有匹配结果；字段不存在时仍不匹配，避免缺字段意外触发拒绝。

## 动作

### 提示词

```yaml
action:
  type: prompt
  content: 修改 Python 文件后运行项目测试。
  once: true
```

内容是静态系统补充，不会再调用模型生成。`app_start`、`session_start` 和 `turn_start` 的提示词分别在对应作用域内持续注入；其他事件的提示词进入下一次模型请求。`/plan` 中只有静态提示词动作会运行。

### Shell 命令

```yaml
action:
  type: command
  command: ./scripts/check-hook.sh
  command_windows: powershell -File scripts/check-hook.ps1
  timeout_seconds: 60
  async: false
  once: false
```

命令通过系统 Shell 运行，当前事件 JSON 从 stdin 输入。Windows 优先使用 `command_windows`，否则使用 `command`。默认超时 60 秒，与 fakuiCode 普通命令工具一致；该值是当前项目运行经验的可调整默认值，不代表操作系统隔离或资源配额。

退出码 `0` 表示执行成功；退出码 `2` 将 stderr 的单行化文本作为有意拒绝原因；其他非零退出、超时和格式错误只生成脱敏诊断。成功命令可在 stdout 返回：

```json
{"decision":"deny","reason":"Protected branch","additional_context":"Use a feature branch."}
```

`decision` 只能是 `allow` 或 `deny`。多个同步 `pre_tool_use` 规则按声明顺序全部执行，任一 `deny` 即拒绝；`allow` 只表示 Hook 不反对，不能绕过计划模式、危险命令保护、路径沙箱或权限系统。

### HTTP

```yaml
action:
  type: http
  url: https://hooks.example.com/fakuicode
  headers:
    Authorization: "Bearer ${FAKUICODE_HOOK_TOKEN}"
  allowed_env_vars: [FAKUICODE_HOOK_TOKEN]
  include: [/tool/name, /tool/outcome]
  async: true
```

远端 URL 必须使用 HTTPS；`localhost`、`127.0.0.1` 和 `::1` 可使用 HTTP。URL 不允许内嵌凭据，客户端不跟随重定向。默认请求体只有 `event`、Hook `name` 和 `source`；`include` 才会按 JSON Pointer 显式加入事件字段。Header 中的 `${NAME}` 只能引用 `allowed_env_vars` 白名单，值不会出现在诊断或界面。

只有 `2xx` 响应会解析与命令相同的结构化结果；网络错误、非 `2xx`、非法 JSON 和超过 32 KiB 的响应都只记故障并继续 Agent。32 KiB 与项目现有内部结构化输出上限保持一致；若真实遥测表明不足，应连同测试一起调整。

### 子 Agent 占位

```yaml
action:
  type: agent
  prompt: 审查当前改动。
```

本期只验证和加载该动作；触发时记录 `unsupported_action` 诊断，不会启动子 Agent。真实执行将在 SubAgent 能力接入后实现。

## 执行与故障边界

Hook 是可信本地自动化和附加策略层，不是 fakuiCode 的强制安全边界。路径沙箱、危险命令保护、计划模式和权限系统仍负责不可绕过的约束；Hook 返回 `allow` 也不能放宽这些约束。

Hook 故障采用 fail-open：即使命中的 `pre_tool_use` 命令或 HTTP 检查发生超时、退出异常、网络错误或响应格式错误，也只记录脱敏诊断，随后继续执行 fakuiCode 原有的权限判断。只有 Hook 明确返回合法的 `deny` 决策或命令以退出码 `2` 给出拒绝原因时才会拦截。因此，不应把 Hook 作为保护敏感目录、危险命令或其他硬安全要求的唯一防线。

- `once: true` 目前只在当前 fakuiCode 进程内生效，不持久化。
- `async: true` 仅适用于命令和 HTTP；后台结果不能注入提示词或决定允许/拒绝。
- `pre_tool_use` 是拦截事件，禁止异步动作。拒绝会变成普通失败 `ToolResult` 回灌模型，让它调整方案。
- 同步规则保持 YAML 声明顺序；本期没有显式优先级。
- `/plan` 跳过命令、HTTP 和子 Agent 动作，Hook 不能扩大只读工具集合。
- Hook 故障不会中断 Agent。SQLite 时间线只记录空正文的 `hook_diagnostic` 和脱敏元数据：Hook 名称/来源、事件、动作、错误类别、退出或 HTTP 状态、耗时及是否后台。事件正文、工具参数、stdout/stderr、Header、提示词、环境变量值与密钥都不会进入诊断；界面对同类故障只提示一次。
