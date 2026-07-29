# SubAgent

SubAgent 用于把边界明确、可以独立完成的工作从主对话中隔离出去。主 Agent 始终只看到一组稳定工具：

- `agent`：启动定义式或 Fork 式子 Agent。
- `task_list`：列出当前进程内的任务。
- `task_get`：读取任务状态、结果与用量。
- `task_stop`：请求取消任务。
- `send_message`：向仍存活且空闲的命名子 Agent 续派任务。

## 两种启动方式

定义式调用提供 `subagent_type`，从空白消息历史启动，并把角色正文作为持续指令。内置角色是 `general-purpose`、`explore` 和 `plan`。

Fork 式调用省略 `subagent_type`。它复制父 Agent 最近一次成功 Provider 请求的消息前缀、稳定 system prompt、动态 supplement 与输出上限，再追加带运行边界的新任务。Fork 固定继承当前 Profile、固定后台运行，不能覆盖模型 Profile。

Fork 的目标是让稳定 system 和消息前缀具备 prompt-cache 复用条件，不承诺一定命中：Provider 的缓存策略、TTL、工具 schema 和上下文安全处理都会影响实际结果。为阻断递归，Fork 不继承 `agent`、任务控制和 Skill 管理工具；因此工具集合并非父请求的逐字节副本。

## 角色定义

项目角色放在 `<workspace>/.fakuicode/agents/*.md`，用户角色放在 `~/.fakuicode/agents/*.md`。文件必须由 YAML frontmatter 和 Markdown 正文组成：

```markdown
---
name: reviewer
description: 只读检查改动并报告风险
tools: [read_file, find_files, search_code]
disallowedTools: []
profile: inherit
maxTurns: 15
permissionMode: plan
background: true
---
你是只读审查子 Agent。基于实际代码和测试给出结论。
```

字段说明：

- `name`：必填，1–32 位小写字母、数字或连字符。
- `description`：必填，单行说明。
- `tools`：可选白名单；缺省表示不额外收窄。
- `disallowedTools`：可选黑名单，应用在白名单之后。
- `profile`：`inherit` 或现有 Profile 名称。
- `maxTurns`：1–30，缺省使用 Agent Loop 上限。
- `permissionMode`：`inherit`、`default`、`strict`、`trusted`、`dontAsk` 或 `plan`。
- `background`：是否固定后台运行。

加载优先级是项目 > 用户 > 内置 > 插件占位。同一来源重名、非法 YAML、未知字段或非法值会让该文件失效并显示诊断；用户或项目中的高优先级坏定义不会静默回退到同名低优先级角色。内置定义损坏会在启动时直接失败。

## 运行与权限

子 Agent 使用与主 Agent 相同的有界 ReAct Loop；模型不再调用工具时完成，达到角色 `maxTurns` 时失败。运行时分别持有消息、上下文管理、权限会话规则、token 用量和 Hook 实例。Provider 实例独立，但内置 Provider 可复用父会话 HTTP 连接池。

权限按以下顺序处理：

1. 父会话当前已有的显式精确放行规则可以被子 Agent 使用。
2. 子角色的模式只能等于或严于父模式，不能提权。
3. 仍需询问时，审批请求转发到主 TUI，并标注子 Agent 名称。

`dontAsk` 表示不弹审批框：未被现有规则明确允许的副作用会拒绝，而不是静默提权。`plan` 只允许只读工具。危险命令保护、路径边界、全局 deny 与 Hook 拒绝始终生效。

## 前台与后台

定义式子 Agent 默认前台运行。显式 `run_in_background: true` 或角色 `background: true` 会立即返回任务 ID；前台超过 60 秒会自动转后台，Esc 或 Ctrl+B 也可手动切换。60 秒只是面向终端交互延迟的启发式默认值，不是执行超时或安全边界；应根据真实等待时长和 Provider 延迟数据再调整。

后台并发默认上限为 2，同样是保守的本地资源默认值，不代表模型或 Provider 的官方限制。任务只在当前进程内存中存活，退出应用会请求取消全部任务。

后台任务启动或续派成功后，主 Agent 当前轮会立即以确定性回执结束，不再调用 `task_list` / `task_get` 轮询。手动查询到 `running`、`queued`、`waiting_approval` 或 `cancelling` 时也会结束当前轮，等待完成通知；这样不会用空转的模型请求消耗 Agent Loop 轮数。

任务完成后，TUI 会立即显示一个默认展开、可以用键盘折叠的结果块，标题固定包含子 Agent 名称、状态和完整 task ID；正文按纯文本显示，不能解释为 Rich markup。完整结果仍可通过 `task_get` 获取。

同一结果也会以 `<task-notification>` 包装成不可信的 user-role 数据，在主 Agent 下一轮请求前注入；大结果沿用上下文产物外置机制。完成通知本身不会额外调用模型，也不会抢占正在进行的主对话。

关闭后台功能时，定义式子 Agent 强制前台运行，Fork 返回结构化错误。

## 有意不做

- Worktree 只隔离 Git 写入，不提供操作系统级文件、进程或网络沙箱。
- 普通 `agent` 子 Agent 不允许递归委派；长期多成员协作由独立的 Agent Team 工作流提供，见 [Agent Team 文档](AGENT_TEAMS.md)。
- 不跨进程、跨会话持久化后台任务。
- 不加载真实插件角色，当前仅保留来源占位。
- 不把子 Agent 用量合并进主会话 `/status`。

## 设计依据

核对日期：2026-07-24。

- Anthropic 的公开示例把子 Agent 定义为 Markdown 角色，并强调独立指令、工具、上下文和并行执行：<https://platform.claude.com/cookbook/claude-agent-sdk-01-the-chief-of-staff-agent>
- OpenAI Codex 公开说明每个委派任务使用独立环境并支持并行、异步协作：<https://openai.com/index/introducing-codex/>
- OpenAI 的安全说明强调沙箱、审批策略与明确授权边界：<https://openai.com/index/running-codex-safely/>

这些公开资料没有给出适用于本地终端子 Agent 的自动转后台秒数或并发默认值，因此本项目没有照搬参考项目的 120 秒或任意并发数，而把 60 秒和并发 2 明确标为待实测的启发式默认值。
