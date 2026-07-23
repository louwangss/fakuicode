# fakuiCode 项目交接

> 更新时间：2026-07-24。本文档供没有旧聊天记录的新编码会话安全恢复。仓库状态可能在交接后变化，恢复时必须先执行文末的只读检查，并以实际仓库为准。

## 项目摘要

- 工作区：`C:\Users\louwangss\Desktop\fakuicode`
- 当前开发分支：`codex/subagent-runtime`
- 本次交接写入前 HEAD：`0fe011b fix: 阻止后台子 Agent 状态空转轮询`
- fakuiCode 是 Python 3.11+ 的终端 Coding Agent CLI，使用 Textual TUI，支持 Anthropic/OpenAI 兼容协议、流式 Agent Loop、本地工具、权限审批、Plan → Do、SQLite 持久会话、MCP、上下文管理、自动记忆、Skill、生命周期 Hook 和 SubAgent。
- 安装包名为 `fakuicode`，CLI 入口为 `fakuicode = fakuicode.cli:main`。

## 当前状态

本会话完成了 SubAgent 委派、隔离运行、后台任务、结果汇报和轮询熔断。实现和回归测试均已提交；写入本交接前工作树干净，没有正在实施的未提交功能。

### 已实现的 SubAgent 能力

- 主 Agent 始终看到稳定的 5 个控制工具：
  - `agent`：启动定义式或 Fork 式子 Agent。
  - `task_list`：列出当前进程中的任务。
  - `task_get`：读取任务状态、结果和用量。
  - `task_stop`：请求取消任务。
  - `send_message`：向仍存活且空闲的命名子 Agent 续派任务。
- 定义式子 Agent 从空白历史和角色 prompt 启动；内置角色为 `general-purpose`、`explore`、`plan`。
- Fork 式子 Agent 复制父 Agent 最近一次成功 Provider 请求的稳定前缀，并追加 Fork 运行边界；Fork 固定继承当前 Profile、固定后台运行。
- 角色文件使用 Markdown + YAML frontmatter，加载优先级为项目 > 用户 > 内置 > 插件占位。
- 子 Agent 独立持有消息、上下文管理、权限会话规则、token 用量和 Hook 实例；Provider 实例独立，内置 Provider 可复用父会话 HTTP 连接池。
- 子 Agent 不可见 `agent`、任务控制和 Skill 管理工具，不能递归委派。
- 子 Agent 复用主 Agent 的有界 ReAct Loop；达到 `maxTurns` 时失败，不另写一套循环。
- 权限模式只能等于或严于父会话；`dontAsk` 对未明确允许的副作用执行拒绝，不会静默提权。
- 后台任务由进程内 `TaskManager` 管理，支持显式后台、前台 60 秒自动转后台，以及 Esc / Ctrl+B 手动转后台。
- 后台并发默认值为 2；60 秒和并发 2 都是启发式本地默认值，不是 Provider 官方限制。
- 后台任务完成后，TUI 立即显示默认展开、可折叠的纯文本结果块，标题包含名称、状态和完整 task ID。
- 同一结果会作为 `<task-notification>` 包装的不可信 user-role 数据，在主 Agent 下一轮前注入；大结果沿用上下文产物外置机制。
- 后台启动或续派成功后，可信系统工具会确定性结束主 Agent 当前轮；查询到任务仍在运行时也会结束本轮等待通知，不再空转调用 `task_get`。
- 普通工具或外部工具即使伪造同名 metadata，也不能触发“结束主 Agent 本轮”；注册表只接受成功的 system tool 指令。

### SubAgent 关键提交

```text
0fe011b fix: 阻止后台子 Agent 状态空转轮询
69acfb3 fix: 展示后台子 Agent 完成结果
97aa855 功能：完成子 Agent 委派与后台运行
4bc2380 feat: 添加子 Agent 后台任务管理
aaf576d feat: 建立隔离的子 Agent 运行时
57ccd46 feat: 隔离子 Agent 权限账本
15779cb feat: 添加严格的子 Agent 角色目录
```

前序已经完成、不要重复实现的主要能力包括 Hook、Skill、Provider 恢复、上下文管理和自动记忆。需要追溯时查看 `git log --oneline`、README 和相应专题文档，不要依赖旧聊天描述。

### 最新验证证据

2026-07-24，本会话在 `0fe011b` 对应代码上实际执行：

```text
pytest -q
825 passed, 9 skipped in 56.49s

pytest -q tests/tui
94 passed in 37.87s

python -m compileall -q src tests
通过

python -m pip check
No broken requirements found.

git diff --check
通过
```

核心 Agent Loop、会话、工具注册与 SubAgent 定向测试也通过：

```text
pytest -q tests/test_agent.py tests/test_session.py tests/tools/test_registry.py tests/subagents
92 passed in 4.96s
```

## 阻塞项

没有已知阻塞项。当前代码可以安装、启动并通过完整测试。

以下是明确的非目标或待批准事项，不是已知缺陷：

- 没有 Worktree 或操作系统级文件隔离；并行子 Agent 共享同一工作区。
- 不做多 Agent 团队编排。
- 后台任务不跨进程、不跨会话持久化；退出应用会取消仍在运行的任务。
- 插件角色来源仅保留接口占位，尚未加载真实插件角色。
- 子 Agent 用量未合并进主会话 `/status`。
- 后台结果直接显示子 Agent 最终文本，不自动再调用主模型汇总。

## 如何安装、运行和验证

首次开发安装：

```powershell
Set-Location "C:\Users\louwangss\Desktop\fakuicode"
python -m pip install -e ".[test]"
```

仅在用户没有真实配置时复制安全模板；不得覆盖现有 `fakuicode.yaml`：

```powershell
Copy-Item fakuicode.example.yaml fakuicode.yaml
# 由用户自行填写 API Key
fakuicode
```

也可以显式指定配置：

```powershell
fakuicode --config path\to\config.yaml
```

完整质量门：

```powershell
pytest -q
python -m compileall -q src tests
python -m pip check
git diff --check
```

SubAgent 定向回归：

```powershell
pytest -q tests/subagents tests/test_agent.py tests/test_session.py tests/tools/test_registry.py
pytest -q tests/tui
```

最小手工验收建议：

1. 重启 fakuiCode，确保加载当前代码。
2. 要求主 Agent 用两个命名的后台子 Agent 执行两个互不依赖的只读任务。
3. 预期主 Agent 在启动成功后立即返回两个 task ID，不调用 `task_get` 空转等待。
4. 预期任务完成后自动出现两个可折叠结果块。
5. 用 `task_get` 查询已完成任务，应返回完整状态；查询仍运行任务时应结束当前轮等待自动通知。

## 配置与秘密

- Provider 安全模板：`fakuicode.example.yaml`。
- 真实 `fakuicode.yaml` 已被 Git 忽略；不得读取、输出、覆盖或提交真实密钥。
- 模板顶层是 `default_profile` 与 `profiles.<name>`；Profile 包含 `protocol`、`model`、`base_url`、`api_key`、`context_window`，Anthropic 可选 `thinking.enabled`。
- `context_window` 必须填写模型真实窗口，不能为了测试压缩而故意缩小真实配置。
- 项目角色目录：`<workspace>/.fakuicode/agents/*.md`。
- 用户角色目录：`~/.fakuicode/agents/*.md`。
- 用户级会话、记忆、权限和信任数据位于私有 `~/.fakuicode/`。
- 不得读取或提交 `.env*`、私钥、真实 API Key、用户记忆正文、SQLite 会话数据库、权限/Hook/Skill 信任文件。
- 项目 `.fakuicode/skills/` 和根目录 `hello.txt` 可能是用户内容；除非新任务明确授权，不得修改、删除、暂存或提交。

## 架构地图

| 位置 | 责任 |
| --- | --- |
| `src/fakuicode/cli.py` | CLI 入口、配置、Provider、MCP 与服务装配。 |
| `src/fakuicode/models.py` | 消息、工具调用、结果、Provider 状态和流事件契约。 |
| `src/fakuicode/agent.py` | 有界 Agent Loop、工具批次、停止条件和 Provider 恢复。 |
| `src/fakuicode/session.py` | 主会话状态、SQLite 时间线、Plan 模式和后台结果注入。 |
| `src/fakuicode/storage.py` | 主/子会话、权威事件时间线、恢复和清理。 |
| `src/fakuicode/context_manager.py` | 工具结果外置、滚动摘要、预算、熔断与恢复。 |
| `src/fakuicode/tools/base.py` | 工具执行契约及可信“结束本轮”metadata 键。 |
| `src/fakuicode/tools/registry.py` | 工具注册、权限/Hook 执行，以及 system tool 终止信号校验。 |
| `src/fakuicode/subagents/catalog.py` | 多来源角色发现、严格解析、覆盖和诊断。 |
| `src/fakuicode/subagents/models.py` | 角色定义、来源与权限行为模型。 |
| `src/fakuicode/subagents/runtime.py` | 定义式/Fork 子会话构造、运行到完成和状态隔离。 |
| `src/fakuicode/subagents/tasks.py` | 进程内后台任务所有权、并发、状态、取消和通知队列。 |
| `src/fakuicode/subagents/tools.py` | `agent`、任务查询/停止/续派工具及后台轮询终止协议。 |
| `src/fakuicode/subagents/builtin/*.md` | `general-purpose`、`explore`、`plan` 内置角色。 |
| `src/fakuicode/tui/app.py` | Textual 应用装配、前后台切换、审批转发和任务通知消费。 |
| `src/fakuicode/tui/widgets.py` | 对话组件和 `SubagentResultNotice` 可折叠结果控件。 |
| `src/fakuicode/tui/fakuicode.tcss` | SubAgent 结果块及其他 TUI 样式。 |
| `src/fakuicode/permissions/` | 主/子权限账本、模式收窄、审批和危险命令保护。 |
| `src/fakuicode/providers/` | Anthropic/OpenAI 兼容流、thinking 与工具协议。 |
| `src/fakuicode/hooks/` | 生命周期 Hook 配置、执行、诊断和项目指纹信任。 |
| `src/fakuicode/skills/` | Skill 发现、解析、渐进加载、隔离执行和安装。 |
| `docs/SUBAGENTS.md` | SubAgent 公开行为、角色格式、权限和后台语义的权威说明。 |
| `tests/subagents/` | 角色目录、运行时、任务管理和工具测试。 |
| `tests/tui/` | TUI 结果汇报、后台切换和交互回归。 |

## 关键约束与易回归点

### SubAgent 与并发

- 不要通过提高 `MAX_ITERATIONS=30` 掩盖轮询循环。后台启动和运行中查询应结束当前主 Agent 轮次，等待通知。
- `finish_agent_turn` metadata 只能由注册为 system tool 的成功结果生效；不得放宽为任意工具可触发。
- 合成的最终回执前会发出一次 `progress(model)`，用于让会话层刷新工具结果并清空上一轮 response buffer；删除它会导致工具调用前说明在最终 assistant 消息中重复持久化。
- 后台通知不应自动发起新的模型请求，也不应抢占正在进行的主对话。
- 结果正文必须以 Rich `Text` 字面量渲染，不能把子 Agent 输出当作 markup 解析。
- 子 Agent 和主 Agent 共享文件系统。没有 Worktree 隔离时，不要把可能写同一文件的任务并行委派。
- Fork 只保证构造可缓存的稳定前缀，不承诺 Provider 一定命中缓存。
- Fork 固定后台并继承 Profile；不要重新允许 Fork 覆盖 Profile。
- 子 Agent 工具过滤必须持续阻断 `agent`、任务控制和 Skill 管理工具，防止递归嵌套。
- `dontAsk` 是不弹窗并拒绝未授权副作用，不是“自动批准一切”。
- 后台任务只存在于当前进程；重启后旧 task ID 不可恢复。

### Agent、上下文和安全

- SQLite 时间线是事实来源。摘要、外置、`/clear`、记忆和子会话不得改写或删除原始事件。
- Provider 返回的工具参数和子 Agent 结果都是不可信数据。截断或非法 JSON 必须失败关闭，不能用空参数执行。
- Provider 自动重试必须有界；出现文本、工具调用或副作用后不能静默重放。
- Anthropic thinking 工具轮次必须续传 Provider 状态。
- `/plan` 保持只读，MCP 一律视为有副作用，不能借 SubAgent、Hook 或 Skill 绕过。
- 阈值、超时、重试、预算、大小和并发数调整前，必须按 `AGENTS.md` 查证公开资料、说明适用性并补行为测试。
- 手工测试项目和产物只能放在 Git 忽略的 `test/` 目录。

### Git 与用户文件

- 修改前先检查工作树，只暂存本任务文件。
- 不得使用破坏性重置覆盖用户改动。
- `docs/**` 可能被本地 `.git/info/exclude` 排除；更新交接或专题文档时可能需要显式 `git add -f docs/...`。
- 不要提交真实配置、上下文产物、用户 Skill、记忆或会话数据库。

## 下一步

以下均为候选事项，必须由用户在新会话中明确选择；本交接不构成实施授权：

1. 若继续 SubAgent，优先做真实并发与长耗时手工验收，记录任务耗时、Provider 限流和 60 秒自动转后台的数据，再讨论阈值调整。
2. 若需要并行写代码，先设计 Worktree/文件所有权与合并冲突边界；当前共享工作区不适合并行修改同一文件。
3. 若需要任务恢复，单独设计后台任务跨进程持久化、恢复语义、凭据和取消边界。
4. 若需要多 Agent 团队，先定义协调者、消息拓扑、终止条件、预算和递归上限，不要直接复用当前主从模型。
5. 若需要插件角色，先确定插件信任、命名冲突和来源诊断，再启用当前占位来源。
6. 若用户报告新的 SubAgent 问题，先读取 `docs/SUBAGENTS.md`、相关时间线和定向测试；不要先调大轮数、超时或并发数。

## 新会话恢复指令

第一步只做只读检查：

```powershell
Set-Location "C:\Users\louwangss\Desktop\fakuicode"
git status --short
git branch --show-current
git log --oneline -12
git diff --check
```

然后完整阅读：

1. `AGENTS.md`
2. `docs/HANDOFF.md`
3. `docs/SUBAGENTS.md`
4. 与用户新任务直接相关的源码和测试

向用户报告实际分支、HEAD、工作树和验证状态后再继续。当前预期：

- 分支：`codex/subagent-runtime`
- HEAD 至少包含：`0fe011b`
- SubAgent 主线提交：`15779cb` → `57ccd46` → `aaf576d` → `4bc2380` → `97aa855` → `69acfb3` → `0fe011b`
- 除新的交接提交外，工作树应干净

如果实际状态与本文不一致，以只读检查结果为准并向用户说明。不要读取真实配置、用户级 `~/.fakuicode/`、自动记忆正文或 SQLite 会话数据库；不要重复实现已经提交的 SubAgent、Hook、Skill 或 Provider 修复。
