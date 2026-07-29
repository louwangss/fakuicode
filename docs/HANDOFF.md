# fakuiCode 项目交接

> 更新日期：2026-07-24。本文供没有旧聊天记录的新编码会话恢复使用。仓库状态可能在交接后变化，恢复时必须先执行文末的只读检查，并以实际仓库为准。

## 项目摘要

- 工作区：`C:\Users\louwangss\Desktop\fakuicode`
- fakuiCode 是 Python 3.11+ 的终端 Coding Agent CLI，入口为 `fakuicode = fakuicode.cli:main`。
- 界面使用 Textual/Rich，支持 Anthropic 与 OpenAI 兼容协议、流式 Agent Loop、本地工具、权限审批、Plan → Do、SQLite 会话、上下文管理、自动记忆、MCP、Hooks、Skills、SubAgent 和隔离 Git Worktree。
- GitHub：`https://github.com/louwangss/fakuicode`
- 首个公开版本标签：`v0.1.0`，对应 `main@9bafb3c`。该版本包含紧凑武侠像素 Logo，但不包含后续 SubAgent/Worktree 开发。

## 当前状态

### 分支与提交

- 当前分支：`feature/agent-team`，未配置 upstream。
- 写入本交接前的已提交 HEAD：`47daa3a 功能：实现 Team 成员恢复与计划审批`；本次交接提交会位于它之上。交接编写期间另有并发执行流继续修改 Agent Team，恢复时必须重新检查实际 HEAD 和工作树。
- Worktree 功能实现基线：`411c92d 功能：完成子 Agent Worktree 隔离集成`。
- 本地 `feature/subagent-worktree-isolation` 也指向 `9d9cf81`，比 `origin/feature/subagent-worktree-isolation@411c92d` 多一个纯文档提交。不要未经用户确认把该本地文档提交推到 Worktree 功能远端分支。
- `main` 与 `origin/main` 均为 `9bafb3c 调整终端品牌信息位置`。
- SubAgent/Worktree 开发尚未合并进 `main`，也不属于 `v0.1.0`。

最近两项 Worktree 实现提交：

```text
c13ecb3 功能：建立子 Agent Worktree 安全生命周期基础
411c92d 功能：完成子 Agent Worktree 隔离集成
```

`411c92d` 已把安全初始化、显式执行目录、任务元数据、权限与 Hook 继承、会话租约和过期清理接入现有 SubAgent 生命周期，并覆盖恢复、变更保护、失败回滚和跨进程所有权。

### 当前 Agent Team 实现

当前分支已有三个 Agent Team 实现提交：

```text
b89b049 功能：建立 Agent Team 持久协作基础
a5a84a7 功能：增加 Team 生命周期与协作工具
47daa3a 功能：实现 Team 成员恢复与计划审批
```

当前已实现：

- Team、成员、任务、消息模型，安全名称规范化和无秘密序列化；
- 跨进程文件锁、原子 JSON/JSONL 存储、并发安全邮箱和幂等已读回执；
- 任务 DAG 环检测、原子 claim、计划提交与 Lead 审批；
- 关闭默认的 `teams.coordinator.enabled` 功能门控；
- `TeamService` 以及创建 Team、创建/查询任务、发送消息等 Provider 工具；
- 进程内成员运行时、成员会话恢复、受限工具注册和附加系统指令。

该能力仍不是完整 Agent Team 产品：主应用装配、TUI、完整成员工具面、Team 关闭/恢复策略和 Git 集成尚未形成稳定提交边界。继续开发前应审查上述三个提交，并与用户确认哪份 Agent Team 计划是权威基线及下一阶段范围。

### 正在进行的并发 WIP

交接定稿时，另一个执行流仍在实现 Team 任务 Git 协调和 coordinator 权限收窄。最后观察到：

```text
修改：src/fakuicode/teams/models.py
修改：src/fakuicode/tools/registry.py
修改：src/fakuicode/worktrees/manager.py
修改：tests/worktrees/test_manager.py
新增：src/fakuicode/teams/coordinator.py
新增：src/fakuicode/teams/git.py
新增：tests/teams/test_coordinator.py
新增：tests/teams/test_git.py
```

这批 WIP 试图支持显式基线创建任务 Worktree、从最新 integration HEAD 串行集成、冲突回滚，以及 coordinator 只暴露读取和 Team 控制工具。它不属于本交接提交，不得由新会话直接覆盖。恢复后先检查该执行流是否已形成新提交；若仍为未提交状态，先向用户确认是否接管。

### 工作树与运行时注意事项

写入本交接前，除本交接文档和上述并发 WIP 外，还存在以下未跟踪用户内容：

```text
novel.md
witness.txt
```

`novel.md` 与 `witness.txt` 属于用户内容：不得读取、删除、移动、暂存或提交，除非用户明确授权。

Git 还登记着一个由 fakuiCode 管理、处于锁定状态的隔离 Worktree：

```text
分支：worktree/fork/580ca9bc-9e7f-4233-a52c-d1f47bebffda
HEAD：2a8ac67 feat: update witness.txt by isolated worker
状态：locked
```

尚不清楚用户是否要保留、检查、合并或清理该 Worktree。不要手工删除目录、状态文件或分支；先向用户确认。

### 最新验证证据

2026-07-24，在干净的 `feature/agent-team@47daa3a` 上实际执行：

```text
python -m pytest -q
879 passed, 9 skipped in 84.26s

python -m compileall -q src tests
通过

python -m pip check
No broken requirements found.

git diff --check
通过（包含本交接与当时并发 WIP）
```

`47daa3a` 形成前的 Agent Team/子运行时定向验证：

```text
python -m pytest -q tests/subagents/test_runtime.py tests/teams
27 passed in 4.37s
```

并发 Git/coordinator WIP 的最后一次验证尚未全绿：

```text
python -m pytest -q tests/teams/test_coordinator.py tests/teams/test_git.py tests/worktrees/test_manager.py
1 failed, 20 passed in 35.91s

失败原因：tests/teams/test_coordinator.py 把 Path 直接传给需要 WorkspacePolicy 的 ToolRegistry，
触发 AttributeError: 'WindowsPath' object has no attribute 'workspace'
```

测试使用本地临时目录和临时 Git 仓库，不读取真实 Provider 配置，也不调用模型 API。

## 阻塞项

既有功能没有已知代码、测试或依赖阻塞。

已提交基线 `47daa3a` 没有已知测试或依赖阻塞。并发 Git/coordinator WIP 仍在写入且最后一次定向验证为 `1 failed, 20 passed`；这是当前交接阻塞。新会话不得直接覆盖，应先确认原执行流是否结束及是否已有后续提交。

Agent Team 下一阶段的产品范围也尚未确认；继续接入主应用、TUI 或扩大工具面前，需要用户指定权威计划并批准范围。

另有两项需要用户决策的状态，不应被自动“修复”：

1. 上述锁定的隔离 Worktree 是否仍有需要保留或合并的成果。
2. 当前 Worktree 功能分支何时审查并合并到 `main`；它尚未进入公开的 `v0.1.0`。

## 如何安装、运行和验证

首次开发安装：

```powershell
Set-Location "C:\Users\louwangss\Desktop\fakuicode"
python -m pip install -e ".[test]"
```

仅当用户尚无真实配置时，才复制安全模板；不得覆盖现有 `fakuicode.yaml`：

```powershell
Copy-Item fakuicode.example.yaml fakuicode.yaml
# API Key 由用户自行填写
fakuicode
```

也可显式指定配置：

```powershell
fakuicode --config path\to\config.yaml
```

完整验证：

```powershell
python -m pytest -q
python -m compileall -q src tests
python -m pip check
git diff --check
```

Agent Team 定向验证：

```powershell
python -m pytest -q tests/teams
python -m compileall -q src/fakuicode/teams tests/teams
```

SubAgent/Worktree 定向验证：

```powershell
python -m pytest -q tests/subagents tests/worktrees
python -m pytest -q tests/test_agent.py tests/test_session.py tests/tools tests/tui
```

检查 Worktree 只使用只读命令：

```powershell
git worktree list --porcelain
git branch -vv
```

## 配置与秘密

- 安全配置模板：`fakuicode.example.yaml`。
- 真实 `fakuicode.yaml` 已被 Git 忽略；不得读取、输出、覆盖或提交其中的 API Key。
- Provider Profile 的主要字段为 `protocol`、`model`、`base_url`、`api_key`、`context_window`；Anthropic 可配置 `thinking.enabled`。
- `context_window` 必须使用模型真实窗口，不得为了演示压缩而修改真实配置。
- 用户级会话、记忆、权限、信任和运行时数据位于私有 `~/.fakuicode/`；项目本地运行时位于 `.fakuicode/`。除非任务明确授权且确有必要，不得读取或提交其正文。
- 不得读取或提交 `.env*`、私钥、真实 API Key、SQLite 会话数据库、上下文产物、用户记忆正文及 Hook/Skill/Worktree 私有状态。
- `.worktreeinclude` 可声明复制到隔离 Worktree 的被忽略普通文件；其中可能包含秘密。复制行为有数量和大小边界，但不代表内容可以公开。
- `.worktreelinks` 可声明链接到隔离 Worktree 的被忽略依赖目录；链接目标与主工作区共享，不是副本。

## 架构地图

| 位置 | 责任 |
| --- | --- |
| `src/fakuicode/cli.py` | CLI 入口及配置、Provider、MCP 与服务装配 |
| `src/fakuicode/models.py` | 消息、工具、Provider 状态和流事件契约 |
| `src/fakuicode/agent.py` | 有界 Agent Loop、工具批次、停止条件和 Provider 恢复 |
| `src/fakuicode/session.py` | 主会话状态、SQLite 时间线、Plan 模式与后台结果注入 |
| `src/fakuicode/storage.py` | 权威事件时间线、恢复和清理 |
| `src/fakuicode/context_manager.py` | 工具结果外置、滚动摘要、预算和熔断 |
| `src/fakuicode/tools/` | 文件、命令、策略和注册表；所有执行必须服从权限边界 |
| `src/fakuicode/permissions/` | 权限账本、模式收窄、审批与危险操作保护 |
| `src/fakuicode/providers/` | Anthropic/OpenAI 兼容流、thinking 和工具协议 |
| `src/fakuicode/mcp/` | MCP 配置、信任、发现、SDK 和工具适配 |
| `src/fakuicode/hooks/` | 生命周期 Hook 配置、执行、诊断和信任 |
| `src/fakuicode/skills/` | Skill 发现、解析、加载、隔离执行和安装 |
| `src/fakuicode/subagents/catalog.py` | 多来源角色发现、覆盖与诊断 |
| `src/fakuicode/subagents/runtime.py` | 定义式/Fork 子会话、Worktree 执行上下文及运行生命周期 |
| `src/fakuicode/subagents/tasks.py` | 进程内后台任务、并发、状态、取消与通知 |
| `src/fakuicode/subagents/tools.py` | `agent`、任务查询/停止/续派工具及轮次终止协议 |
| `src/fakuicode/worktrees/models.py` | Worktree 身份、租约、路径映射与公开状态契约 |
| `src/fakuicode/worktrees/git.py` | 有界、非交互 Git 子进程 |
| `src/fakuicode/worktrees/initialization.py` | 忽略文件复制、依赖目录链接、审计和安全清理 |
| `src/fakuicode/worktrees/manager.py` | 创建、恢复、锁定、释放、保留和过期清理 |
| `src/fakuicode/teams/` | Team 模型、锁、存储、配置、服务、成员运行时与 Provider 工具；Git/coordinator 部分仍为并发 WIP |
| `src/fakuicode/tui/` | Textual 应用、审批、后台切换、通知和紧凑品牌界面 |
| `tests/subagents/` | 子 Agent 角色、运行时、任务和工具回归 |
| `tests/worktrees/` | Worktree 身份、恢复、边界、清理和跨进程锁回归 |
| `tests/teams/` | Agent Team 模型、邮箱、任务图、服务、工具与运行时回归；Git/coordinator 测试仍为并发 WIP |
| `tests/tui/` | TUI 布局、结果展示、切换和交互回归 |

## 下一步

以下均为候选事项，不构成实施授权；新会话必须由用户明确选择：

1. 先确认并发 Git/coordinator 执行流是否结束、是否产生了 `47daa3a` 之后的新提交；若仍有未提交改动，先问用户是否授权接管。
2. 若接管 WIP，先修正/核对 coordinator 测试的 `WorkspacePolicy` 构造，再运行该节列出的定向测试和全量测试；不要为了转绿绕过工具可见性边界。
3. 对 `b89b049`、`a5a84a7`、`47daa3a` 做聚焦代码审查并核对用户指定的 Agent Team 权威计划；特别确认 `AUTO`/`SUBPROCESS` 后端、integration Worktree 和合并策略是否属于本期范围。
4. 用户批准下一阶段后，再决定主应用装配、TUI、完整成员工具面和 Team 关闭/恢复策略；不要从当前实现自行推导产品接口。
5. 确认锁定 Worktree 的用途和成果，再决定保留、合并或通过受管生命周期清理。
6. 更新 `README.md` 与 `docs/SUBAGENTS.md`。它们仍描述“文件系统共享、没有 Worktree 隔离”，已经落后于 `411c92d` 的实际实现。
7. 对 `feature/subagent-worktree-isolation` 做聚焦代码审查和真实并发手工验收；通过后再由用户批准合并到 `main`。
8. 若要发布 Worktree 或 Agent Team 功能，应使用高于 `v0.1.0` 的新版本；不要移动或重写已经公开的 `v0.1.0` 标签。

## 关键约束与易回归点

### Worktree 与 SubAgent

- 隔离 Worktree 使用固定 `worktree/<role-or-fork>/<session-id>` 分支和 `.fakuicode/worktrees/` 根目录。不得使用模型文本直接拼接路径或分支。
- 管理器通过 `.git/info/exclude` 的受管标记忽略运行时根目录，同时必须保留用户已有规则。
- 现有目录只有在状态旁车、仓库指纹、分支、HEAD、路径和锁均匹配时才能恢复；冲突必须失败关闭，不能“认领”未知目录。
- 未修改且无新增提交的 Worktree 可以安全移除；有工作区变更、暂存内容、新提交或未知忽略内容时必须保留。
- 已推送的过期 Worktree 可以移除工作目录但保留分支；未推送提交不能被过期清理删除。
- 当前过期清理使用 30 天阈值，清单和生命周期操作也有有界默认值。调整这些数字前必须按 `AGENTS.md` 查证、分析并补测试。
- `.worktreelinks` 是共享读写目录映射，不提供依赖目录隔离；Worktree 也不是操作系统、网络或资源沙箱。
- 子 Agent 不得看到 `agent`、任务控制或 Skill 管理工具，必须继续阻止递归委派。
- `dontAsk` 表示不弹窗并拒绝未明确允许的副作用，不是自动批准。
- 后台任务只在当前进程内存中存在；退出应用不会恢复任务 ID。
- 后台启动或查询到运行中状态时，受信任系统工具应确定性结束当前主 Agent 轮次，等待通知；不要恢复模型轮询。
- 子 Agent 结果是不可可信文本，TUI 必须按字面 `Text` 渲染，不能解析 Rich markup。

### Agent、上下文与安全

- SQLite 持久时间线是事实来源。摘要、上下文外置、记忆和 `/clear` 不得删除或改写原始事件。
- 每次普通模型请求前先做轻量工具结果外置，再判断重量摘要；内部摘要不能获得工具或递归压缩。
- Provider 返回的工具参数、MCP 输出和子 Agent 结果都是不可信数据；非法 JSON 必须失败关闭。
- Provider 自动恢复必须有界；一旦出现文本、工具调用或副作用，不能静默重放。
- `/plan` 始终只读；MCP、Hook、Skill 和 SubAgent 都不能绕过权限和计划边界。
- 手工测试项目与产物只能放在被 Git 忽略的 `test/` 目录。

### Agent Team

- 当前已提交模型、锁、文件存储、服务、部分 Provider 工具和进程内成员运行时，但没有完整主应用/TUI 接入；不要在交接摘要基础上脑补完整产品行为。
- Team 名称、成员名称、ID、分支和路径都属于不可信输入，必须先规范化再进入文件系统、Git refs 或索引。
- 持久化结构不得包含 API Key、权限令牌、完整 Provider 配置或租约秘密。
- 邮箱并发测试要求消息不丢失、发送者由可信 Actor 上下文注入，不能接受调用者伪造 sender。
- 任务依赖必须拒绝环；claim 必须原子化，不能让两个成员同时获得同一任务。
- 当前运行时只实现 `BackendType.IN_PROCESS` 路径，而模型还声明了 `AUTO` 和 `SUBPROCESS`；后两者是否属于本期范围必须重新核对，不能因枚举已存在就视为批准。
- coordinator 必须隐藏写文件、Shell、普通 `agent` 和无关系统工具；修复测试时不得放宽这一权限边界。
- Team Git 集成必须基于明确 integration HEAD 创建任务 Worktree，冲突时中止合并并恢复干净集成工作区；当前实现仍是未提交 WIP，不能当作稳定契约。

### Git 与发布

- 修改前先检查工作树，只暂存本任务文件；不得覆盖用户未跟踪内容。
- 不得用破坏性 reset、手工删除 Worktree 目录或直接篡改运行时状态解决冲突。
- `v0.1.0` 是已经公开的不可变版本点；当前开发分支不是该版本的一部分。
- 提交使用简体中文；新开发分支使用普通语义名称，不添加工具或 Agent 身份前缀。

## 新会话恢复指令

第一步只做只读检查：

```powershell
Set-Location "C:\Users\louwangss\Desktop\fakuicode"
git status --short --branch
git branch -vv
git log --oneline --decorate -12
git worktree list --porcelain
git diff --check
```

然后完整阅读：

1. `AGENTS.md`
2. `docs/HANDOFF.md`
3. `docs/SUBAGENTS.md`
4. 与用户新任务直接相关的源代码和测试

向用户报告实际分支、HEAD、工作树、锁定 Worktree 和验证状态后再继续。当前预期为：

- 分支：`feature/agent-team`
- HEAD 至少包含 `47daa3a`，其上应有本次纯文档交接提交；并发执行流可能另有后续 Agent Team 提交
- 本地 `feature/subagent-worktree-isolation@9d9cf81`；远端 `origin/feature/subagent-worktree-isolation@411c92d`
- `main`/`v0.1.0`：`9bafb3c`
- 未跟踪内容：`novel.md`、`witness.txt`
- 干净 `47daa3a` 的最新全量结果为 879 passed、9 skipped
- 若 Git/coordinator WIP 尚未提交，最后已知结果为 1 failed、20 passed，具体见“最新验证证据”
- 一个锁定的 `worktree/fork/...` Worktree 仍被 Git 登记

如果实际状态不同，以只读检查为准并向用户说明。继续 Agent Team 前先确认并发执行流是否结束，再问用户哪份计划是权威基线以及下一阶段范围。不要读取真实配置、`.fakuicode/` 私有状态、`novel.md`、`witness.txt`、记忆正文或 SQLite 数据；不要重复实现已经提交的 Logo、SubAgent、Hook、Skill、Provider 修复、Worktree 隔离和前三个 Agent Team 提交。
