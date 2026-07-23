# Fakuicode

Fakuicode 是一个 Python 3.11+ 的终端 Coding Agent。它提供 Textual TUI、SSE 流式输出、多轮会话、本地工具调用与会话持久化；支持 Anthropic Claude 和 OpenAI Chat Completions 原生工具协议。

## 安装与启动

```powershell
python -m pip install -e ".[test]"
Copy-Item fakuicode.example.yaml fakuicode.yaml
# 编辑 fakuicode.yaml，填写自己的 API Key
fakuicode
```

也可以使用其他配置文件：`fakuicode --config path\to\config.yaml`。

API Key 填写在 `fakuicode.yaml` 的每个 Profile 的 `api_key` 字段。该文件已被 Git 忽略，不应提交。

## Profile 与模型

配置支持多个 Profile，启动时使用 `default_profile`。运行中可以输入：

- `/model`：打开 Profile 选择器；切换后重建 Provider 并开始新会话。
- `/status`：显示当前 Profile、模型、会话编号、权限模式和项目信任状态。

旧版单 Profile YAML（`protocol`、`model`、`base_url`、`api_key` 位于顶层）仍然可用，会自动命名为 `default`。

Anthropic 可设置：

```yaml
thinking:
  enabled: true
```

Claude 模型使用 adaptive thinking；通过 Anthropic 兼容接口调用 DeepSeek 模型时，fakuiCode 会按 DeepSeek 协议映射为 `enabled` / `disabled`。不要配置 `thinking.budget_tokens`。OpenAI Profile 不支持 `thinking`。

## 会话与命令

会话自动保存到 `~/.fakuicode/conversations.sqlite3`。启动会恢复最近会话；原始记录始终保留，即使为了模型上下文生成了摘要。

- `/help`：显示命令帮助。
- `/new`：新建本地会话。
- `/clear`：清空当前内存上下文，不删除已保存的记录。
- `/sessions`（别名 `/session`）：列出本地会话。
- `/resume`：打开本地会话选择器。
- `/delete`：打开当前工作区的会话删除选择器并在确认后删除；会话用首次普通提问作为标题，旧默认标题会自动回填；仍兼容 `/delete <id>`。
- `/retry`：重新发送上一条用户请求。
- `/status`：显示当前状态。
- `/mcp`：只读显示 MCP Server 的连接状态、传输类型和工具数。
- `/model`：选择模型 Profile。
- `/memory`：查看自动记忆状态、当前用户/项目条目数、安全摘要和最近更新状态。
- `/memory on` / `/memory off`：开启或关闭后续轮次的记忆注入与后台维护；关闭不会删除已有笔记。
- `/memory forget`：打开当前用户与当前项目记忆选择器并在确认后删除；仍兼容 `/memory forget <uuid>` 精确删除。
- `/plan`：使用只读工具生成并暂存计划；完成后可在对话内直接选择执行。
- `/do`：有暂存计划时执行计划；尚无计划但处于 Plan 模式时退回默认执行模式。计划执行仍重新经过权限判断。
- `/permissions`（别名 `/permission`）：切换当前会话的权限模式或管理项目共享规则信任。
- `/skills` / `/skills list`：列出当前有效 Skill、来源和被禁用包的诊断。
- `/skills install <url> [--skill <name>] [--global] [--preset instruction|read-only|coding] [--replace]`：预览并安装公开 GitHub Skill；默认写入当前项目且不暂存 Git。
- `/commit`、`/review`、`/test`：内置 Skill。`review` 保持主会话只读审查，`commit` 只创建本地提交且不会推送，`test` 在隐藏子会话中运行并只回流摘要。

输入 `/` 可搜索本地命令和别名；固定参数命令（如 `/memory on|off|forget`）支持空格后的二级候选。方向键切换候选，Tab 只补全、不执行。状态栏使用 `[DEFAULT]` 和 `[PLAN]` 标明当前 Agent 模式。输入内容超过一行时自动换行，编辑框最多显示两行，更多内容随光标纵向滚动。Enter 发送；Esc 取消当前回合；Ctrl+C 或 Ctrl+Q 退出。向上滚动查看历史时，流式输出不会强制拉回底部；回到底部后会继续自动跟随。

### 可复用 Skill

Skill 把可重复的 SOP 保存为目录型能力包。项目、用户和内置目录按“项目 > 用户 > 内置”覆盖；启动时模型只看到有界的名称与一句说明，调用 `load_skill` 后才加载完整 SOP 和专属工具。共享 Skill 固定在当前会话上下文中；独立 Skill 使用隐藏的 SQLite 子会话运行并把最终回复回流主会话。详细格式、执行模式、脚本协议和安全边界见 [Skill 文档](docs/SKILLS.md)。

用户明确要求安装时，可以直接说“把这个 skill 装下：`https://www.skills.sh/anthropics/skills/frontend-design`”，也可以使用 `/skills install`。fakuiCode 会通过内置 `install_skill` 流程解析 skills.sh 或公开 GitHub 来源，固定到 commit、下载并校验目标子树，再一次性展示来源、许可证、文件、脚本、覆盖关系和工具预设。确认后默认安装到 `<workspace>/.fakuicode/skills/<name>/`，立即进入补全，但不会自动激活、执行或暂存 Git。

`visible-tools` 只收窄模型本轮可见和可执行的工具集合，不等于权限放行。项目 Skill 中的 Python 脚本必须按完整能力包指纹确认信任，之后每次调用仍需经过普通权限系统；脚本进程使用 `shell=False`，但不具备容器或操作系统级沙箱。

### 子 Agent

主 Agent 可以通过稳定的 `agent` 工具把边界明确的任务交给独立子 Agent。指定 `subagent_type` 时从空白上下文启动预定义角色；省略时从父 Agent 最近一次成功请求 Fork，并强制在后台运行。后台任务可通过 `task_list`、`task_get`、`task_stop` 和 `send_message` 查询、停止与续派；前台子 Agent 运行时按 Esc 或 Ctrl+B 可转入后台。角色格式、加载优先级、权限与缓存边界见 [SubAgent 文档](docs/SUBAGENTS.md)。

子 Agent 不会看到 `agent`、任务控制和 Skill 管理工具，因此不能递归委派。它拥有独立消息、权限决策状态、上下文管理和 token 计数，但共享 Provider 传输、文件系统与主 TUI 审批入口。后台任务完成时，TUI 会展示包含 task ID 的可折叠纯文本结果；同一结果在下一轮以不可信数据注入模型上下文，也可通过 `task_get` 读取，不会冒充 system 指令。

### 自动记忆与会话连续性

自动记忆默认开启。首次启动会说明：完成的执行轮次可由当前模型在后台提取精炼笔记，笔记只保存在本机 `~/.fakuicode/memory/`，可随时用 `/memory off` 关闭。用户级偏好与当前项目知识分开存放；项目身份按经验证的 Git common-dir 或非 Git 实际路径隔离，不会写入工作区或提交到 Git。

每个普通用户轮次开始时只加载一次有界索引，同一工具循环复用该快照。索引是不可信的辅助线索，不能覆盖当前请求、项目指令、权限、计划模式或当前文件证据；详情只能通过当前快照中的精确 UUID 读取。后台维护不拥有普通 Agent 工具，失败时会安全放弃且不影响前台回复或 SQLite 权威会话时间线。

`/resume` 选择器会直接从 SQLite 时间线计算可见消息数，不维护第二份计数。恢复已中断至少 24 小时的会话时，界面会提醒状态可能过时，下一普通请求也会要求模型重新验证关键事实；提醒只使用一次且不写入会话历史。

## 本地工具边界

模型可以请求六项基础本地工具；支持 Skill 的 Provider 还会看到系统级 `load_skill` 和仅用于用户明确安装请求的 `install_skill`，激活能力包后可能出现命名为 `skill__<skill>__<tool>` 的专属工具。`install_skill` 在 Plan 模式和隔离子会话中不可用：

- `read_file`：读取 UTF-8 文件，结果带行号。
- `write_file`：创建父目录并整体写入文件。
- `edit_file`：仅在原文片段唯一匹配时替换；失败会返回准确匹配数。
- `run_command`：以参数数组和 `shell=False` 执行命令，返回 stdout、stderr 与退出码。
- `find_files`：用 glob 模式查找文件，可限定相对路径范围。
- `search_code`：在文件中进行字面文本搜索，返回 `路径:行号:内容`，可限定相对路径范围。

所有内置文件工具限定在启动时的工作目录。路径会先解析既有符号链接和最近存在的祖先，再按路径层级检查是否仍在工作区；执行前还会复核冻结后的目标。`.git`、真实 `fakuicode.yaml`、`.env*`、权限配置、私钥与证书类文件不可被工具访问。

Agent Loop 最多运行 30 轮。连续的安全读取可以并发执行，有副作用的工具按顺序授权和执行；每个已宣布调用都会按原调用 ID 回灌一个结果。权限拒绝是一条普通的失败工具结果，不会直接终止循环，模型可以改用更安全的方案。

工具活动会显示为紧凑状态行；工具输出被截断，权限确认以内联选择列表显示在对话底部，只展示规范化目标，不显示文件内容、替换文本或搜索词。

## 权限系统

权限判断按以下边界和策略逐层进行：计划模式只读边界、不可配置的危险命令熔断器、路径与敏感文件沙箱、分层规则、当前权限模式，最后才是人工确认。任一硬拒绝都不能被规则、模式或确认覆盖。

三种权限模式为：

- `strict`：安全读取自动放行；没有显式 `allow` 的文件修改和命令直接拒绝，不弹窗。
- `default`：安全读取自动放行；没有规则结论的文件修改和命令请求确认。这是默认值。
- `trusted`：工作区文件写入和编辑自动放行；命令仍请求确认，所有 `deny` 与硬边界继续生效。

`/permissions` 中切换模式只影响当前 Agent 会话；新建、恢复会话或切换模型后恢复启动时加载的用户默认模式。`/plan` 始终只读，不属于权限模式，也不能被 `trusted` 放宽。计划完成后会出现“执行计划 / 暂不执行”选择；只有用户明确选择执行或输入 `/do` 才切换到执行模式，并重新经过权限判断。

需要确认时有四种选择：

- 拒绝：当前调用返回失败，Agent Loop 可以继续调整策略。
- 仅本次：只放行当前调用及其已冻结参数。
- 本会话：为当前规范化目标增加一条内存中的精确 `allow`。
- 永久：先把同一条精确 `allow` 原子保存到项目本地配置，保存成功后才执行。

自动生成的会话和永久规则不会扩大成 glob。权限弹窗中的“4. 仅拒绝此次调用”会让 Agent 继续当前任务；Esc 会拒绝此次调用并停止整个当前 Agent 任务。退出、切换会话和切换模型会清除会话放行。

### 配置文件

权限配置仅在应用启动时加载，本期不热重载：

- 用户全局：`~/.fakuicode/permissions.yaml`，可设置默认模式和全局规则。
- 项目共享：`<workspace>/.fakuicode/permissions.yaml`，可提交到版本控制，只能包含规则。
- 项目本地：`<workspace>/.fakuicode/permissions.local.yaml`，已被 Git 忽略，只能包含规则；“永久放行”写入这里。
- 项目信任：`~/.fakuicode/trusted-workspaces.yaml`，由 `/permissions` 维护，项目文件和模型工具不能自行建立信任。

用户全局配置示例：

```yaml
version: 1
mode: default
rules:
  allow:
    - run_command(git status)
    - read_file(src/*)
  deny:
    - run_command(git push *)
```

项目共享或项目本地配置不能包含 `mode`：

```yaml
version: 1
rules:
  allow:
    - run_command(python -m pytest *)
  deny:
    - write_file(dist/*)
```

规则严格使用 `工具名(模式)`。内置工具名必须是实际的 `read_file`、`write_file`、`edit_file`、`run_command`、`find_files` 或 `search_code`，不支持 `Bash` 别名；已发现的 MCP 工具使用界面显示的完整 `mcp__...` 名称。模式对完整规范化目标做精确或 glob 匹配，支持 `*`、`?` 和字符类；例如 `run_command(git *)` 匹配规范化后的完整 Git 参数序列。MCP 的会话/永久规则固定匹配 `__all_arguments__`，不会保存参数。文件规则匹配解析后的工作区相对 POSIX 路径，发现和搜索规则匹配搜索范围；文件内容、替换文本和搜索词不参与匹配。

规则优先级为：用户全局 `deny` 安全底线 → 当前会话 → 项目本地 → 项目共享 → 用户全局。项目未获信任时，共享 `deny` 仍生效，共享 `allow` 被忽略；只有用户通过 `/permissions` 明确信任后，共享 `allow` 才参与判断。同一来源内精确匹配优先 glob，同等级 `deny` 优先 `allow`。

YAML 使用严格校验：语法错误、重复键、未知字段、错误版本或非法规则都会使权限系统锁定为 `strict`，保留安全读取并拒绝副作用，界面会显示错误来源。信任文件损坏时则按项目未信任处理并显示警告。

## MCP 外部工具

fakuiCode 可在启动时通过 stdio 或 Streamable HTTP 发现外部 MCP Server 的工具，并以 `mcp__<server>__<tool>` 注册到现有工具中心。首次使用项目级 Server 时会先请求信任；工具调用仍经过权限确认，且不会在 `/plan` 中执行。

配置路径、可复制示例、覆盖规则、安全边界和故障排查见 [MCP 客户端文档](docs/mcp.md)。

## 生命周期 Hooks

fakuiCode 可以在应用、会话、轮次、模型消息、工具和上下文压缩节点运行声明式自动化。规则使用必填 `event` + `action` 和可选 `if`；支持静态提示词、Shell、HTTP 与尚未执行的子 Agent 占位动作。`pre_tool_use` 可返回拒绝并把原因作为失败工具结果交还模型，但不能绕过计划模式、危险命令保护或权限系统。项目 Hook 按独立文件内容指纹请求信任，Hook 自身故障只写脱敏诊断，不中断 Agent。完整事件、条件语法、动作协议和安全边界见 [Hooks 文档](docs/HOOKS.md)。

### 安全边界

危险命令熔断器只拦截直接可识别的一小组灾难操作，包括通用 Shell 入口、磁盘格式化/分区工具、块设备直写、关机重启，以及递归强制删除文件系统根目录、用户主目录或整个工作区。它不会把 `git push`、依赖安装、构建、`curl` 或项目内普通删除一概硬拒绝；这些操作由规则和人工确认控制。

路径沙箱只覆盖 Fakuicode 的内置文件工具以及内置 `ls` 兼容路径。`run_command` 使用 `shell=False`、固定工作目录、60 秒超时和输出上限，但它启动的进程及后代没有操作系统级文件或网络隔离。本期也不提供网络域名控制、资源配额、持久化审计日志或完整恶意命令检测；请不要把权限模式理解为容器或系统沙箱。

## 开发验证

```powershell
python -m pytest -q
python -m compileall -q src
python -m pip check
```

测试只使用本地 Mock Provider/HTTPX MockTransport，不会读取真实配置或访问模型 API。

协议实现参考：[Anthropic streaming](https://platform.claude.com/docs/en/build-with-claude/streaming)、[Anthropic tool use](https://platform.claude.com/docs/en/build-with-claude/tool-use)、[OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)。
