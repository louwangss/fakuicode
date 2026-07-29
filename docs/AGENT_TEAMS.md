# Agent Team

Agent Team 把一次性的星型子 Agent 委派扩展为长期小组协作。Team 绑定创建它的 Lead 会话、Git 仓库和目标分支；成员拥有独立持久会话，任务、成员状态和邮箱统一保存在用户目录 `~/.fakuicode/teams/teams.sqlite3`。

## 工作流

1. 主 Agent 调用 `team_create` 创建 Team。
2. Lead 用 `team_task_create` 建立共享任务及 `blocked_by` 依赖。
3. Lead 用 `team_member_start` 创建成员并原子分配任务。
4. 成员使用 `team_message_send` 和 `team_inbox_list` 横向协作。
5. 要求计划审批的成员先调用 `team_plan_submit`。Lead 必须用匹配的 `request_id` 与 `revision` 调用 `team_plan_review`；批准后系统从同一持久会话恢复成员。
6. 写任务成员在自己的任务 Worktree 中提交并验证，然后调用 `team_task_complete` 登记确切提交和验证摘要。
7. Lead 调用 `team_integrate_task` 串行合入 Team 集成分支。冲突会执行 `git merge --abort`，保留任务 Worktree 并把任务标为 `integration_failed`。
8. 所有任务结束后，Lead 先调用 `team_finalize_prepare` 获取一次性令牌，再把令牌原样传给 `team_finalize`。系统仅在目标分支未漂移、主工作树干净且能够快进时交付。

共享任务支持 `team_task_create`、`team_task_get`、`team_task_list`、`team_task_update` 和 `team_task_delete`。更新与软删除只允许发生在 `pending` 状态，并在同一任务锁内检查 revision、下游依赖和 DAG 环。已有空闲成员可通过 `team_member_assign` 领取新任务，同时沿用原 conversation ID。

普通主 Agent 在创建 Team 前只看到 `team_create`；普通 SubAgent 看不到 Team 协作工具。Team 成员只能看到与其身份绑定的协作工具，模型不能伪造发件人或跨 Team 身份。

## 权限确认

`team_create` 仍按当前权限模式确认。创建或恢复 Team 后，host 会为当前会话签发仅绑定该 Team ID 的临时工作流能力；任务管理、进程内成员启动与续派、消息、计划审批、完成登记、Team 内部集成和最终交付准备不再逐项询问。

这份能力只作用于明确标记的 Team 协作工具，不会放宽成员继承的文件、Shell、网络或 MCP 权限，也不能覆盖 `/plan`、无效权限配置或显式拒绝规则。`team_finalize` 更新用户目标分支前仍需单独精确确认；显式 `subprocess` 启动同样不使用进程内工作流能力。能力只存在于当前权限会话，关闭会话时清除。

## 后端选择

`team_member_start.backend` 接受：

- `auto`：选择当前已配置的最高优先级后端，并在工具结果中返回 `requested_backend`、`selected_backend` 与 `selection_reason`。
- `in_process`：在当前进程的有界任务管理器中运行独立 Agent 会话。
- `subprocess`：保留给独立终端窗格运行器。当前主应用尚未配置该运行器；显式请求会返回错误，不会降级。

进程内成员自然结束后会标为 `idle` 并通知 Lead。后续 `team_member_resume` 会使用原 conversation ID 恢复上下文；向空闲进程内成员投递邮箱消息也会触发恢复。

### 关闭与取消边界

进程内成员采用协作式取消：关闭应用时先发送取消信号，尚未启动的任务立即记为 `cancelled`；正在运行的任务最多等待 1 秒。超过宽限期的任务会在结构化关闭报告中记为 detached，仍保持 `cancelling`，不会误报为已经终止。后台工作线程是 daemon，因此未响应的第三方调用不会继续阻塞 Python 解释器退出；如果调用稍后返回，会补做会话关闭。

Python 线程不能被安全强杀，真正的硬终止需要未来的 `subprocess` 后端提供进程隔离。这里的 1 秒是可通过运行数据调整的内部启发式默认值，不是照搬竞品：截至 2026-07-29，[Python 3.11 `concurrent.futures` 文档](https://docs.python.org/3.11/library/concurrent.futures.html#concurrent.futures.Executor.shutdown)明确 `shutdown(wait=False)` 也不会让存活任务随解释器直接退出；核对的 [Claude Code CLI reference](https://docs.anthropic.com/en/docs/claude-code/cli-usage) 与 [OpenAI Codex 文档](https://developers.openai.com/codex/)均未公开数值化关闭宽限期。后续应根据真实取消耗时分布验证该值，而不是无依据扩大等待时间。

## Coordinator 模式

Coordinator 必须同时打开两把锁：

```yaml
teams:
  coordinator:
    enabled: true
```

并在启动进程设置：

```powershell
$env:FAKUICODE_COORDINATOR = "1"
fakuicode
```

只开其中一项不会生效。启用后 Lead 仅保留读取工具以及明确的 Team 调度、审批和 Git 交付工具；`write_file`、`edit_file`、`run_command`、普通 `agent` 和无关系统工具均不可见、不可执行。Git 合并由固定参数的专用工具完成，避免通用 Shell 绕过 coordinator 边界。

## 持久化与并发

- Team 配置、成员、任务、依赖和邮箱以 SQLite 作为唯一权威状态；跨任务、成员和工作流消息的变更在同一事务中提交。
- 消息由 host 自动写入时间戳，默认未读；已读更新幂等，并与消息存在性在同一事务中校验。
- 首次启动会在进程间迁移锁下原子导入旧的 `<team-name>/config.json`、成员/任务 JSON 和邮箱 JSONL。导入成功后旧文件保持原样，作为只读备份；后续只读取 SQLite，不再回写旧格式。
- 名称经过严格规范化，持久化路径会拒绝目录穿越、符号链接和 Windows reparse point。
- 任务 claim、状态 revision 和 DAG 环检测在同一写事务内完成，避免两个成员同时领取同一任务。

## 当前非目标

本版本不提供跨机器 Team、成员间实时流式输出、优先级或 deadline 等复杂调度，也未启用独立终端子进程运行器。对 `subprocess` 的显式请求会安全失败，后续实现必须提供完整 worker 启动、邮箱唤醒、退出与恢复协议后才能启用。
