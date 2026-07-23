# fakuiCode Skill

Skill 是目录型的可复用 Agent SOP。模型启动时只接收有界目录，需要时通过系统工具 `load_skill` 渐进加载完整指令；用户也可以直接输入同名斜杠命令。

## 安装公开 Skill

用户明确要求安装时，可以直接输入：

```text
把这个 skill 装下：https://www.skills.sh/anthropics/skills/frontend-design
```

也可以使用确定性的本地命令：

```text
/skills
/skills list
/skills install <url> [--skill <name>] [--global] [--preset instruction|read-only|coding] [--replace]
```

自然语言入口与斜杠入口共用同一个安装服务，不会让模型调用 `curl`、`npx` 或自行拼装下载命令。默认安装到 `<workspace>/.fakuicode/skills/<name>/`；只有 `--global` 才安装到用户目录。确认成功后立即热刷新命令补全，但不会自动激活、执行或暂存 Git。

第一版只接受以下 HTTPS 公共来源：

- `https://skills.sh/<owner>/<repo>/<skill>`
- 公开 GitHub 仓库 URL
- GitHub `tree/<ref>/<path>` URL

skills.sh URL 只用于确定 GitHub 仓库和 Skill 名称，不抓取网页，也不调用第三方 CLI。仓库存在多个候选时必须用 `--skill <name>` 明确选择；同层已有目录默认拒绝，`--replace` 才允许进入替换确认。

安装确认默认停在“取消”，并一次显示规范 GitHub 来源、固定 commit、Skill 子目录、目标路径、文件及总大小、许可证、脚本、专属工具、覆盖/遮蔽关系和工具预设。可选预设为：

- `instruction`：不向 Skill 暴露基础工具。
- `read-only`：`read_file`、`find_files`、`search_code`。
- `coding`：只读工具加 `write_file`、`edit_file`、`run_command`；`frontend-design` 默认建议此预设。

上游 `SKILL.md`、许可证、`scripts/`、`references/` 和 `assets/` 保持原字节。宿主另写 `.fakuicode/install.yaml`，记录原始 URL、规范来源、commit、上游指纹和用户确认后的有效配置。`allowed-tools` 只作为预览信息，绝不会直接授予权限；普通脚本不会自动注册为结构化工具。

安装使用 GitHub REST API 固定 commit 后逐文件下载目标子树，不下载整个仓库归档。文件数、单文件大小、路径穿越、符号链接、子模块、重定向和限流均按失败关闭处理。落盘采用同文件系统临时目录和原子切换；取消、校验失败、刷新失败或替换失败不会留下半安装目录，并会恢复旧版本。

安装确认只允许文件落盘，不等于信任或执行授权。远程用户级 Skill 也不会自动获得脚本注册信任；专属工具首次激活仍按完整内容指纹确认，后续调用继续经过 Plan、路径保护和 PermissionManager。Skill 脚本不具备 OS 沙箱。

## 目录与覆盖

```text
项目：<workspace>/.fakuicode/skills/<name>/SKILL.md
用户：~/.fakuicode/skills/<name>/SKILL.md
内置：fakuicode 包内 skills/builtin/<name>/SKILL.md
```

同名 Skill 按项目、用户、内置的顺序选择。最高优先级候选即使解析失败也会遮蔽低层同名包，避免错误配置静默执行另一份指令。单个项目或用户包失败只禁用自身并显示诊断；内置包失败会阻止 Agent 会话启动。

## `SKILL.md`

```yaml
---
name: test
description: 运行与当前改动相关的测试并总结结果
fakuicode:
  invocation: auto
  visible-tools:
    - read_file
    - run_command
  execution: isolated
  history-turns: 1
  profile: inherit
---
按照以下步骤处理 $ARGUMENTS。
```

- `name` 只允许小写字母、数字和单连字符分段，最长 64 个字符，并且必须等于目录名。
- `description` 是启动目录中的一句说明，最长 1024 个字符。
- `invocation` 为 `auto` 或 `manual`，默认 `auto`。`manual` 不进入模型目录，模型不能用 `load_skill` 调用，但斜杠命令始终可用。
- `visible-tools` 默认为空，只控制模型可见和执行层白名单；它不会跳过 Plan 模式、危险命令保护、路径边界或权限确认。
- `execution` 为 `shared` 或 `isolated`，默认 `shared`。
- `history-turns` 是独立模式带入的最近完整用户轮次数，默认 `0`。
- `profile` 是独立模式使用的已有 Profile 名，默认 `inherit`。共享模式不能设置非零历史或其他 Profile。

frontmatter、扩展字段和 YAML 键都严格校验。v1 只替换正文中的 `$ARGUMENTS`；正文没有占位符但用户传入了参数时，宿主会在末尾追加 `ARGUMENTS:` 数据块。

共享 Skill 的渲染后 SOP、参数、来源和包指纹会作为快照固定在系统补充上下文中，并在每轮请求及上下文压缩后重新挂载。重复调用同名 Skill 会原子替换；`/clear` 会清除全部激活。文件热更新不会偷偷替换已固定的 SOP；指纹变化会撤销专属工具并标记为过期，重新调用后才使用新版本。

独立 Skill 创建隐藏子会话，不出现在 `/sessions` 和 `/resume` 中。它只继承基础提示、当前项目指令、只读记忆快照、指定的最近完整轮次和目标 Skill，不继承父会话其他 Skill，也不提供 `load_skill`。最终模型回复直接作为摘要回流，不再调用第二个摘要模型。

## 专属 Python 工具

能力包可以增加严格 JSON 描述符和入口脚本：

```text
my-skill/
  SKILL.md
  tools/
    format_text.json
  scripts/
    format_text.py
```

```json
{
  "name": "format_text",
  "description": "格式化输入文本",
  "input_schema": {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": false
  },
  "entrypoint": "scripts/format_text.py"
}
```

描述符文件名必须等于 `name`，只允许上述四个字段，入口必须是包内直接的 `scripts/*.py`。运行时工具名为 `skill__my-skill__format_text`。进程使用当前 Python 解释器、固定工作目录和 `shell=False`；stdin 接收参数 JSON，stdout 必须是且只能是：

```json
{"output":"完整结果","summary":"紧凑状态摘要"}
```

非零退出、无效 JSON、取消和超时都会成为有界失败。脚本一律视为有副作用，不能声明只读，也不能在 `/plan` 中执行。

项目脚本首次激活时会显示包路径和能力摘要，并将完整包内容指纹保存到私有的 `~/.fakuicode/skill-trust.yaml`。任何文件或路径变化都会改变指纹；旧信任和旧权限目标不能复用。用户目录和内置包按来源信任，但工具调用仍经过普通权限系统。

信任只允许注册代码，不代表允许执行。Python 脚本没有 OS 级文件、网络或资源沙箱；只应安装和信任自己审查过的能力包。

## 热更新与排错

Skill 在启动、新建/恢复会话、切换 Profile、MCP 工具发现完成以及每次提交输入前刷新，不会在 Agent 工具循环中途改变工具集合。常见禁用原因包括重复 YAML 键、未知字段、目录名不一致、与核心命令冲突、未知工具、非法 JSON Schema、越界入口、符号链接或 Windows reparse point。

仓库提供两个可复制、但不随程序内置的示例：最小纯指令包 [`explain-change`](../examples/skills/explain-change/SKILL.md)，以及多轮共享会话示例 [`backend-interview`](../examples/skills/backend-interview/SKILL.md)。本期不包含市场分发、版本解析、嵌套独立 Skill、多语言脚本或子会话浏览 UI。
