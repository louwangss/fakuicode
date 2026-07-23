---
name: commit
description: 检查当前改动、分组暂存并创建一条简体中文 Git 提交
fakuicode:
  invocation: manual
  visible-tools:
    - read_file
    - run_command
  execution: shared
---
你正在执行一次 Git 提交工作流。先检查工作区和差异，只处理与当前任务相关的文件；保留并说明无关改动或未跟踪文件。运行与风险相称的验证，检查暂存差异中是否出现密钥，再创建一条清晰、原子的简体中文提交。不得推送远端。

用户补充参数：$ARGUMENTS
