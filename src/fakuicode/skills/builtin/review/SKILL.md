---
name: review
description: 只读审查当前工作树相对 HEAD 的已跟踪改动
fakuicode:
  invocation: manual
  visible-tools:
    - read_file
    - find_files
    - search_code
    - run_command
  execution: shared
---
请只读审查当前工作树相对 HEAD 的已跟踪改动，并读取理解这些改动所必需的相关代码和测试。优先报告会导致错误行为的缺陷、安全风险、兼容性或回归风险以及缺失的关键测试；按严重程度排序，给出准确的文件和行号。不要修改文件、暂存改动或创建提交。如果没有发现问题，请明确说明，并列出仍存在的验证盲区或残余风险。

用户补充参数：$ARGUMENTS
