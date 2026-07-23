---
name: test
description: 选择并运行与当前改动相关的测试，然后总结结果
fakuicode:
  invocation: auto
  visible-tools:
    - read_file
    - find_files
    - search_code
    - run_command
  execution: isolated
  history-turns: 1
  profile: inherit
---
根据当前改动和带入的最近用户轮次，选择覆盖面足够且成本合理的测试。先检查项目已有测试方式，再运行相关测试；如失败，区分产品缺陷、测试缺陷和环境问题。最终只返回测试命令、通过/失败数量、关键失败原因和仍未覆盖的风险，不修改产品代码。

用户补充参数：$ARGUMENTS
