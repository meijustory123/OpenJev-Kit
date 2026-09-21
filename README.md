# OpenJev 本地决策训练准备

基于 Qwen3.5-0.8B 的中文决策模型：**全量微调文本骨干与决策头，不使用 LoRA，不输出 confidence**。支持 Choice、Score、Noul。采用 Jev 风格的输入输出结构，按用户要求省略置信度字段。

## 从这里开始

1. [数据生成计划](docs/1、决策训练数据生成计划.md)：直接交给后续 agent 执行。
2. [500 个中文命题](docs/2、500个中文命题.md)：50 个领域，含 25 题情感分类和 25 题意图识别，每题生成 10 条。
3. [本机全量微调与使用](docs/3、本机全量微调与使用.md)：环境、校验、训练、恢复、评估与服务。
4. [当前准备状态](reports/当前准备状态.md)：已运行检查和仍需完成的步骤。
5. [本轮实际生成说明](docs/4、实际生成执行说明.md)：按每个 subagent 两个领域分批生成全部数据。

机器读取 `data/topics.jsonl`。每组 25 题，共 20 组；目标 5,000 条请求、6,000 个问题。训练与新断点推理的输入上限为 **8,192 token**（每个候选的完整编码输入），超长报错。数据目录 `data/drafts`、`data/shards`、`data/prepared` 由后续流程创建。

```powershell
.\.venv\Scripts\python.exe -m scripts.check_environment
.\.venv\Scripts\python.exe -m scripts.validate_data --group 1
# 全部数据齐全后
.\.venv\Scripts\python.exe -m scripts.prepare_data
.\.venv\Scripts\python.exe -m scripts.preflight_data --prepared
.\.venv\Scripts\python.exe -m scripts.train --config configs/train.json
```

基础模型原目录只读；所有训练输出写入 `outputs/`。本轮正在生成正式 5,000 条数据，分工与进度记录在 `reports/generation/dispatch.json`，检查结果在 `reports/generation-audit.json`。完整模型训练尚未启动。
