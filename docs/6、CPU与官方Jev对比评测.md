# CPU 与官方 Jev 对比评测

本流程将本地完整微调检查点与官方 `jev-latest` 放在同一份固定测试集上比较。参考标签来自 `agent_synthetic`，所以这里的“准确率”指与合成标注的一致率，不等同于人工验证的业务准确率。

## 评测范围

- 数据：`data/prepared/test_requests.jsonl`，50 个测试命题、500 条请求、600 个独立问题。
- 问题类型：250 个 Choice、200 个 Score、150 个 Noul。
- 任务标签：460 条综合判断请求、30 条情感分类请求、10 条意图识别请求；后两类样本较少，不单凭本表断言专项能力。
- 本地检查点：第 600 步和第 1200 步，分别对应第 1、2 轮结束；在查看本次测试结果前确定。
- 本地推理：CPU、FP32 参数、每个进程 8 个计算线程、候选微批次 1，保留检查点的 8,192 token 上限。
- 官方模型：请求使用 `jev-latest`，以每次响应的 `model` 字段记录实际版本。[官方 API 文档](https://docs.typesafe.ai/api)
- 输入：两种后端均读取原始 `request`，不传入标准答案、审阅理由或其他标签信息。长文本不截断。

测试集按命题与训练、验证集分开。同一命题的场景不会跨集合，但各集合共享领域和数据生成规范，因此这不是完全陌生领域的泛化测试。

## 统一指标

| 指标 | 计算方法 | 数量 |
| --- | --- | ---: |
| Choice 一致率 | 返回的 `choice` 等于标注概率最高的唯一选项 | 249；排除 1 个标注并列项 |
| Score 等级一致率 | 预测与标注的最高概率等级相同；预测并列计为不一致 | 200 |
| Noul 明确判断一致率 | 标注 ≤0.1 为否、≥0.9 为是；预测 <0.5 为否、>0.5 为是，恰为 0.5 计为不一致 | 115；排除 35 个非明确标注 |
| 综合离散判断一致率 | 上述三个集合的正确数之和 / 564，不是三类百分比的简单平均 | 564 |
| Score MAE | 返回的 `score` 与标注概率加权分值的平均绝对误差 | 全部 200 |
| 归一化 Score MAE | 每个问题的分值绝对误差除以该题最高等级下标，再求平均 | 全部 200 |
| Noul MAE | 返回 `noul` 与标注 `noul` 的平均绝对误差 | 全部 150 |

报告还保存各类软标签交叉熵和概率平方误差。离散一致率越高越好，误差越低越好。

官方接口的概率和分值分别舍入。分布的概率和偏差仅允许落在每个候选 0.005 的累计舍入范围内，再统一归一化；评分误差使用实际返回的 `score`，不从舍入后的概率重新推算。官方 `confidence` 字段不参与指标。本地仍不输出该字段。

**解读限制：** 合成软标签的概率没有经过真实频率校准；官方响应中舍入成 0 的概率也会放大交叉熵（计算时下限为 `1e-12`）。因此不能仅依据软标签误差或交叉熵，断言某个模型的真实概率校准更好。所有非明确标注仍参与对应的软标签误差统计。

## 执行本地 CPU 评测

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark run --backend cpu `
  --checkpoint outputs/decision-full/checkpoint-000600 --threads 8 `
  --output outputs/benchmark-20260922/cpu-600

.\.venv\Scripts\python.exe -m scripts.benchmark run --backend cpu `
  --checkpoint outputs/decision-full/checkpoint-001200 --threads 8 `
  --output outputs/benchmark-20260922/cpu-1200
```

脚本直接把 FP32 检查点加载到 CPU，并检查全部参数的设备与精度，不经过 BF16 中转。没有量化、阈值调参或测试集再训练。

## 执行官方评测

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark run --backend official `
  --output outputs/benchmark-20260922/jev
```

脚本会提示输入 API 密钥且不回显，也可读取已有的 `TYPESAFE_API_KEY` 环境变量。密钥不写入结果或配置文件。请求发送至官方 HTTPS 地址；不跟随重定向。遇到限流、暂时性服务错误或网络错误时做有限次数重试。

每条完成的结果立即追加到 `predictions.jsonl`。重新运行相同命令会跳过已有结果；数据文件、样本列表、CPU 配置或检查点指纹不匹配时拒绝混用。API 别名可能随时间更新，续跑后应检查报告中的实际模型版本；跨版本结果应另开输出目录重测。

不要把 `--limit` 冒烟测试的输出目录用于正式评测。正式汇总要求每个模型均完成整份测试集，不会静默丢弃失败样本或只比较成功子集。

## 汇总报告

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark report `
  --runs outputs/benchmark-20260922/cpu-600 outputs/benchmark-20260922/cpu-1200 outputs/benchmark-20260922/jev `
  --output reports/2026-09-22-cpu-vs-jev.json
```

报告包含测试文件 SHA-256、全部样本 ID、检查点权重和配置指纹、官方实际版本、样本覆盖情况、指标定义和分项结果。原始响应保存在各自的本地输出目录，`outputs/` 不进入 Git。

评测记录的请求时间仅用于排查运行过程。本次本地 CPU 评测与训练、其他本地进程共享机器资源，官方 API 时间又包含网络往返且服务端硬件未知，不能据此比较两个模型的纯推理速度。

仓库默认不附带本机测试数据和模型权重。第三方复核需要取得相同文件，并先核对报告中的指纹。
