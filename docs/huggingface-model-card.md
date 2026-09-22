---
license: apache-2.0
language:
  - zh
base_model: Qwen/Qwen3.5-0.8B
base_model_relation: finetune
tags:
  - openjev
  - qwen3.5
  - decision-making
  - text-classification
  - sentiment-analysis
  - intent-classification
  - full-finetune
  - safetensors
---

# OpenJev-Qwen3.5-0.8b

基于 Qwen3.5-0.8B 文本模型全量微调的中文决策模型。本仓库发布第 **1800 步（第 3 轮结束）** 的完整推理权重。

给模型一段内容和判断规则，即可完成分类、评分或是非判断，适用于情感分析、意图识别和客服分流等任务。

| 判断方式 | 返回内容 |
| --- | --- |
| Choice | 最匹配的选项与各选项概率 |
| Score | 按给定等级计算的分数与各等级概率 |
| Noul | 某个判断成立的概率 |

接口参考 Jev，由 OpenJev 独立实现，输出不包含 `confidence` 字段。最长完整输入为 8192 token。

## 如何使用

配套代码：[OpenJev-Kit](https://github.com/meijustory123/OpenJev-Kit)。

Windows 用户下载或克隆项目后，双击 **启动决策模型.cmd**。启动器会准备 Python 和依赖；本地没有完整模型时，自动从本仓库下载并显示进度，然后打开网页。

```powershell
git clone https://github.com/meijustory123/OpenJev-Kit.git
cd OpenJev-Kit
.\启动决策模型.cmd
```

支持 CPU 和兼容的 NVIDIA 显卡。公开下载无需 Hugging Face 账号或 API 密钥。

这是“文本骨干 + 共享决策头”的模型，按候选计算匹配得分，由程序组装 JSON。请使用 OpenJev 的推理代码；它不是直接生成聊天回复或 JSON 文本的普通聊天模型。

## 训练与评测

- 训练方式：全量微调，3 轮，共 1800 步，不使用 LoRA。
- 数据：50 个领域、500 个中文命题、5000 条合成请求；其中 4000 条训练、500 条验证、500 条测试。
- 同一测试集共 500 条请求、600 个判断项。下表为第 1800 步 GPU 结果与合成测试标注的一致率。

| 指标 | OpenJev 第 1800 步 | 官方 Jev 1.13.0 |
| --- | ---: | ---: |
| 综合判断一致率 | 72.52% | 94.33% |
| 分类一致率 | 73.90% | 95.98% |
| 评分等级一致率 | 69.00% | 89.50% |
| 明确是非判断一致率 | 75.65% | 99.13% |

结果以合成标注为参照，实际业务效果请使用自己的数据验证。详细评测与代码见配套项目。

## 文件说明

- `model.safetensors`、`config.json`：微调后的文本骨干。
- `decision_head.safetensors`、`decision_config.json`：决策头和训练配置。
- `tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`：分词器文件。
- `model-manifest.json`：文件大小、SHA-256 与本地安装路径。
- `COMPLETE`：完整检查点标记。

启动器会把骨干和分词器整理到本地 `outputs/decision-full/checkpoint-001800/` 下。不需要另外下载原始 Qwen 权重。本仓库不包含训练数据或优化器状态。

模型基于 Apache-2.0 授权的 Qwen3.5-0.8B，已进行全量微调修改，许可证见 `LICENSE`。
