# OpenJev

**基于 Qwen3.5-0.8B 的本地中文决策模型。**

给模型一段内容和判断规则，它就能帮你做分类、评分或是非判断，适用于情感分析、意图识别、客服分流等场景。

项目支持全量微调、一键打开网页和 HTTP 接口调用，最长输入为 8,192 token。接口参考 [TypeSafe / Jev](https://docs.typesafe.ai/introduction)，由本项目独立实现，输出不包含 `confidence` 字段。

## 可以做什么

| 判断方式 | 用途 | 示例 |
| --- | --- | --- |
| 分类（Choice） | 从几个选项中选择最合适的一项 | 客户想退款、换货，还是咨询？ |
| 评分（Score） | 按给定等级打分 | 这条工单有多紧急？ |
| 是非判断（Noul） | 返回某个判断成立的概率 | 客户是否明确要求取消订单？ |

## 快速使用

下载项目：

```powershell
git clone https://github.com/meijustory123/OpenJev-Kit.git
cd OpenJev-Kit
```

在 **Windows 10/11 x64** 上，双击 **[启动决策模型.cmd](启动决策模型.cmd)** 即可：

1. 自动下载安装缺失的 Python 3.11 和依赖，首次安装需要联网。
2. 本地没有完整模型时，自动从 [Hugging Face](https://huggingface.co/cainai/OpenJev-Qwen3.5-0.8b) 下载第 1800 步模型，并显示进度。
3. 自动打开网页，加载模型。
4. 在网页填写内容和问题，或使用内置示例查看结果。

模型保存在 `outputs/decision-full/` 下。下载中断后再次双击可继续；已有完整模型时直接复用，无需登录 Hugging Face。只有想训练自己的模型时，才需要按下文准备训练数据。

使用结束后，点击网页“退出工具”，或双击 **[关闭决策模型.cmd](关闭决策模型.cmd)**。更多操作见 [网页使用说明](docs/5、一键网页工具.md)。

## 训练自己的模型

训练需要支持 CUDA 的 NVIDIA 显卡，训练后的模型也能用 CPU 推理。

先从 [Qwen 官方仓库](https://huggingface.co/Qwen/Qwen3.5-0.8B) 下载基础模型，放入 `Qwen3.5-0.8B/`。再按 [数据生成计划](docs/1、决策训练数据生成计划.md) 和 [500 个中文命题](docs/2、500个中文命题.md) 生成数据，保存到 `data/shards/`。

在项目目录运行：

```powershell
# 安装环境
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1

# 检查并整理数据
.\.venv\Scripts\python.exe -m scripts.validate_data --require-all
.\.venv\Scripts\python.exe -m scripts.prepare_data
.\.venv\Scripts\python.exe -m scripts.preflight_data --prepared

# 开始训练
.\.venv\Scripts\python.exe -m scripts.train --config configs/train.json
```

默认进行 **3 轮全量微调**，训练中会自动保存检查点，支持中断后继续。参数可在 [训练配置](configs/train.json) 中修改，详细步骤见 [训练与使用说明](docs/3、本机全量微调与使用.md)。

## 数据概况

当前已准备 **50 个领域、500 个中文命题、5,000 条请求**，包含情感分类、意图识别和多个领域的判断任务。

其中 4,000 条用于训练，500 条用于验证，500 条用于测试。同一命题的数据只会放入一个集合。

## 与官方 Jev 对比

**评测日期：2026-09-22。** 使用同一测试集的 **500 条请求、600 个判断项**，比较模型结果与测试集标注的一致程度。

第 **600、1200、1800 步**分别对应第 **1、2、3 轮**训练结束。

| 指标 | 第 1 轮（600 步） | 第 2 轮（1200 步） | 第 3 轮（1800 步） | 官方 Jev 1.13.0 |
| --- | ---: | ---: | ---: | ---: |
| **综合判断一致率** | **67.20%** | **72.52%** | **72.52%** | **94.33%** |
| 分类一致率（Choice） | 65.86% | 69.48% | 73.90% | 95.98% |
| 评分等级一致率（Score） | 61.00% | 70.50% | 69.00% | 89.50% |
| 明确是非判断一致率（Noul） | 80.87% | 82.61% | 75.65% | 99.13% |

完整结果见 [评测报告](reports/2026-09-22-checkpoints-vs-jev.json)，评测方法与复现步骤见 [评测说明](docs/6、CPU与官方Jev对比评测.md)。

## 接入其他程序

提供 `POST /v1/systemone` 接口，支持一次提交多个问题，并返回 JSON 结果。

可参考 [请求示例](examples/request.json) 和 [接口启动说明](docs/3、本机全量微调与使用.md)。

## 更多文档

- [数据生成计划](docs/1、决策训练数据生成计划.md)
- [500 个中文命题](docs/2、500个中文命题.md)
- [训练与使用说明](docs/3、本机全量微调与使用.md)
- [网页使用说明](docs/5、一键网页工具.md)
- [评测说明](docs/6、CPU与官方Jev对比评测.md)
