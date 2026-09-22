# OpenJev

**基于 Qwen3.5-0.8B 的本地中文决策模型，支持全量微调、一键网页调用和结构化 HTTP 接口。**

OpenJev 将状态、问题和判断标准转换为分类、评分或是非判断，适用于意图识别、情感分类、工单分流、信息完整度检查和业务规则判断等场景。项目提供从命题设计、合成数据准备、训练、断点恢复到本地推理的完整工具链。

接口设计参考 [TypeSafe / Jev](https://docs.typesafe.ai/introduction)，支持 `Choice`、`Score`、`Noul` 三种判断方式。OpenJev 是独立实现，采用 **Jev 风格去掉 `confidence` 字段的兼容变体**，不保证与所有官方 SDK 直接兼容。

> **项目状态 · 2026-09-22**：本机已完成 500 个命题、5,000 条请求的数据准备及 3 轮全量微调，网页工具与接口已就绪。第 600、1200 步完成全测试集 CPU 推理，第 1800 步完成 GPU 推理，并与官方 `jev-1.13.0` 对比；实测结果见下文。

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 全量微调 | 训练全部文本骨干与决策头，共 752,394,049 个参数，不使用 LoRA 或冻结文本层 |
| 三种决策类型 | 分类选择、等级评分、命题成立概率，共用一套输入输出结构 |
| 中文专项任务 | 包含情感分类、意图识别及多个领域的规则与事实判断 |
| 一键网页工具 | 自动安装缺失的 Python 3.11 和依赖，打开浏览器并加载完整微调断点，支持表单和 JSON 输入 |
| 本地 HTTP 接口 | 提供 `POST /v1/systemone`，便于业务程序接入 |
| 可恢复训练 | 保存完整权重、优化器、随机状态和数据游标，恢复时检查数据文件哈希 |
| 输入长度检查 | 每个候选的完整输入最多 8,192 token，超长报错，不静默截断 |

### 三种判断方式

| 类型 | 适用任务 | 返回内容 |
| --- | --- | --- |
| `choice` | 从互斥候选中选择最匹配的一项 | `choice` 与各候选的 `probabilities` |
| `score` | 按从低到高排列的等级评价 | `score`、`legend` 与各等级的 `probabilities` |
| `noul` | 判断一个命题是否成立 | `noul`，范围为 0–1 |

Score 的等级从 **0** 开始，最终分数是各等级概率的加权期望。Noul 返回命题成立的概率。输出不包含 `confidence`，但保留用于决策的概率分布。

## 如何工作

模型分别读取“状态 + 单个问题 + 单个候选描述”，由 Qwen3.5 文本骨干和共享评分头计算候选得分，再通过 softmax 得到概率。服务端据此组装 JSON 响应。

同一请求可以包含多个问题；每个问题独立处理，不读取其他问题的答案。训练使用软标签监督，不训练自回归解释文本或 JSON 生成。Qwen3.5 的视觉模块不参与本项目的纯文本任务。

## 快速开始

### 1. 准备环境

一键启动支持 **Windows 10/11 x64**，会自动准备缺失的 **Python 3.11** 和项目依赖，无需预先安装 Python。训练需要 NVIDIA CUDA 显卡，推理支持 CPU；已在 RTX 5060 Ti 16GB 上验证真实模型的全量更新和 8,192 token 输入流程。

```powershell
git clone https://github.com/meijustory123/openjev.git
cd openjev

powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
```

脚本优先复用现有环境；缺少 Python 时，将校验过的 Python 3.11.16 x64 安装到项目 `.runtime/`，再创建 `.venv`，无需管理员权限。首次安装 PyTorch 时，有可用 NVIDIA 驱动则安装 CUDA 12.8 版，否则安装 CPU 版；版本固定为 PyTorch 2.10.0、Transformers 5.5.0。仅使用网页时可直接双击启动器，它会执行同样的环境准备流程。

依赖见 [requirements.txt](requirements.txt)，原环境参考快照为 [requirements.lock.txt](requirements.lock.txt)。本次安装的实际版本记录在 `outputs/setup/requirements-installed.txt`。默认使用阿里云 PyPI 镜像，下载来源、校验方式和指定 CPU/CUDA 版本的方法见 [本机全量微调与使用](docs/3、本机全量微调与使用.md)。

需要训练时，从 [Qwen 官方模型仓库](https://huggingface.co/Qwen/Qwen3.5-0.8B) 准备完整模型文件，放入项目根目录的 `Qwen3.5-0.8B/`，然后检查 CUDA 训练环境：

```powershell
.\.venv\Scripts\python.exe -m scripts.check_environment
```

**克隆仓库不等于获得完整运行数据。** `.gitignore` 默认排除了基础模型、`.venv`、正式 JSONL 分片、准备后的数据和训练输出。首次使用需要自行准备模型、生成训练数据并得到完整微调断点；仓库中的命题、生成规范和代码用于完成这一流程。

### 2. 准备数据并训练

依据 [数据生成计划](docs/1、决策训练数据生成计划.md) 和 [500 个中文命题](docs/2、500个中文命题.md) 生成数据，每题一个分片，保存到 `data/shards/`。正式训练要求 500 题齐全。

```powershell
# 检查全部分片并生成训练、验证、测试集合
.\.venv\Scripts\python.exe -m scripts.validate_data --require-all
.\.venv\Scripts\python.exe -m scripts.prepare_data
.\.venv\Scripts\python.exe -m scripts.preflight_data --prepared

# 开始全量微调
.\.venv\Scripts\python.exe -m scripts.train --config configs/train.json
```

配置文件为 [configs/train.json](configs/train.json)。训练进度写入 `outputs/decision-full/metrics.jsonl`，最近完整断点路径写入 `outputs/decision-full/latest.txt`。

如果已有训练断点，应从断点恢复，或选择新的输出目录；脚本不会覆盖已有断点：

```powershell
$checkpoint = (Get-Content outputs/decision-full/latest.txt -Raw).Trim()
.\.venv\Scripts\python.exe -m scripts.train --config configs/train.json --resume "$checkpoint"
```

### 3. 双击打开网页

双击项目根目录的 **[启动决策模型.cmd](启动决策模型.cmd)**。

- 自动检查 Python 3.11 和依赖；缺失时下载安装，已有兼容环境直接复用。首次安装需联网，窗口会显示进度，失败日志保存在 `outputs/setup/`。
- 自动打开默认浏览器，并加载 `outputs/decision-full/` 中步数最大的完整微调断点。
- 没有断点时显示训练进度，首个断点保存后自动加载。
- 可填写表单，或使用完整 JSON 提交结构化状态和多个问题。
- 提供意图识别、情感分类、混合判断示例，以及结果复制、下载功能。
- 训练仍在运行时，自动模式优先使用 CPU；训练结束且显卡有足够空闲显存时优先使用显卡。
- 新断点保存后，可在网页点击“加载最新模型”切换。重复双击启动器会复用已有服务。

默认地址为 `http://127.0.0.1:8765`，端口被占用时自动选择空闲端口。点击网页“退出工具”或双击 **[关闭决策模型.cmd](关闭决策模型.cmd)** 可关闭服务；只关闭浏览器页面不会结束后台服务。关闭网页服务不影响训练。

完整说明见 [一键网页工具](docs/5、一键网页工具.md)。

## HTTP 接口

外部程序可使用独立 HTTP 服务。准备好完整微调断点后启动：

```powershell
$checkpoint = (Get-Content outputs/decision-full/latest.txt -Raw).Trim()
.\.venv\Scripts\python.exe -m scripts.serve --checkpoint "$checkpoint" --port 8000
```

服务默认绑定 `127.0.0.1`，提供 `GET /health` 和 `POST /v1/systemone`。网页工具使用自己的本机会话管理；上述独立服务用于程序接入。

请求示例：

```json
{
  "model": "openjev-qwen3.5-0.8b-v1",
  "state": "客户原本想退款，随后明确改口：不退款了，请换成大一码。",
  "questions": {
    "当前诉求": {
      "type": "choice",
      "instructions": "识别客户最后明确提出且未撤回的处理意图。",
      "criteria": {
        "换货": "明确要求更换商品或规格",
        "退款": "明确要求退回货款",
        "咨询": "仅询问条件，尚未提出办理请求"
      }
    },
    "要求退款": {
      "type": "noul",
      "instructions": "客户当前是否仍要求退款？"
    }
  }
}
```

`state` 和 `instructions` 支持字符串、对象或数组。响应顶层包含 `model`、`answers` 和 `usage`；`answers` 按原问题 ID 返回结果。请求也接受兼容别名 `jev-latest`，该别名不会调用官方 Jev 服务。

可以直接发送仓库中的 [混合请求示例](examples/request.json)：

```powershell
$body = Get-Content examples/request.json -Raw -Encoding utf8
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/systemone `
  -Method Post -ContentType 'application/json; charset=utf-8' `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) |
  ConvertTo-Json -Depth 20
```

字段约束见 [请求 Schema](schemas/request.schema.json) 与 [响应 Schema](schemas/response.schema.json)。

## 数据集设计

当前本机数据覆盖 **50 个领域、500 个中文命题**。每题包含 10 条独立请求，其中一条为混合请求，因此总计 **5,000 条请求、6,000 个独立问题**。

| 集合 | 命题数 | 请求数 | 问题数 |
| --- | ---: | ---: | ---: |
| 训练集 | 400 | 4,000 | 4,800 |
| 验证集 | 50 | 500 | 600 |
| 测试集 | 50 | 500 | 600 |

- 25 题情感分类、25 题意图识别，其余 450 题为综合判断。
- 每题固定覆盖明确匹配、易混类别、信息不足、否定或干扰、低/中/高评分、明确肯定、明确否定和混合判断。
- 按命题划分集合，同一命题的全部场景只进入一个集合。
- 长度按每条请求中最长的完整候选输入计算，包含任务前后缀、状态、问题和候选描述。


## 本地模型与官方 Jev 实测

**评测日期：2026-09-22。** 四组均完成同一固定测试集的 **500 条请求、600 个判断项**，覆盖 50 个领域、50 个未用于训练的命题。以测试集现有合成标注为参照，官方 Jev 也作为待评测模型计分。

第 600、1200 步分别对应第 1、2 轮结束，使用 **Intel Core Ultra 7 265K CPU、全部参数 FP32**。追加的第 1800 步对应第 3 轮结束，使用 **NVIDIA GeForce RTX 5060 Ti GPU、文本骨干 BF16、决策头参数 FP32、CUDA BF16 自动混合精度**。本地各组均设 8 个 CPU 计算线程、候选微批次 1，保留 8,192 token 输入上限，不截断长样本。官方请求使用 `jev-latest`，全部 500 条响应的实际版本均为 `jev-1.13.0`；本次 GPU 对比复用这些官方响应，未重复调用接口。

| 指标 | 第 600 步 · CPU FP32 | 第 1200 步 · CPU FP32 | 第 1800 步 · GPU BF16 | 官方 Jev · 1.13.0 |
| --- | ---: | ---: | ---: | ---: |
| **综合离散判断一致率 ↑** | **67.20%（379/564）** | **72.52%（409/564）** | **72.52%（409/564）** | **94.33%（532/564）** |
| Choice 选项一致率 ↑ | 65.86%（164/249） | 69.48%（173/249） | 73.90%（184/249） | 95.98%（239/249） |
| Score 最高概率等级一致率 ↑ | 61.00%（122/200） | 70.50%（141/200） | 69.00%（138/200） | 89.50%（179/200） |
| Noul 明确是非判断一致率 ↑ | 80.87%（93/115） | 82.61%（95/115） | 75.65%（87/115） | 99.13%（114/115） |
| Score 分值 MAE ↓（200 项） | 0.5358 | 0.3495 | 0.3657 | 0.1337 |
| 归一化 Score MAE ↓（200 项） | 0.2482 | 0.1659 | 0.1703 | 0.0626 |
| Noul 概率 MAE ↓（150 项） | 0.2961 | 0.1818 | 0.1950 | 0.1045 |

第 1200 步 CPU 相较第 600 步 CPU 的综合一致率提高 **5.32 个百分点**。第 1800 步 GPU 与第 1200 步 CPU 的综合一致率相同：Choice 多判对 11 项，Score 少判对 3 项，Noul 少判对 8 项，评分和概率误差也有所增大。官方 Jev 在表中各项仍领先，综合一致率高出第 1800 步 GPU **21.81 个百分点**。

**设备与精度差异：** 第 1800 步采用网页推理引擎的默认 GPU 精度，与前两组 CPU FP32 不同。这些结果描述各运行配置的表现，不能将变化完全归因于训练步数。

**指标口径：** 综合一致率为三类可判定项目的正确数合计除以 564。Choice 排除 1 个标注最高概率并列项；Noul 仅把标注 ≤0.1 或 ≥0.9 的 115 项纳入是非一致率，预测以 0.5 为分界，恰为 0.5 计为不一致，其余 35 项仍参与概率误差统计。Score 等级按概率最高项判断，预测并列计为不一致；分值 MAE 使用实际返回的 `score`，归一化误差再除以该题最高等级下标。官方 `confidence` 不参与指标。

**适用范围：** 标签来源为 `agent_synthetic`，上述结果表示与合成标注的一致程度，不直接等同于真实业务准确率。集合按命题分开，但共享领域与生成规范；测试集中的情感分类、意图识别请求分别为 30、10 条，专项样本量较小。本次未根据测试结果调参。

完整指标、测试文件 SHA-256、检查点指纹、GPU 配置及官方实际版本见 [四组机器可读评测报告](reports/2026-09-22-checkpoints-vs-jev.json)，原 [CPU 评测报告](reports/2026-09-22-cpu-vs-jev.json) 保留供追溯。复现命令、舍入处理和续跑方法见 [本地 CPU / GPU 与官方 Jev 对比评测](docs/6、CPU与官方Jev对比评测.md)。测试数据与权重未随仓库分发；本地逐条响应保存在 `outputs/benchmark-20260922/`。本次与其他本地任务共享机器资源，运行耗时不作为模型速度对比。

## 默认训练配置

| 项目 | 配置 |
| --- | --- |
| 基础模型 | Qwen3.5-0.8B 的文本骨干 |
| 训练参数 | FP32，全量更新 |
| 前向计算 | CUDA BF16 自动混合精度 |
| 优化器 | Adafactor |
| 骨干 / 决策头学习率 | `2e-5` / `1e-4` |
| 训练轮数 | 3，完整数据对应 1,800 次优化更新 |
| 梯度累积 | 8 个独立问题 |
| 候选微批次 | 1 |
| 激活检查点 | 开启 |
| 保存频率 | 每 100 步及每轮结束 |

训练先收集候选得分，再按相同随机状态逐候选重算并反向传播，以减少同时保留的计算图。该实现保留组内 softmax 损失与全量参数更新，代价是增加计算时间。

先在验证集选定断点，再评估测试集。评估脚本提供 Choice 选项一致率、Score 均值误差、Noul 概率误差及软标签交叉熵等指标，详见 [训练、恢复与评估说明](docs/3、本机全量微调与使用.md)。

## 项目结构

```text
openjev/
├── configs/                 # 训练配置
├── data/
│   ├── topics.jsonl         # 固定命题、领域与集合划分
│   ├── authoring/           # 数据创作源文件
│   ├── drafts/              # 草稿与软标签（本地生成）
│   ├── shards/              # 校验后的正式分片（本地生成）
│   └── prepared/            # 训练、验证、测试文件（本地生成）
├── docs/                    # 数据规范、训练与网页使用文档
├── examples/                # 接口请求示例
├── openjev/                 # 协议、模型、数据处理与网页后端
├── reports/                 # 环境检查与对比评测报告
├── schemas/                 # 请求、响应 JSON Schema
├── scripts/                 # 环境安装、校验、训练、评估与启动入口
├── tests/                   # 协议、数据、模型与服务测试
├── web/                     # 本地网页界面
├── outputs/                 # 训练断点与运行日志（本地生成）
├── 启动决策模型.cmd
└── 关闭决策模型.cmd
```

## 文档导航

- [决策训练数据生成计划](docs/1、决策训练数据生成计划.md)
- [500 个中文命题](docs/2、500个中文命题.md)
- [本机全量微调与使用](docs/3、本机全量微调与使用.md)
- [实际生成执行说明](docs/4、实际生成执行说明.md)
- [一键网页工具](docs/5、一键网页工具.md)
- [本地 CPU / GPU 与官方 Jev 对比评测](docs/6、CPU与官方Jev对比评测.md)


