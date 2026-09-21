# OpenJev

**基于 Qwen3.5-0.8B 的本地中文决策模型，支持全量微调、一键网页调用和结构化 HTTP 接口。**

OpenJev 将状态、问题和判断标准转换为分类、评分或是非判断，适用于意图识别、情感分类、工单分流、信息完整度检查和业务规则判断等场景。项目提供从命题设计、合成数据准备、训练、断点恢复到本地推理的完整工具链。

接口设计参考 [TypeSafe / Jev](https://docs.typesafe.ai/introduction)，支持 `Choice`、`Score`、`Noul` 三种判断方式。OpenJev 是独立实现，采用 **Jev 风格去掉 `confidence` 字段的兼容变体**，不保证与所有官方 SDK 直接兼容。

> **项目状态 · 2026-09-21**：本机已完成 500 个命题、5,000 条请求的数据准备，首次正式全量训练已启动；网页工具与接口已就绪。训练尚未完成正式评估，目前没有可报告的业务准确率。

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 全量微调 | 训练全部文本骨干与决策头，共 752,394,049 个参数，不使用 LoRA 或冻结文本层 |
| 三种决策类型 | 分类选择、等级评分、命题成立概率，共用一套输入输出结构 |
| 中文专项任务 | 包含情感分类、意图识别及多个领域的规则与事实判断 |
| 一键网页工具 | 双击启动、自动打开浏览器、加载完整微调断点，支持表单和 JSON 输入 |
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

当前安装脚本面向 **Windows x64、Python 3.11 和 NVIDIA CUDA 显卡**。已在 RTX 5060 Ti 16GB 上验证真实模型的全量更新和 8,192 token 输入流程；推理也支持 CPU。

```powershell
git clone https://github.com/meijustory123/openjev.git
cd openjev

py -3.11 -m venv .venv
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
```

脚本在项目内创建独立环境，安装固定版本的 PyTorch CUDA 和训练依赖。当前组合为 PyTorch 2.10.0 / CUDA 12.8、Transformers 5.5.0；依赖见 [requirements.txt](requirements.txt) 与 [requirements.lock.txt](requirements.lock.txt)。安装脚本默认使用阿里云 PyPI 镜像，具体下载来源与校验方式见 [本机全量微调与使用](docs/3、本机全量微调与使用.md)。

从 [Qwen 官方模型仓库](https://huggingface.co/Qwen/Qwen3.5-0.8B) 准备完整模型文件，放入项目根目录的 `Qwen3.5-0.8B/`，然后检查环境：

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


