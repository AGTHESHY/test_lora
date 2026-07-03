# LoRA / QLoRA / PEFT 学习项目

这是一个用 Jupyter Notebook 循序渐进学习 **参数高效微调 (PEFT)** 的中文项目,涵盖:

- **PEFT** 家族总览与 **LoRA** 原理及实战
- **QLoRA**(4-bit 量化 + LoRA)原理及实战
- 微调后的权重合并、推理与前后效果对比
- 从 SFT 衍生到 **强化学习 (RLHF / DPO)** 的概念桥梁(仅铺垫,动手实现放到下一个仓库)

## 运行环境

本项目在以下环境开发与验证:

- Windows 10/11
- NVIDIA RTX 5080,16GB 显存(**Blackwell 架构 / sm_120**)
- Conda(miniconda)

> **重要:Blackwell (RTX 50 系) 用户必看**
>
> RTX 50 系是 Blackwell 架构(计算能力 sm_120)。普通稳定版 PyTorch 常常没有编译对应 kernel,运行时会报
> `CUDA error: no kernel image is available for execution on the device`。
> 因此**必须安装 CUDA 12.8 (cu128) 的 PyTorch(≥ 2.7)**,并使用较新的 `bitsandbytes`(≥ 0.45,4-bit kernel 才支持 Blackwell)。
> 具体安装命令见 `notebooks/00_环境准备与GPU检测.ipynb`。

## 环境安装

遵循本机的 conda 约定,新建独立环境 `test-lora`,不污染系统 Python。

```bash
# 1) 创建并激活环境
conda env create -f environment.yml
conda activate test-lora

# 2) 安装 Blackwell 可用的 PyTorch(cu128)——注意这一步不能用普通 pip install torch
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# 3) 安装其余依赖
pip install -r requirements.txt
```

安装完成后,先运行 `notebooks/00_环境准备与GPU检测.ipynb` 验证 GPU 与 4-bit 是否可用。

## 阅读顺序

| 顺序 | Notebook | 内容 | 演示模型 |
| --- | --- | --- | --- |
| 0 | `00_环境准备与GPU检测.ipynb` | 环境安装、GPU 检测、Blackwell 注意事项 | - |
| 1 | `01_PEFT与LoRA原理与实战.ipynb` | PEFT 总览、LoRA 原理与微调 | Qwen2.5-1.5B-Instruct |
| 2 | `02_QLoRA原理与实战.ipynb` | 量化 / NF4 原理、4-bit QLoRA 微调 | Qwen2.5-7B-Instruct (4-bit) |
| 3 | `03_微调后_合并推理与评估.ipynb` | 适配器保存/加载、合并权重、前后对比 | 承接 01 |
| 4 | `04_衍生_从SFT走向强化学习.ipynb` | RLHF / DPO 概念桥梁,下一仓库预告 | 仅概念 |

## 显存与模型对照(16GB 参考)

| 方法 | 底座加载精度 | 可训练参数 | 大致显存 | 备注 |
| --- | --- | --- | --- | --- |
| 全量微调 7B | fp16/bf16 | 全部 | 远超 16GB | 单卡跑不动 |
| LoRA (1.5B) | bf16 | < 1% | ~5-8GB | 本项目 01 |
| QLoRA (7B) | 4-bit NF4 | < 1% | ~8-12GB | 本项目 02 |

## 目录结构

```
test-lora/
├── README.md
├── environment.yml
├── requirements.txt
├── data/
│   └── instructions.jsonl      # 小型中文指令样例数据
└── notebooks/
    ├── 00_环境准备与GPU检测.ipynb
    ├── 01_PEFT与LoRA原理与实战.ipynb
    ├── 02_QLoRA原理与实战.ipynb
    ├── 03_微调后_合并推理与评估.ipynb
    └── 04_衍生_从SFT走向强化学习.ipynb
```

## 说明

- 每个 notebook 顶部都注明「需先完成 `00` 的环境安装」。
- 为了在本机快速跑通,训练步数刻意设得很小,注释中说明了如何加大。
- 若 7B 模型下载不便,`02` 中提供了改用更小模型的开关。
