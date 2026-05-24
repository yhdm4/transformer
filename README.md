# 中文文本分类：手写 Transformer Encoder 实验

本项目使用 PyTorch 从零实现一个简化版 Transformer Encoder，并将其用于中文文本情感分类任务。

本项目的重点不是追求很高的准确率，而是通过一个完整的小项目理解 Transformer 的核心结构，包括：

- 字符级 tokenizer
- Embedding
- Positional Embedding
- Multi-Head Self-Attention
- Transformer Encoder Block
- Residual Connection
- LayerNorm
- FFN
- 文本分类训练流程

---

## 1. 项目目标

本项目目标是手写一个基于 Transformer Encoder 的中文文本分类模型。

输入是一段中文评论文本，输出是情感类别：

```text
1 = 正面
0 = 负面