# 中文文本分类：手写 Attention 与 Tiny Transformer

这个项目用 PyTorch 从零实现 Self-Attention、Multi-Head Self-Attention 和一个简化版 Transformer Encoder，并把它用于中文短文本情感分类实验。

项目重点是理解 Transformer 的核心结构，而不是追求很高的分类准确率。代码尽量保持小而直接，适合作为 Attention/Transformer 入门练习。

## 功能概览

- 字符级 tokenizer：将中文文本按单字切分并映射为 token id。
- Self-Attention：演示单头注意力的基本计算过程。
- Multi-Head Self-Attention：实现多头注意力的 Q/K/V 投影、分头计算和输出投影。
- Transformer Encoder Block：包含残差连接、LayerNorm、FFN 和 Dropout。
- TinyTransformerClassifier：基于 Encoder 的中文情感二分类模型。
- 数据生成脚本：内置少量正负样本，可生成训练集和验证集 CSV。
- 训练脚本：完成数据加载、模型训练和验证集准确率评估。

## 项目结构

```text
.
├── attention.py                    # 单头 Self-Attention 示例
├── multi_head_self_attention.py    # 多头 Self-Attention 实现
├── tiny_transformer.py             # Transformer Encoder Block 和分类模型
├── tokenizer.py                    # 字符级 tokenizer
├── dataset.py                      # 文本分类 Dataset
├── make_data.py                    # 生成示例中文情感分类数据
├── train.py                        # 训练入口
├── data/                           # 生成后的 CSV 数据目录
└── README.md
```

## 环境依赖

建议使用 Python 3.9+。项目依赖：

```bash
pip install torch pandas
```

如果你使用 CUDA 版本的 PyTorch，请按自己的显卡和 CUDA 版本从 PyTorch 官方安装命令安装。

## 数据格式

训练和验证数据是 CSV 文件，包含两列：

```text
text,label
这个产品质量不错,1
服务太差了,0
```

标签含义：

```text
1 = 正面
0 = 负面
```

`make_data.py` 会生成：

- `data/all.csv`
- `data/train.csv`
- `data/valid.csv`

注意：`.gitignore` 当前忽略了 `data/`，因此生成的数据默认不会提交到 Git。

## 使用方法

### 1. 生成数据

```bash
python make_data.py
```

脚本会把内置的中文评论样本按类别分层切分为训练集和验证集，并打印样本数量与标签分布。

### 2. 训练模型

```bash
python train.py
```

训练脚本会：

1. 读取 `data/train.csv` 和 `data/valid.csv`。
2. 用训练集文本构建字符级词表。
3. 将文本编码到固定长度 `max_len=64`。
4. 训练一个 2 层 Tiny Transformer 分类器。
5. 每个 epoch 输出训练 loss 和验证集准确率。

默认训练参数在 `train.py` 中：

```python
max_len = 64
batch_size = 32
num_epochs = 200
learning_rate = 2e-4
```

模型配置：

```python
dim = 128
num_heads = 4
ff_dim = 256
num_layers = 2
num_classes = 2
```

## 核心实现说明

### Self-Attention

`attention.py` 展示了最基础的自注意力计算：

```text
Q = XWq
K = XWk
V = XWv
Attention(Q, K, V) = softmax(QK^T / sqrt(d))V
```

输入和输出形状保持为：

```text
[batch_size, seq_len, dim]
```

### Multi-Head Self-Attention

`multi_head_self_attention.py` 将 embedding 维度拆成多个 head：

```text
[B, L, D] -> [B, H, L, head_dim]
```

每个 head 独立计算注意力，再拼回原始维度并通过输出线性层。

### Transformer Encoder Block

`tiny_transformer.py` 中的 `TransformerEncoderBlock` 包含：

- Multi-Head Self-Attention
- 残差连接
- LayerNorm
- 前馈网络 FFN
- Dropout

结构是：

```text
x -> attention -> residual + norm -> ffn -> residual + norm
```

### 文本分类

`TinyTransformerClassifier` 会先把 token id 转成 token embedding，再加上 position embedding。经过多层 Encoder 后，对序列维度做 mean pooling，最后用线性层输出二分类 logits。

## 注意事项

- 当前数据集很小，只适合验证流程和理解模型结构，不适合作为真实情感分类模型。
- `tiny_transformer.py` 和 `attention.py` 末尾包含简单的形状测试代码，直接运行文件时会打印输出 tensor 形状。
- `train.py` 从训练集构建 tokenizer，因此验证集中没有见过的字符会被映射为 `[UNK]`。
- 当前实现没有 attention mask，padding token 也会参与 mean pooling；如果要提升实验严谨性，可以进一步加入 padding mask。

## 可扩展方向

- 添加 attention mask，避免 `[PAD]` 参与注意力和 pooling。
- 增加模型保存与加载逻辑。
- 把超参数改成命令行参数。
- 使用更大的中文文本分类数据集。
- 增加测试用例，验证各模块输入输出形状。
- 对比 RNN、CNN、Transformer 在同一数据集上的表现。
