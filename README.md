# 中文文本分类：手写 Transformer 与 BERT 微调对比

这个项目包含两个版本的中文情感分类实现：

- 版本 A：手写 Tiny Transformer，用来学习 Attention 和 Transformer Encoder 的底层原理。
- 版本 B：基于 `bert-base-chinese` 的 BERT 微调，用来贴近真实工业项目流程。

推荐学习路径是先跑版本 A，理解 tokenizer、embedding、self-attention、multi-head attention、encoder block 和 classifier；再跑版本 B，对比预训练模型微调后的效果。

## 项目结构

```text
.
├── data/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── version_a_tiny_transformer/
│   ├── experiments/
│   │   └── tiny_transformer_config.yaml
│   ├── src/
│   │   ├── attention.py
│   │   ├── dataset.py
│   │   ├── evaluate.py
│   │   ├── model.py
│   │   ├── tokenizer.py
│   │   ├── train.py
│   │   └── utils.py
│   └── results/
├── version_b_bert_finetune/
│   ├── experiments/
│   │   └── bert_finetune_config.yaml
│   ├── src/
│   │   ├── dataset.py
│   │   ├── evaluate.py
│   │   ├── train.py
│   │   └── utils.py
│   └── results/
├── make_data.py
└── README.md
```

`results/` 是训练时自动生成的目录，用来保存模型、指标、错误样本和可视化图片。

## 环境依赖

基础依赖：

```bash
pip install torch pandas numpy
```

如果要保存 loss 曲线、准确率曲线和混淆矩阵图片，再安装：

```bash
pip install matplotlib
```

如果要运行 BERT 微调版，再安装：

```bash
pip install transformers accelerate
```

如果你的环境里是 `torch 2.0.x`，不要安装较新的 `transformers`，因为新版会要求更高版本的 PyTorch。可以使用兼容版本：

```bash
python -m pip install "transformers==4.30.2" accelerate
```

如果你想保留 `transformers>=5`，则需要升级 PyTorch：

```bash
python -m pip install "torch>=2.4" transformers accelerate
```

本项目也提供了三个依赖文件：

```bash
pip install -r requirements-a.txt
pip install -r requirements-b.txt
pip install -r requirements-b-torch20.txt
```

## 数据准备

数据文件使用 CSV 格式，至少包含两列：

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

生成示例数据：

```bash
python make_data.py
```

当前配置默认读取：

- `data/train.csv`
- `data/valid.csv`

如果你换成自己的数据，只要保持 `text,label` 两列即可。

## 版本 A：手写 Tiny Transformer

代码位置：

```text
version_a_tiny_transformer/
```

这个版本自己实现：

- `CharTokenizer`
- token embedding
- position embedding
- `SelfAttention`
- `MultiHeadSelfAttention`
- `TransformerEncoderBlock`
- `TinyTransformerClassifier`
- 训练、验证、保存指标和错误样本

运行训练：

```bash
python version_a_tiny_transformer/src/train.py
```

默认配置文件：

```text
version_a_tiny_transformer/experiments/tiny_transformer_config.yaml
```

核心参数：

```yaml
max_len: 64
batch_size: 32
epochs: 30
learning_rate: 0.0002
dim: 128
num_heads: 4
ff_dim: 256
num_layers: 2
```

输出结果：

```text
version_a_tiny_transformer/results/
├── best_model.pt
├── history.json
├── metrics.json
├── error_cases.csv
├── loss_curve.png
├── acc_curve.png
└── confusion_matrix.png
```

这个版本适合写原理说明，也适合面试时讲清楚 Transformer 的前向传播过程。

## 版本 B：BERT 微调

代码位置：

```text
version_b_bert_finetune/
```

这个版本使用 HuggingFace Transformers：

- `AutoTokenizer`
- `AutoModelForSequenceClassification`
- `bert-base-chinese`
- AdamW 微调
- 验证集评估、保存最佳模型、错误样本和曲线

运行训练：

```bash
python version_b_bert_finetune/src/train.py
```

默认配置文件：

```text
version_b_bert_finetune/experiments/bert_finetune_config.yaml
```

核心参数：

```yaml
model_name: bert-base-chinese
max_len: 128
batch_size: 16
epochs: 3
learning_rate: 0.00002
weight_decay: 0.01
```

输出结果：

```text
version_b_bert_finetune/results/
├── best_model/
├── history.json
├── metrics.json
├── error_cases.csv
├── loss_curve.png
├── acc_curve.png
└── confusion_matrix.png
```

HuggingFace 模型缓存默认写入：

```text
version_b_bert_finetune/hf_cache/
```

单独评估最佳模型：

```bash
python version_b_bert_finetune/src/evaluate.py
```

第一次运行 BERT 版时，`transformers` 会下载 `bert-base-chinese` 模型文件。没有网络时，需要提前把模型缓存好，或者把配置里的 `model_name` 改成本地模型目录。

如果你的终端设置了无效代理，会导致模型下载失败。PowerShell 中可以先临时清理代理再运行：

```powershell
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue
python version_b_bert_finetune/src/train.py
```

如果访问 `huggingface.co` 不稳定，可以在 `version_b_bert_finetune/experiments/bert_finetune_config.yaml` 中打开镜像配置：

```yaml
hf_endpoint: https://hf-mirror.com
```

## 两个版本的区别

版本 A 关注“原理”：

- 优点：能看懂并讲清楚 Transformer 的每一层。
- 缺点：数据少时效果有限，也没有预训练知识。

版本 B 关注“工程效果”：

- 优点：效果通常更好，更接近企业项目中的文本分类方案。
- 缺点：如果只会调用 API，不理解底层结构，面试解释会比较薄。

这个项目的完整价值在于把两者放在一起：A 版证明你懂原理，B 版证明你会做工业化微调实验。

## 后续可扩展方向

- 给 `data/` 增加正式的 `test.csv`，训练后统一做测试集评估。
- 把训练日志接入 TensorBoard。
- 增加 precision、recall、F1-score。
- 加入命令行参数覆盖配置文件。
- 增加模型推理脚本 `predict.py`。
- 用更大的真实中文评论数据集做对比实验。
- 在 README 中补充 Attention 公式和 BERT 微调原理图。
