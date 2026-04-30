# Fashion-MNIST 三层 MLP（从零实现）

> 复旦大学大数据学院计算机视觉作业

这是一个从零实现的三层神经网络（MLP）作业项目，满足以下要求：
- 手写前向传播与反向传播
- 支持 `ReLU` / `Sigmoid` / `Tanh`
- 使用 `CrossEntropy + L2` 与 `SGD + 学习率衰减`
- 验证集保存最佳模型
- 输出测试集准确率、混淆矩阵、误分类样本、第一层权重可视化
- 输出 `metrics_summary.json` 

## 1. 环境准备

建议 Python 3.9+。

安装依赖：

```bash
pip install -r requirements.txt
```

## 2. 项目结构

```text
.
├─ main.py
├─ requirements.txt
├─ src/
│  ├─ data_utils.py
│  ├─ mlp.py
│  ├─ train_eval.py
│  ├─ experiments.py
│  └─ analysis.py
├─ models/
├─ outputs/
└─ report/
   └─ report.md
```

## 3. 完整训练

```bash
python main.py --mode train
```

### 第一步：网格搜索

```bash
python main.py --mode search
```

默认搜索空间：`hidden_dim ∈ {128, 256}`，`lr ∈ {0.1, 0.01}`，`l2_lambda ∈ {0, 1e-4}`（每组用 `--search_epochs` 轮，默认 8）。结果写入 `outputs/grid_search_results.csv`，按 **val_acc** 取最优一行的 `hidden_dim`、`lr`、`l2_lambda`。

### 第二步：完整训练

用上一步超参跑满 epoch，例如：

```bash
python main.py --mode train --hidden_dim 256 --lr 0.1 --l2 0.001 --epochs 20
```

（`--hidden_dim`、`--lr`、`--l2` 与 CSV 最优行一致；`--activation` 等与搜索时保持一致。）

训练完成后会生成：

- `models/best_mlp.npz` 验证集最优权重  
- `outputs/training_curves.png`、`outputs/confusion_matrix.png`、`outputs/misclassified_examples.png`、`outputs/first_layer_weights.png`  
- `outputs/metrics_summary.json`  

## 4. 代码说明

- `src/mlp.py`：模型定义与梯度计算核心
- `src/train_eval.py`：训练循环、验证选择最佳、保存模型
- `src/analysis.py`：混淆矩阵、误分类和权重可视化
- `src/data_utils.py`：自动下载 Fashion-MNIST 与划分数据
- `src/experiments.py`：超参数网格搜索，输出 CSV
