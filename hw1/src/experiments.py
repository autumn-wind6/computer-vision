import csv
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.mlp import SimpleMLP
from src.train_eval import train_model


def grid_search(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    activation: str,
    epochs: int,
    batch_size: int,
    lr_decay: float,
    seed: int,
    out_csv: str = "outputs/grid_search_results.csv",
) -> Tuple[Optional[Dict[str, Any]], float, List[Dict[str, Any]]]:
    """在固定搜索空间上按验证集准确率选最优超参，结果写入 CSV。"""
    hidden_dims = [128, 256]
    lrs = [0.1, 0.01]
    l2_list = [0.0, 1e-4]

    rows: List[Dict[str, Any]] = []
    best_val_acc = -1.0
    best_cfg: Optional[Dict[str, Any]] = None

    for hidden_dim in hidden_dims:
        for lr in lrs:
            for l2_lambda in l2_list:
                model = SimpleMLP(
                    hidden_dim=hidden_dim,
                    activation=activation,
                    l2_lambda=l2_lambda,
                    seed=seed,
                )
                _, val_acc, best_epoch = train_model(
                    model=model,
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    lr_decay=lr_decay,
                    seed=seed,
                )
                row = {
                    "hidden_dim": hidden_dim,
                    "lr": lr,
                    "l2_lambda": l2_lambda,
                    "val_acc": val_acc,
                    "best_epoch": int(best_epoch),
                }
                rows.append(row)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_cfg = {
                        "hidden_dim": hidden_dim,
                        "lr": lr,
                        "l2_lambda": l2_lambda,
                        "best_epoch": int(best_epoch),
                    }

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["hidden_dim", "lr", "l2_lambda", "val_acc", "best_epoch"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return best_cfg, best_val_acc, rows
