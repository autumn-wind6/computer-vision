import argparse
import json
import os

import numpy as np

from src.analysis import (
    confusion_matrix,
    plot_confusion_matrix,
    plot_training_curves,
    save_misclassified_images,
    visualize_first_layer_weights,
)
from src.data_utils import load_fashion_mnist
from src.experiments import grid_search
from src.mlp import SimpleMLP
from src.train_eval import evaluate, save_model, train_model


def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("report", exist_ok=True)


def run_train(args):
    data = load_fashion_mnist(val_ratio=args.val_ratio, seed=args.seed)
    model = SimpleMLP(
        hidden_dim=args.hidden_dim,
        activation=args.activation,
        l2_lambda=args.l2,
        seed=args.seed,
    )
    history, best_val_acc, best_epoch = train_model(
        model=model,
        x_train=data["x_train"],
        y_train=data["y_train"],
        x_val=data["x_val"],
        y_val=data["y_val"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_decay=args.lr_decay,
        seed=args.seed,
    )
    print(f"Best val acc: {best_val_acc:.4f} (epoch {best_epoch})")

    test_loss, test_acc, y_pred = evaluate(model, data["x_test"], data["y_test"])
    print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

    save_model(
        model,
        "models/best_mlp.npz",
        norm_mean=data["norm_mean"],
        norm_std=data["norm_std"],
        best_val_acc=np.array(best_val_acc),
        best_epoch=np.array(best_epoch),
    )
    plot_training_curves(history, "outputs/training_curves.png")
    cm = confusion_matrix(data["y_test"], y_pred)
    plot_confusion_matrix(cm, "outputs/confusion_matrix.png")
    save_misclassified_images(
        data["x_test"], data["y_test"], y_pred, "outputs/misclassified_examples.png", max_samples=16
    )
    visualize_first_layer_weights(model.W1, "outputs/first_layer_weights.png")
    with open("outputs/metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_acc": float(best_val_acc),
                "best_epoch": int(best_epoch),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "hidden_dim": int(args.hidden_dim),
                "activation": args.activation,
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "lr_decay": float(args.lr_decay),
                "l2": float(args.l2),
                "seed": int(args.seed),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Artifacts saved to outputs/ and models/.")


def run_search(args):
    data = load_fashion_mnist(val_ratio=args.val_ratio, seed=args.seed)
    best_cfg, best_val_acc, _ = grid_search(
        x_train=data["x_train"],
        y_train=data["y_train"],
        x_val=data["x_val"],
        y_val=data["y_val"],
        activation=args.activation,
        epochs=args.search_epochs,
        batch_size=args.batch_size,
        lr_decay=args.lr_decay,
        seed=args.seed,
        out_csv="outputs/grid_search_results.csv",
    )
    print(f"Best cfg: {best_cfg}, best val acc: {best_val_acc:.4f}")


def build_parser():
    parser = argparse.ArgumentParser(description="Simple 3-layer MLP for Fashion-MNIST")
    parser.add_argument("--mode", choices=["train", "search"], default="train")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--search_epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr_decay", type=float, default=0.98)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        run_train(args)
    elif args.mode == "search":
        run_search(args)


if __name__ == "__main__":
    main()
