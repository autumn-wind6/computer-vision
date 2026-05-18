import math

import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def plot_training_curves(history: dict, out_path: str):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    if "train_acc" in history:
        plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy (train / val)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm: np.ndarray, out_path: str):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(10), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(10), CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_misclassified_images(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    max_samples: int = 16,
):
    wrong_idx = np.where(y_true != y_pred)[0]
    if wrong_idx.size == 0:
        return

    chosen = wrong_idx[:max_samples]
    cols = 4
    rows = math.ceil(len(chosen) / cols)

    plt.figure(figsize=(cols * 3, rows * 3))
    for i, idx in enumerate(chosen, start=1):
        plt.subplot(rows, cols, i)
        plt.imshow(x[idx].reshape(28, 28), cmap="gray")
        plt.title(f"T:{CLASS_NAMES[y_true[idx]]}\nP:{CLASS_NAMES[y_pred[idx]]}", fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def visualize_first_layer_weights(w1: np.ndarray, out_path: str, max_neurons: int = 25):
    k = min(max_neurons, w1.shape[1])
    cols = 5
    rows = math.ceil(k / cols)
    plt.figure(figsize=(cols * 2, rows * 2))

    for i in range(k):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(w1[:, i].reshape(28, 28), cmap="seismic")
        plt.axis("off")
        plt.title(f"N{i}", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
