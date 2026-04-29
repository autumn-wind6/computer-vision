import numpy as np
from torchvision.datasets import FashionMNIST


def one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    y = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    y[np.arange(labels.shape[0]), labels] = 1.0
    return y


def load_fashion_mnist(val_ratio: float = 0.1, seed: int = 42):
    """
    通过 torchvision 下载并加载 Fashion-MNIST，返回 train/val/test 划分。
    train=True 为 60000，train=False 为 10000。
    """
    train_set = FashionMNIST(root="data", train=True, download=True)
    test_set = FashionMNIST(root="data", train=False, download=True)

    x_train_full = train_set.data.numpy().astype(np.float32).reshape(-1, 28 * 28) / 255.0
    y_train_full = train_set.targets.numpy().astype(np.int64)
    x_test = test_set.data.numpy().astype(np.float32).reshape(-1, 28 * 28) / 255.0
    y_test = test_set.targets.numpy().astype(np.int64)

    rng = np.random.default_rng(seed)
    indices = np.arange(x_train_full.shape[0])
    rng.shuffle(indices)

    val_size = int(x_train_full.shape[0] * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    x_train = x_train_full[train_idx]
    y_train = y_train_full[train_idx]
    x_val = x_train_full[val_idx]
    y_val = y_train_full[val_idx]

    # Use only train split statistics to normalize all splits.
    mean = np.mean(x_train, axis=0, keepdims=True)
    std = np.std(x_train, axis=0, keepdims=True)
    std = np.clip(std, 1e-6, None)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    data = {
        "x_train": x_train,
        "y_train": y_train,
        "y_train_onehot": one_hot(y_train),
        "x_val": x_val,
        "y_val": y_val,
        "y_val_onehot": one_hot(y_val),
        "x_test": x_test,
        "y_test": y_test,
        "y_test_onehot": one_hot(y_test),
        "norm_mean": mean.astype(np.float32),
        "norm_std": std.astype(np.float32),
    }
    return data


def batch_iterator(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True, seed: int = 42):
    n = x.shape[0]
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        yield x[batch_idx], y[batch_idx]
