import numpy as np

from src.data_utils import batch_iterator, one_hot


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def evaluate(model, x: np.ndarray, y: np.ndarray):
    probs, _ = model.forward(x)
    loss = model.loss(one_hot(y), probs)
    pred = np.argmax(probs, axis=1)
    acc = accuracy(y, pred)
    return loss, acc, pred


def train_model(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 0.1,
    lr_decay: float = 0.98,
    seed: int = 42,
):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = -1.0
    best_epoch = -1
    best_state = model.get_state()

    for epoch in range(epochs):
        cur_lr = lr * (lr_decay ** epoch)
        batch_losses = []
        total_correct = 0
        total_samples = 0
        for xb, yb in batch_iterator(x_train, y_train, batch_size=batch_size, shuffle=True, seed=seed + epoch):
            yb_onehot = one_hot(yb)
            probs, cache = model.forward(xb)
            batch_loss = model.loss(yb_onehot, probs)
            grads = model.backward(yb_onehot, cache)
            model.step(grads, cur_lr)
            batch_losses.append(batch_loss)

            batch_pred = np.argmax(probs, axis=1)
            total_correct += int(np.sum(batch_pred == yb))
            total_samples += xb.shape[0]

        train_loss = float(np.mean(batch_losses))
        train_acc = float(total_correct / total_samples)
        val_loss, val_acc, _ = evaluate(model, x_val, y_val)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(cur_lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = model.get_state()

        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"| lr={cur_lr:.5f} | train_loss={train_loss:.4f} "
            f"| train_acc={train_acc:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

    model.set_state(best_state)
    return history, best_val_acc, best_epoch


def save_model(model, path: str, **extra_arrays):
    payload = {
        "W1": model.W1,
        "b1": model.b1,
        "W2": model.W2,
        "b2": model.b2,
        "activation": np.array(model.activation),
        "hidden_dim": np.array(model.hidden_dim),
        "l2_lambda": np.array(model.l2_lambda),
    }
    payload.update(extra_arrays)
    np.savez(path, **payload)


def load_model(path: str):
    ckpt = np.load(path, allow_pickle=True)
    state = {
        "W1": ckpt["W1"],
        "b1": ckpt["b1"],
        "W2": ckpt["W2"],
        "b2": ckpt["b2"],
    }
    meta = {
        "activation": str(ckpt["activation"]),
        "hidden_dim": int(ckpt["hidden_dim"]),
        "l2_lambda": float(ckpt["l2_lambda"]),
    }
    extras = {}
    for key in ckpt.files:
        if key not in {"W1", "b1", "W2", "b2", "activation", "hidden_dim", "l2_lambda"}:
            extras[key] = ckpt[key]
    return state, meta, extras
