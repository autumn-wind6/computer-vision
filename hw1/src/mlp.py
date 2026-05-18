import numpy as np


class SimpleMLP:
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 128,
        output_dim: int = 10,
        activation: str = "relu",
        l2_lambda: float = 0.0,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation.lower()
        self.l2_lambda = l2_lambda

        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.01, size=(input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.W2 = rng.normal(0, 0.01, size=(hidden_dim, output_dim)).astype(np.float32)
        self.b2 = np.zeros((1, output_dim), dtype=np.float32)

    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(0.0, z)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-z))
        if self.activation == "tanh":
            return np.tanh(z)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def _activate_grad(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return (z > 0).astype(np.float32)
        if self.activation == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-z))
            return s * (1.0 - s)
        if self.activation == "tanh":
            t = np.tanh(z)
            return 1.0 - t * t
        raise ValueError(f"Unsupported activation: {self.activation}")

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def forward(self, x: np.ndarray):
        z1 = x @ self.W1 + self.b1
        a1 = self._activate(z1)
        z2 = a1 @ self.W2 + self.b2
        probs = self._softmax(z2)
        cache = {"x": x, "z1": z1, "a1": a1, "probs": probs}
        return probs, cache

    def loss(self, y_onehot: np.ndarray, probs: np.ndarray) -> float:
        eps = 1e-12
        ce = -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))
        l2 = 0.5 * self.l2_lambda * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2))
        return float(ce + l2)

    def backward(self, y_onehot: np.ndarray, cache: dict):
        x = cache["x"]
        z1 = cache["z1"]
        a1 = cache["a1"]
        probs = cache["probs"]

        batch_size = x.shape[0]
        dz2 = (probs - y_onehot) / batch_size
        dW2 = a1.T @ dz2 + self.l2_lambda * self.W2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self._activate_grad(z1)
        dW1 = x.T @ dz1 + self.l2_lambda * self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def step(self, grads: dict, lr: float):
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs, _ = self.forward(x)
        return np.argmax(probs, axis=1)

    def get_state(self):
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
            "activation": self.activation,
            "hidden_dim": self.hidden_dim,
            "l2_lambda": self.l2_lambda,
        }

    def set_state(self, state: dict):
        self.W1 = state["W1"].copy()
        self.b1 = state["b1"].copy()
        self.W2 = state["W2"].copy()
        self.b2 = state["b2"].copy()
