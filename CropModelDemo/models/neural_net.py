"""
neural_net.py
-------------
A small feedforward neural network implemented from scratch using only
NumPy: forward pass, manual backpropagation, and gradient-descent updates.
No PyTorch / TensorFlow / scikit-learn model classes are used here -- this
is the actual learning algorithm, written out by hand, so you can see
exactly what "training a model" means at the lowest useful level.

Architecture: input -> hidden (ReLU) -> hidden (ReLU) -> output
Supports two output modes:
    - "regression": linear output, trained with mean-squared-error loss
    - "binary":     sigmoid output, trained with binary cross-entropy loss
"""

import numpy as np


class NeuralNetwork:
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output=1,
                 mode="regression", seed=42):
        assert mode in ("regression", "binary")
        self.mode = mode
        rng = np.random.default_rng(seed)

        # He initialization, good default for ReLU networks
        self.W1 = rng.standard_normal((n_input, n_hidden1)) * np.sqrt(2 / n_input)
        self.b1 = np.zeros((1, n_hidden1))
        self.W2 = rng.standard_normal((n_hidden1, n_hidden2)) * np.sqrt(2 / n_hidden1)
        self.b2 = np.zeros((1, n_hidden2))
        self.W3 = rng.standard_normal((n_hidden2, n_output)) * np.sqrt(2 / n_hidden2)
        self.b3 = np.zeros((1, n_output))

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _relu_deriv(x):
        return (x > 0).astype(float)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self._relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        if self.mode == "binary":
            self.out = self._sigmoid(self.z3)
        else:
            self.out = self.z3  # linear output for regression
        return self.out

    def compute_loss(self, y_pred, y_true):
        if self.mode == "binary":
            eps = 1e-8
            return -np.mean(
                y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
            )
        else:
            return np.mean((y_pred - y_true) ** 2)

    def backward(self, X, y_true, lr=0.01):
        m = X.shape[0]

        # Output layer gradient. For both MSE-with-linear-output and
        # BCE-with-sigmoid-output, dL/dz3 simplifies to (y_pred - y_true).
        dz3 = (self.out - y_true) / m
        dW3 = self.a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * self._relu_deriv(self.z2)
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self._relu_deriv(self.z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X, y, epochs=1000, lr=0.01, verbose_every=100):
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            losses.append(loss)
            self.backward(X, y, lr=lr)
            if verbose_every and epoch % verbose_every == 0:
                print(f"  epoch {epoch:5d}  loss {loss:.5f}")
        return losses

    def predict(self, X):
        return self.forward(X)

    def save(self, path):
        np.savez(
            path,
            W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3,
            mode=self.mode,
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        n_input, n_hidden1 = data["W1"].shape
        _, n_hidden2 = data["W2"].shape
        _, n_output = data["W3"].shape
        mode = str(data["mode"])
        net = cls(n_input, n_hidden1, n_hidden2, n_output, mode=mode)
        net.W1, net.b1 = data["W1"], data["b1"]
        net.W2, net.b2 = data["W2"], data["b2"]
        net.W3, net.b3 = data["W3"], data["b3"]
        return net
