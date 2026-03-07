"""
Logistic Regression Model

Author: Saad
Project: PQ-Federated Logistic Regression
"""

import numpy as np
from typing import Tuple

from core.models.base_model import BaseModel


class LogisticRegression(BaseModel):
    """
    Binary Logistic Regression implemented from scratch.

    Uses:
    - Sigmoid activation
    - Binary cross-entropy loss
    """

    def __init__(self, input_dim: int) -> None:
        """
        Initialize model.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        """
        super().__init__(input_dim)

        # Small random initialization (1D to match BaseModel)
        self._weights = np.random.randn(input_dim) * 0.01
        self._bias = 0.0

    # ==========================
    # Forward Pass
    # ==========================

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid predictions.
        """
        z = np.dot(X, self._weights) + self._bias
        return self._sigmoid(z).reshape(-1, 1)

    # ==========================
    # Loss
    # ==========================

    def compute_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        epsilon: float = 1e-15,
    ) -> float:
        """
        Compute binary cross-entropy loss.
        """
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        m = y_true.shape[0]

        loss = - (1 / m) * np.sum(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

        return float(loss)

    # ==========================
    # Gradient Computation
    # ==========================

    def compute_gradients(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for weights and bias.
        """
        m = X.shape[0]

        predictions = self.forward(X)
        dz = predictions - y_true  # (n_samples, 1)

        dW = (1 / m) * np.dot(X.T, dz).flatten()  # ensure 1D
        db = float((1 / m) * np.sum(dz))

        return dW, db

    # ==========================
    # Utility
    # ==========================

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid.
        """
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))