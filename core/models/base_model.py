"""
BaseModel

Abstract base class for all machine learning models
used in the federated learning framework.

Defines the minimum required interface for any
trainable model.

Author: Saad
Project: PQ-Federated Logistic Regression
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for machine learning models.

    Any concrete model (e.g., LogisticRegression)
    must implement all abstract methods defined here.
    """

    def __init__(self, n_features: int) -> None:
        """
        Initialize model.

        Parameters
        ----------
        n_features : int
            Number of input features.
        """
        if n_features <= 0:
            raise ValueError("n_features must be positive.")

        self._n_features = n_features
        self._weights = np.zeros(n_features, dtype=float)
        self._bias = 0.0

    # ==========================
    # Abstract Interface
    # ==========================

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute model predictions.

        Parameters
        ----------
        X : np.ndarray
            Input matrix of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples, 1) or (n_samples,)
        """
        raise NotImplementedError

    @abstractmethod
    def compute_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Compute scalar loss value.

        Returns
        -------
        float
        """
        raise NotImplementedError

    @abstractmethod
    def compute_gradients(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute gradients of loss with respect to
        weights and bias.

        Returns
        -------
        Tuple[np.ndarray, float]
            (weight_gradients, bias_gradient)
        """
        raise NotImplementedError

    # ==========================
    # Parameter Utilities
    # ==========================

    def get_parameters(self) -> Tuple[np.ndarray, float]:
        """
        Retrieve model parameters.

        Returns
        -------
        Tuple[np.ndarray, float]
            (weights, bias)
        """
        return self._weights.copy(), float(self._bias)

    def set_parameters(
        self,
        weights: np.ndarray,
        bias: float,
    ) -> None:
        """
        Set model parameters.
        """
        if weights.ndim != 1:
            raise ValueError("Weights must be a 1D array.")

        if weights.shape[0] != self._n_features:
            raise ValueError("Weight dimension mismatch.")

        self._weights = weights.astype(float).copy()
        self._bias = float(bias)

    def update_parameters(
        self,
        weight_gradients: np.ndarray,
        bias_gradient: float,
        learning_rate: float,
    ) -> None:
        """
        Update parameters using gradient descent.
        """
        if weight_gradients.shape[0] != self._n_features:
            raise ValueError("Gradient dimension mismatch.")

        self._weights -= learning_rate * weight_gradients
        self._bias -= learning_rate * bias_gradient

    # ==========================
    # Properties
    # ==========================

    @property
    def n_features(self) -> int:
        """
        Number of input features.
        """
        return self._n_features