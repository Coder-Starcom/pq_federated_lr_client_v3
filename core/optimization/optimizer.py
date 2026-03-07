"""
Optimizer Interface

Defines the abstract base class for all optimization algorithms.

Author: Saad
Project: PQ-Federated Logistic Regression
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class Optimizer(ABC):
    """
    Abstract base class for optimization algorithms.

    All concrete optimizers must implement the `step()` method.
    """

    def __init__(self, learning_rate: float) -> None:
        """
        Initialize optimizer.

        Parameters
        ----------
        learning_rate : float
            Positive step size for parameter updates.
        """
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")

        self._learning_rate = float(learning_rate)

    @abstractmethod
    def step(
        self,
        weights: np.ndarray,
        bias: float,
        weight_gradients: np.ndarray,
        bias_gradient: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Perform one optimization step.

        Parameters
        ----------
        weights : np.ndarray
            Current weight vector (1D).
        bias : float
            Current bias.
        weight_gradients : np.ndarray
            Gradient w.r.t. weights (same shape as weights).
        bias_gradient : float
            Gradient w.r.t. bias.

        Returns
        -------
        Tuple[np.ndarray, float]
            Updated (weights, bias).
        """
        raise NotImplementedError

    @property
    def learning_rate(self) -> float:
        """
        Return learning rate.
        """
        return self._learning_rate