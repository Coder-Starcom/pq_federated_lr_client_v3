"""
Gradient Descent Optimizer

Concrete implementation of basic gradient descent.

Author: Saad
Project: PQ-Federated Logistic Regression
"""

from typing import Tuple
import numpy as np

from core.optimization.optimizer import Optimizer


class GradientDescent(Optimizer):
    """
    Basic Gradient Descent optimizer.
    """

    def step(
        self,
        weights: np.ndarray,
        bias: float,
        weight_gradients: np.ndarray,
        bias_gradient: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Perform one gradient descent update.
        """

        if weights.shape != weight_gradients.shape:
            raise ValueError("Weight gradient dimension mismatch.")

        new_weights = weights - self._learning_rate * weight_gradients
        new_bias = bias - self._learning_rate * bias_gradient

        return new_weights, new_bias