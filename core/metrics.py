"""
Metrics Module

Provides evaluation metrics for binary classification.

Author: Saad
Project: PQ-Federated Logistic Regression
"""

import numpy as np


class Metrics:
    """
    Collection of static evaluation metric functions
    for binary classification.
    """

    # ==========================
    # Core Metrics
    # ==========================

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute classification accuracy.
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        if y_true.shape != y_pred.shape:
            raise ValueError("Shape mismatch between y_true and y_pred.")

        return float(np.mean(y_true == y_pred))

    @staticmethod
    def binary_cross_entropy(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        epsilon: float = 1e-15,
    ) -> float:
        """
        Compute binary cross-entropy loss.
        """
        y_true = y_true.flatten()
        y_prob = y_prob.flatten()

        if y_true.shape != y_prob.shape:
            raise ValueError("Shape mismatch between y_true and y_prob.")

        y_prob = np.clip(y_prob, epsilon, 1 - epsilon)

        loss = -np.mean(
            y_true * np.log(y_prob) +
            (1 - y_true) * np.log(1 - y_prob)
        )

        return float(loss)

    # ==========================
    # Confusion-Based Metrics
    # ==========================

    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute precision.
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))

        denominator = tp + fp
        return float(tp / denominator) if denominator > 0 else 0.0

    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute recall.
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        denominator = tp + fn
        return float(tp / denominator) if denominator > 0 else 0.0

    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute F1-score.
        """
        precision = Metrics.precision(y_true, y_pred)
        recall = Metrics.recall(y_true, y_pred)

        denominator = precision + recall
        return float(2 * precision * recall / denominator) if denominator > 0 else 0.0