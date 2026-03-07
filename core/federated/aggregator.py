"""
Federated Aggregator

Implements Federated Averaging (FedAvg) for model parameter aggregation.

Author: Saad
Project: PQ-Federated Logistic Regression
"""

from typing import List, Tuple
import numpy as np


class Aggregator:
    """
    Stateless Federated Averaging (FedAvg) aggregator.

    Aggregates model parameters received from multiple clients
    by computing their arithmetic mean.
    """

    @staticmethod
    def average_parameters(
        client_params: List[Tuple[np.ndarray, float]]
    ) -> Tuple[np.ndarray, float]:
        """
        Compute the average of client model parameters.

        Parameters
        ----------
        client_params : List[Tuple[np.ndarray, float]]
            List of (weights, bias) tuples from clients.

        Returns
        -------
        Tuple[np.ndarray, float]
            Averaged (weights, bias).
        """
        if not client_params:
            raise ValueError("client_params must not be empty.")

        num_clients = len(client_params)

        weight_sum = np.zeros_like(client_params[0][0])
        bias_sum = 0.0

        for weights, bias in client_params:
            weight_sum += weights
            bias_sum += bias

        avg_weights = weight_sum / num_clients
        avg_bias = bias_sum / num_clients

        return avg_weights, avg_bias