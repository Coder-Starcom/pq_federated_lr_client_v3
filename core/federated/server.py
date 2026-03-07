"""
Federated Server

Coordinates federated learning across clients using FedAvg.

Author: Saad
Project: PQ-Federated Logistic Regression
"""

from typing import List, Tuple
import numpy as np

from core.models.base_model import BaseModel
from core.federated.client import FederatedClient
from core.federated.aggregator import Aggregator


class FederatedServer:
    """
    Central coordinator for federated learning.

    Responsibilities:
    - Maintain global model
    - Broadcast global parameters
    - Collect client updates
    - Aggregate parameters
    """

    def __init__(
        self,
        global_model: BaseModel,
        clients: List[FederatedClient],
    ) -> None:
        """
        Initialize server.

        Parameters
        ----------
        global_model : BaseModel
            Global model instance.
        clients : List[FederatedClient]
            Participating clients.
        """
        if not clients:
            raise ValueError("FederatedServer requires at least one client.")

        self._global_model = global_model
        self._clients = clients

    # ==========================
    # Core Federated Logic
    # ==========================

    def train_one_round(self, local_epochs: int = 5) -> None:
        """
        Perform one Federated Averaging (FedAvg) round.
        """
        if local_epochs <= 0:
            raise ValueError("local_epochs must be positive.")

        # Step 1: Broadcast global parameters
        global_weights, global_bias = self._global_model.get_parameters()

        for client in self._clients:
            client.set_global_parameters(global_weights, global_bias)

        # Step 2: Clients train locally
        client_params = [
            client.train_local(local_epochs)
            for client in self._clients
        ]

        # Step 3: Aggregate client parameters
        avg_weights, avg_bias = Aggregator.average_parameters(client_params)

        # Step 4: Update global model
        self._global_model.set_parameters(avg_weights, avg_bias)

    # ==========================
    # Evaluation
    # ==========================

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Evaluate global model accuracy.
        """
        predictions = self._global_model.forward(X)
        predicted_labels = (predictions >= 0.5).astype(int)
        accuracy = float(np.mean(predicted_labels == y))
        return accuracy

    def get_global_parameters(self) -> Tuple[np.ndarray, float]:
        """
        Retrieve current global model parameters.
        """
        return self._global_model.get_parameters()