"""
Federated Client

Simulates a local federated learning participant in a FedAvg setting.

Author: Saad
Project: PQ-Federated Logistic Regression
"""

from typing import Tuple
import numpy as np

from core.models.base_model import BaseModel
from core.optimization.optimizer import Optimizer


class FederatedClient:
    """
    Represents a federated learning client.

    Each client:
    - Holds its own local dataset
    - Maintains a local model instance
    - Performs local training
    - Sends updated parameters to the server
    """

    def __init__(
        self,
        client_id: int,
        model: BaseModel,
        optimizer: Optimizer,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Initialize client with local data and model.
        """
        self._client_id = client_id
        self._model = model
        self._optimizer = optimizer
        self._X = X
        self._y = y

    # ==========================
    # FedAvg Local Training
    # ==========================

    def train_local(self, local_epochs: int) -> Tuple[np.ndarray, float]:
        """
        Perform local training for a number of epochs.

        Parameters
        ----------
        local_epochs : int
            Number of local training epochs.

        Returns
        -------
        Tuple[np.ndarray, float]
            Updated (weights, bias) after local training.
        """
        if local_epochs <= 0:
            raise ValueError("local_epochs must be positive.")

        for _ in range(local_epochs):
            dw, db = self._model.compute_gradients(self._X, self._y)

            weights, bias = self._model.get_parameters()
            new_weights, new_bias = self._optimizer.step(
                weights,
                bias,
                dw,
                db,
            )

            self._model.set_parameters(new_weights, new_bias)

        return self._model.get_parameters()

    # ==========================
    # Parameter Synchronization
    # ==========================

    def set_global_parameters(
        self,
        weights: np.ndarray,
        bias: float,
    ) -> None:
        """
        Synchronize local model with global parameters.
        """
        self._model.set_parameters(weights, bias)

    def get_parameters(self) -> Tuple[np.ndarray, float]:
        """
        Retrieve current local model parameters.
        """
        return self._model.get_parameters()

    # ==========================
    # Properties
    # ==========================

    @property
    def client_id(self) -> int:
        """
        Unique identifier of the client.
        """
        return self._client_id