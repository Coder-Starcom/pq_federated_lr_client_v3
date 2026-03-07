"""
Federated Training Round

Encapsulates a single federated learning round.

Author: Saad
Project: PQ-Federated Logistic Regression
"""

from typing import Optional, Dict
import numpy as np

from core.federated.server import FederatedServer


class FederatedRound:
    """
    Represents one complete federated training round.

    A round consists of:
    - Server coordinating client updates
    - Aggregation of client parameters
    - Optional evaluation of the global model
    """

    def __init__(
        self,
        server: FederatedServer,
        round_id: int,
    ) -> None:
        """
        Initialize a federated round.

        Parameters
        ----------
        server : FederatedServer
            Central coordinating server.
        round_id : int
            Identifier for this training round.
        """
        self._server = server
        self._round_id = round_id

    def execute(
        self,
        evaluation_data: Optional[np.ndarray] = None,
        evaluation_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Execute one federated training round.

        Parameters
        ----------
        evaluation_data : Optional[np.ndarray]
        evaluation_labels : Optional[np.ndarray]

        Returns
        -------
        Dict[str, float]
            Dictionary containing round metrics.
        """
        # Perform one server-coordinated training round
        self._server.train_one_round()

        metrics: Dict[str, float] = {}

        # Optional evaluation (only if both provided)
        if evaluation_data is not None and evaluation_labels is not None:
            accuracy = self._server.evaluate(
                evaluation_data,
                evaluation_labels,
            )
            metrics["accuracy"] = accuracy

        metrics["round_id"] = self._round_id

        return metrics

    @property
    def round_id(self) -> int:
        """
        Return round identifier.
        """
        return self._round_id