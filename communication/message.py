"""
Message Schemas

Defines Pydantic request and response models for
coordinator-client communication.

Author: Saad
Project: PQ-Federated Logistic Regression (V3 - PQC Enabled)
"""

from typing import List, Optional, Union
from pydantic import BaseModel


# ==========================
# Global Model Response
# ==========================

class GlobalModelResponse(BaseModel):
    """
    Response model for fetching the current global model.
    """
    round: int
    weights: Optional[List[float]]
    bias: Optional[float]
    message: str


# ==========================
# Client Update Submission
# ==========================

class ModelUpdateRequest(BaseModel):
    round: int
    client_id: str
    weights: List[float]
    bias: float
    # V3 PQC Fields - Must match what the client sends
    signature: Optional[str] = None
    kyber_ciphertext: Optional[str] = None



class ModelUpdateResponse(BaseModel):
    """
    Confirmation sent to client after a successful update submission.
    """

    message: str
    updates_received: int
    updates_required: int
    aggregation_performed: bool


# ==========================
# Round Status
# ==========================

class RoundStatusResponse(BaseModel):
    """
    Summary of the current federated round status.
    """

    round: int
    updates_received: int
    updates_required: int
    ready_for_next_round: bool


# ==========================
# System Operations
# ==========================

class ResetResponse(BaseModel):
    """
    Confirmation sent after resetting the global training state.
    """

    message: str