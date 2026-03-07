"""
Serializer Utilities

Handles conversion between numpy arrays 
and JSON-serializable Python structures.

Now also supports serialization of:
- PQC ciphertext (Kyber)
- PQC signatures (Dilithium)

Author: Saad
Project: PQ-Federated Logistic Regression (V3)
"""

from typing import List, Union
import numpy as np
import base64


class Serializer:
    """
    Utility class for converting model parameters and
    cryptographic objects between Python and JSON-safe formats.
    """

    # ======================================================
    # Numpy → JSON
    # ======================================================

    @staticmethod
    def weights_to_list(weights: np.ndarray) -> List[float]:
        """
        Convert numpy weights to a flat list of floats.
        """
        return weights.flatten().tolist()

    @staticmethod
    def bias_to_float(bias: Union[float, np.ndarray]) -> float:
        """
        Ensure bias is a standard Python float.
        """
        if isinstance(bias, np.ndarray):
            return float(bias.item())
        return float(bias)

    # ======================================================
    # JSON → Numpy
    # ======================================================

    @staticmethod
    def list_to_weights(weights_list: List[float]) -> np.ndarray:
        """
        Convert a flat list back to a numpy array.
        """
        return np.array(weights_list, dtype=np.float64)

    @staticmethod
    def float_to_bias(bias: float) -> float:
        """
        Convert JSON float back to a scalar.
        """
        return float(bias)

    # ======================================================
    # PQC Serialization
    # ======================================================

    @staticmethod
    def bytes_to_base64(data: bytes) -> str:
        """
        Convert raw bytes (ciphertext/signature) to base64 string.
        """
        return base64.b64encode(data).decode("utf-8")

    @staticmethod
    def base64_to_bytes(data: str) -> bytes:
        """
        Convert base64 string back to raw bytes.
        """
        return base64.b64decode(data.encode("utf-8"))

    # ======================================================
    # Signature Serialization
    # ======================================================

    @staticmethod
    def signature_to_string(signature: bytes) -> str:
        """
        Serialize Dilithium signature.
        """
        return Serializer.bytes_to_base64(signature)

    @staticmethod
    def string_to_signature(signature_str: str) -> bytes:
        """
        Deserialize Dilithium signature.
        """
        return Serializer.base64_to_bytes(signature_str)

    # ======================================================
    # Ciphertext Serialization
    # ======================================================

    @staticmethod
    def ciphertext_to_string(ciphertext: bytes) -> str:
        """
        Serialize Kyber ciphertext.
        """
        return Serializer.bytes_to_base64(ciphertext)

    @staticmethod
    def string_to_ciphertext(ciphertext_str: str) -> bytes:
        """
        Deserialize Kyber ciphertext.
        """
        return Serializer.base64_to_bytes(ciphertext_str)