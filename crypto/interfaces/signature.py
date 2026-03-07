# crypto/interfaces/signature.py

from abc import ABC, abstractmethod
from typing import Tuple


class SignatureScheme(ABC):
    """
    Abstract interface for digital signature schemes.

    Implementations may include:
    - Post-Quantum signatures (Dilithium)
    - Classical signatures (RSA, ECDSA)
    """

    @abstractmethod
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate a public/private key pair for signing.

        Returns:
            Tuple[bytes, bytes]: (public_key, private_key)
        """
        pass

    @abstractmethod
    def sign(self, message: bytes, private_key: bytes) -> bytes:
        """
        Sign a message using the private key.

        Args:
            message (bytes): message to be signed
            private_key (bytes): signing private key

        Returns:
            bytes: digital signature
        """
        pass

    @abstractmethod
    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """
        Verify a digital signature.

        Args:
            message (bytes): original message
            signature (bytes): signature to verify
            public_key (bytes): signer's public key

        Returns:
            bool: True if signature is valid, False otherwise
        """
        pass