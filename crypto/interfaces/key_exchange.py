# crypto/interfaces/key_exchange.py

from abc import ABC, abstractmethod
from typing import Tuple


class KeyExchange(ABC):
    """
    Abstract interface for key exchange mechanisms.

    Implementations may include:
    - Post-Quantum schemes (Kyber)
    - Classical schemes (Diffie-Hellman)
    """

    @abstractmethod
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate a public/private key pair.

        Returns:
            Tuple[bytes, bytes]: (public_key, private_key)
        """
        pass

    @abstractmethod
    def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """
        Encapsulate a shared secret using the recipient's public key.

        Args:
            public_key (bytes): recipient public key

        Returns:
            Tuple[bytes, bytes]:
                ciphertext - sent to recipient
                shared_secret - derived symmetric key
        """
        pass

    @abstractmethod
    def decapsulate(self, ciphertext: bytes, private_key: bytes) -> bytes:
        """
        Recover the shared secret from the ciphertext.

        Args:
            ciphertext (bytes): received ciphertext
            private_key (bytes): recipient private key

        Returns:
            bytes: shared_secret
        """
        pass