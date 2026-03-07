# crypto/interfaces/encryptor.py

from abc import ABC, abstractmethod
from typing import Any


class Encryptor(ABC):
    """
    Abstract base class for homomorphic encryption schemes.

    All encryption implementations (Paillier, PQ schemes, etc.)
    must follow this contract.
    """

    @abstractmethod
    def encrypt(self, value: int) -> Any:
        """
        Encrypt an integer value.

        Parameters:
            value (int): Encoded integer

        Returns:
            Ciphertext object
        """
        pass

    @abstractmethod
    def decrypt(self, ciphertext: Any) -> int:
        """
        Decrypt a ciphertext.

        Parameters:
            ciphertext: Encrypted object

        Returns:
            int: Decrypted integer
        """
        pass

    @abstractmethod
    def add(self, c1: Any, c2: Any) -> Any:
        """
        Homomorphic addition of two ciphertexts.

        Parameters:
            c1, c2: Ciphertext objects

        Returns:
            Ciphertext representing encrypted sum
        """
        pass

    @abstractmethod
    def multiply_by_constant(self, ciphertext: Any, scalar: int) -> Any:
        """
        Homomorphic multiplication by plaintext scalar.

        Required for weighted averaging.

        Parameters:
            ciphertext: Encrypted value
            scalar (int): Plain integer

        Returns:
            Ciphertext
        """
        pass