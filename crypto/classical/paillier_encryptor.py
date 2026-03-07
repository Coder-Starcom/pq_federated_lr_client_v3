# crypto/classical/paillier_encryptor.py

from typing import Any
from phe import paillier

from crypto.interfaces.encryptor import Encryptor


class PaillierEncryptor(Encryptor):
    """
    Minimal Paillier encryption implementation
    using the 'phe' library.
    """

    def __init__(self, public_key=None, private_key=None):
        self.public_key = public_key
        self.private_key = private_key

    @staticmethod
    def generate_keypair():
        """
        Generate Paillier public/private keypair.
        """
        return paillier.generate_paillier_keypair()

    def encrypt(self, value: int) -> Any:
        """
        Encrypt integer value.
        """
        if self.public_key is None:
            raise ValueError("Public key not set.")
        return self.public_key.encrypt(value)

    def decrypt(self, ciphertext: Any) -> int:
        """
        Decrypt ciphertext.
        """
        if self.private_key is None:
            raise ValueError("Private key not set.")
        return self.private_key.decrypt(ciphertext)

    def add(self, c1: Any, c2: Any) -> Any:
        """
        Homomorphic addition.
        """
        return c1 + c2

    def multiply_by_constant(self, ciphertext: Any, scalar: int) -> Any:
        """
        Homomorphic multiplication by plaintext scalar.
        """
        return ciphertext * scalar