# crypto/post_quantum/dilithium_signature.py

from typing import Tuple
from dilithium_py.dilithium import Dilithium2  # Windows-friendly import
from crypto.interfaces.signature import SignatureScheme

class DilithiumSignature(SignatureScheme):
    """
    Post-Quantum Digital Signature implementation using Dilithium.
    Compatible with Windows Python 3.12.
    """

    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate Dilithium public/private key pair.
        Returns: (public_key, private_key)
        """
        # Returns (pk, sk)
        public_key, private_key = Dilithium2.keygen()
        return public_key, private_key

    def sign(self, message: bytes, private_key: bytes) -> bytes:
        """
        Sign a message.
        """
        # dilithium-py: Dilithium2.sign(sk, msg)
        signature = Dilithium2.sign(private_key, message)
        return signature

    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """
        Verify a Dilithium signature.
        """
        try:
            # dilithium-py: Dilithium2.verify(pk, msg, sig)
            return Dilithium2.verify(public_key, message, signature)
        except Exception:
            return False
