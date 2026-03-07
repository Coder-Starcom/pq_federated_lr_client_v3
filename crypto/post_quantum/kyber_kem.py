# crypto/post_quantum/kyber_kem.py

from typing import Tuple
from kyber_py.kyber import Kyber512  # Pure Python implementation
from crypto.interfaces.key_exchange import KeyExchange

class KyberKEM(KeyExchange):
    """
    Kyber Post-Quantum Key Encapsulation Mechanism implementation (Windows-friendly).
    Uses the kyber-py library to avoid C-extension compilation issues.
    """

    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate Kyber public/private key pair.
        Returns: (public_key, private_key)
        """
        # Kyber512.keygen() returns (pk, sk) as bytes
        public_key, private_key = Kyber512.keygen()
        return public_key, private_key

    def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """
        Encapsulate a shared secret using the recipient's public key.
        Returns: (ciphertext, shared_secret)
        """
        # Kyber512.encaps() returns (shared_secret, ciphertext)
        # Note: kyber-py returns (key, c), we unpack to match your interface
        shared_secret, ciphertext = Kyber512.encaps(public_key)
        return ciphertext, shared_secret

    def decapsulate(self, ciphertext: bytes, private_key: bytes) -> bytes:
        """
        Recover the shared secret using the private key.
        Returns: shared_secret (bytes)
        """
        # Kyber512.decaps() returns the shared secret as bytes
        shared_secret = Kyber512.decaps(private_key, ciphertext)
        return shared_secret
