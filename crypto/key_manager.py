# crypto/key_manager.py

from typing import Dict, Tuple, Optional

from crypto.post_quantum.kyber_kem import KyberKEM
from crypto.post_quantum.dilithium_signature import DilithiumSignature


class KeyManager:
    """
    Manages all PQC keys in the system.

    Responsibilities
    ----------------
    - Generate Kyber KEM server keys
    - Generate Dilithium server keys
    - Store registered client public keys
    """

    def __init__(self):

        self._kyber = KyberKEM()
        self._dilithium = DilithiumSignature()

        # Server keys
        self._kyber_public: Optional[bytes] = None
        self._kyber_private: Optional[bytes] = None

        self._dilithium_public: Optional[bytes] = None
        self._dilithium_private: Optional[bytes] = None

        # Client registry
        self._client_keys: Dict[str, bytes] = {}

    # -------------------------------------------------
    # Server Key Generation
    # -------------------------------------------------

    def generate_server_keys(self) -> None:
        """
        Generate server Kyber and Dilithium keys.
        """

        self._kyber_public, self._kyber_private = self._kyber.generate_keypair()

        self._dilithium_public, self._dilithium_private = (
            self._dilithium.generate_keypair()
        )

    # -------------------------------------------------
    # Kyber Access
    # -------------------------------------------------

    def get_kyber_public_key(self) -> bytes:
        if self._kyber_public is None:
            raise ValueError("Kyber keys not generated.")
        return self._kyber_public

    def decapsulate(self, ciphertext: bytes) -> bytes:
        if self._kyber_private is None:
            raise ValueError("Kyber private key missing.")

        return self._kyber.decapsulate(ciphertext, self._kyber_private)

    # -------------------------------------------------
    # Dilithium Access
    # -------------------------------------------------

    def get_dilithium_public_key(self) -> bytes:
        if self._dilithium_public is None:
            raise ValueError("Dilithium keys not generated.")
        return self._dilithium_public

    def sign(self, message: bytes) -> bytes:
        if self._dilithium_private is None:
            raise ValueError("Dilithium private key missing.")

        return self._dilithium.sign(message, self._dilithium_private)

    # -------------------------------------------------
    # Client Registry
    # -------------------------------------------------

    def register_client(self, client_id: str, public_key: bytes) -> None:
        """
        Store client Dilithium public key.
        """
        self._client_keys[client_id] = public_key

    def get_client_key(self, client_id: str) -> bytes:
        if client_id not in self._client_keys:
            raise ValueError("Client not registered.")
        return self._client_keys[client_id]

    def verify_client_signature(
        self, client_id: str, message: bytes, signature: bytes
    ) -> bool:

        public_key = self.get_client_key(client_id)

        return self._dilithium.verify(message, signature, public_key)

    # -------------------------------------------------
    # Status
    # -------------------------------------------------

    def server_keys_initialized(self) -> bool:
        return (
            self._kyber_public is not None
            and self._kyber_private is not None
            and self._dilithium_public is not None
            and self._dilithium_private is not None
        )