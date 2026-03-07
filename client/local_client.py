"""
Local Federated Client (V3 - PQC Enabled)

Features
- Kyber KEM handshake
- Dilithium client identity
- Federated training client

Author: Saad
"""

import time
import base64
import requests
import numpy as np

from kyber_py.kyber import Kyber512
from dilithium_py.dilithium import Dilithium2

from core.models.logistic_regression import LogisticRegression
from core.optimization.gradient_descent import GradientDescent
from communication.serializer import Serializer


class LocalClient:
    def __init__(
        self,
        client_id: str,
        coordinator_url: str,
        key_manager_url: str,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.1,
        local_epochs: int = 5,
    ):
        self.client_id = client_id
        # Store both URLs correctly
        self.coordinator_url = coordinator_url.rstrip("/")
        self.key_manager_url = key_manager_url.rstrip("/")

        self.X = X
        self.y = y
        self.local_epochs = local_epochs

        self.model = LogisticRegression(input_dim=X.shape[1])
        self.optimizer = GradientDescent(learning_rate)

        self.dilithium_pk = None
        self.dilithium_sk = None
        self.shared_secret = None

        self._initialize_identity()
        self._kyber_handshake()

    # ==========================================================
    # Dilithium Identity -> Uses Key Manager (8001)
    # ==========================================================
    def _initialize_identity(self):
        print(f"[{self.client_id}] Generating Dilithium identity...")
        pk, sk = Dilithium2.keygen()
        self.dilithium_pk = pk
        self.dilithium_sk = sk

        payload = {
            "client_id": self.client_id,
            "public_key": base64.b64encode(pk).decode()
        }
        # FIX: Point to key_manager_url
        r = requests.post(f"{self.key_manager_url}/clients/register", json=payload)
        r.raise_for_status()
        print(f"[{self.client_id}] Registered with Key Manager.")

    # ==========================================================
    # Kyber Handshake -> Uses Key Manager (8001)
    # ==========================================================
    def _kyber_handshake(self):
        print(f"[{self.client_id}] Starting Kyber handshake...")
        # FIX: Point to key_manager_url
        r = requests.get(f"{self.key_manager_url}/keys/kyber")
        r.raise_for_status()

        server_pk = base64.b64decode(r.json()["public_key"])
        shared_secret, ciphertext = Kyber512.encaps(server_pk)

        payload = {"ciphertext": base64.b64encode(ciphertext).decode()}
        # FIX: Point to key_manager_url
        r = requests.post(f"{self.key_manager_url}/keys/decapsulate", json=payload)
        r.raise_for_status()

        server_secret = base64.b64decode(r.json()["shared_secret"])
        if shared_secret != server_secret:
            raise RuntimeError("Kyber handshake failed")

        self.shared_secret = shared_secret
        print(f"[{self.client_id}] PQC secure channel established.")

    # ==========================================================
    # Fetch Global Model -> Uses Coordinator (8000)
    # ==========================================================
    def _fetch_global_model(self):
        while True:
            try:
                # FIX: Point to coordinator_url
                r = requests.get(f"{self.coordinator_url}/global_model")
                r.raise_for_status()
                data = r.json()
                if data["weights"] is not None:
                    return data
                time.sleep(2)
            except Exception:
                print(f"[{self.client_id}] Waiting for Coordinator...")
                time.sleep(5)

    # ==========================================================
    # Training Round -> Uses Coordinator (8000)
    # ==========================================================
    def train_one_round(self):
        data = self._fetch_global_model()
        round_number = data["round"]

        weights = Serializer.list_to_weights(data["weights"]).flatten()
        bias = Serializer.float_to_bias(data["bias"])
        self.model.set_parameters(weights, bias)

        for _ in range(self.local_epochs):
            dW, db = self.model.compute_gradients(self.X, self.y)
            w, b = self.model.get_parameters()
            new_w, new_b = self.optimizer.step(w, b, dW, db)
            self.model.set_parameters(new_w, new_b)

        weights, bias = self.model.get_parameters()

        payload = {
            "round": round_number,
            "client_id": self.client_id,
            "weights": Serializer.weights_to_list(weights.flatten()),
            "bias": Serializer.bias_to_float(bias),
        }

        # FIX: Point to coordinator_url
        r = requests.post(f"{self.coordinator_url}/submit_update", json=payload)
        r.raise_for_status()
        print(f"[{self.client_id}] Round {round_number} | Success")

    def run(self, rounds=5, delay=10):
        for _ in range(rounds):
            self.train_one_round()
            time.sleep(delay)
