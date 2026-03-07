"""
Secure Federated Learning Integration Test

This script:
1. Initializes secure experiment
2. Spawns 2 clients
3. Runs one federated round
4. Fetches and prints global model

Coordinator must already be running.
"""

import time
import requests
import numpy as np
import multiprocessing

from client.local_client import LocalClient

SERVER_URL = "http://127.0.0.1:8000"


# ======================================================
# Step 1 — Initialize Secure Experiment
# ======================================================

def initialize_secure_experiment():
    print("\nInitializing Secure Experiment...\n")

    response = requests.post(
        f"{SERVER_URL}/initialize",
        json={
            "input_dim": 30,
            "expected_clients": 2,
            "secure_mode": True,
        },
    )

    response.raise_for_status()
    print("Server Response:", response.json())


# ======================================================
# Step 2 — Dummy Data
# ======================================================

def generate_data(seed):
    np.random.seed(seed)
    X = np.random.randn(100, 30)

    true_w = np.random.randn(30,1)
    y_prob = 1 / (1 + np.exp(-(X @ true_w)))
    y = (y_prob > 0.5).astype(float)

    return X, y


# ======================================================
# Step 3 — Client Process
# ======================================================

def run_client(client_id: str, seed: int):
    X, y = generate_data(seed)

    client = LocalClient(
        client_id=client_id,
        server_url=SERVER_URL,
        X=X,
        y=y,
        learning_rate=0.1,
        local_epochs=3,
    )

    client.train_one_round()


# ======================================================
# Step 4 — Fetch Global Model
# ======================================================

def fetch_global_model():
    response = requests.get(f"{SERVER_URL}/global_model")
    response.raise_for_status()
    return response.json()


# ======================================================
# Main Execution
# ======================================================

if __name__ == "__main__":
    NUM_ROUNDS = 5
    initialize_secure_experiment()

    print("\nStarting Secure Clients...\n")

    for r in range(NUM_ROUNDS):
        print(f"\n--- Starting Secure Round {r} ---")
        start_time = time.time()

        p1 = multiprocessing.Process(target=run_client, args=("client_1", 42))
        p2 = multiprocessing.Process(target=run_client, args=("client_2", 99))

        p1.start()
        p2.start()

        p1.join()
        p2.join()

        total_time = time.time() - start_time
        print(f"Round {r} completed in {round(total_time, 2)}s")

        print("\nFetching Final Global Model...\n")

        model = fetch_global_model()

        print("Round:", model["round"])
        print("Weights:", model["weights"])
        print("Bias:", model["bias"])
        print("Total Secure Round Time:", round(total_time, 4), "seconds")

    print("\nSecure FL test completed.\n")