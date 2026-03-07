"""
V3 Multi-Process Localhost Simulation

Spawns independent client processes that connect to:
- Coordinator Service (Port 8000)
- Key Manager Service (Port 8001)
"""

import numpy as np
import multiprocessing
import time
import os

# Ensure your V3 LocalClient is imported
from client.local_client import LocalClient

# V3 Service URLs
COORDINATOR_URL = "http://127.0.0.1:8000"
KEY_MANAGER_URL = "http://127.0.0.1:8001"

def run_client(client_id: str):
    # Stagger starts so they don't hit the registration endpoint at the exact same ms
    if client_id != "client_1":
        time.sleep(2)

    print(f"🚀 Starting {client_id}...")

    # 1. Generate local synthetic data (30 features for V3)
    np.random.seed()
    X = np.random.randn(200, 30).astype(np.float32)
    true_w = np.random.randn(30, 1)
    y = (X @ true_w > 0).astype(np.float32)

    # 2. Instantiate V3 Client pointing to Localhost URLs
    # Your LocalClient.__init__ should now take these URL strings
    client = LocalClient(
        client_id=client_id,
        coordinator_url=COORDINATOR_URL,
        key_manager_url=KEY_MANAGER_URL,
        X=X,
        y=y,
        learning_rate=0.01,
        local_epochs=3,
    )

    # 3. Start the Federated Learning Loop
    # Inside run_client...
    try:
        print(f"📡 {client_id} attempting to join federation...")
        client.run(rounds=5)
        print(f"🏁 {client_id} finished all rounds successfully.")
    except Exception as e:
        print(f"❌ {client_id} encountered an error: {type(e).__name__} -> {e}")


if __name__ == "__main__":
    print("🔥 Starting V3 Multi-Process Simulation...")
    print(f"[*] Coordinator: {COORDINATOR_URL}")
    print(f"[*] Key Manager: {KEY_MANAGER_URL}")

    processes = []
    for i in range(1, 4):  # Spawn 3 clients
        p = multiprocessing.Process(target=run_client, args=(f"client_{i}",))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n✅ All client processes finished.")
