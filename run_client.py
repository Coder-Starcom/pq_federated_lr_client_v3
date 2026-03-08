import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ==========================
# Path Setup
# ==========================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    from client.local_client import LocalClient
except ImportError:
    print("❌ Error: Project folders not found. Ensure you are in the project root.")
    sys.exit(1)

# ==========================
# CONFIGURATION (V3 PQC)
# ==========================

# Update these if you deploy to Render/Cloud
COORDINATOR_URL = "https://pq-federated-coordinator-v3.onrender.com"
KEY_MANAGER_URL = "https://pq-federated-key-manager-v3.onrender.com"

# 🚩 INDIVIDUAL CLIENT SETTINGS 🚩
CLIENT_ID = "pc_node_05"
MY_DATA_FILE = "client_data_10.csv"

ROUNDS = 5
LOCAL_EPOCHS = 3
LEARNING_RATE = 0.01

# ==========================
# Dataset Loader
# ==========================

def load_split_csv(file_path):
    """Loads a specific split CSV and prepares it for training."""
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found!")
        sys.exit(1)

    print(f"📥 Loading training data from {file_path}...")
    df = pd.read_csv(file_path)

    # Assumes 'Class' is the target (Fraud vs Non-Fraud)
    X = df.drop("Class", axis=1).values.astype(np.float32)
    y = df["Class"].values.astype(np.float32).reshape(-1, 1)

    # Standard Scaling for faster convergence in PQC-LR
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# ==========================
# Main Execution
# ==========================

if __name__ == "__main__":
    print(f"🚀 Starting PQC Federated Client: {CLIENT_ID}")

    # 1. Load Local Data
    X_local, y_local = load_split_csv(MY_DATA_FILE)
    
    # 2. Instantiate V3 PQC Client
    # Uses 'coordinator_url' and 'key_manager_url' as defined in your V3 LocalClient
    client = LocalClient(
        client_id=CLIENT_ID,
        coordinator_url=COORDINATOR_URL,
        key_manager_url=KEY_MANAGER_URL,
        X=X_local,
        y=y_local,
        learning_rate=LEARNING_RATE,
        local_epochs=LOCAL_EPOCHS,
    )

    print("\n🔐 PQC Handshake initiated (Dilithium + Kyber)...")
    print(f"📡 Connecting to Coordinator: {COORDINATOR_URL}")

    # 3. Begin Federated Training Loop
    try:
        client.run(rounds=ROUNDS, delay=10.0)
        print(f"\n✅ {CLIENT_ID}: Federated Training Complete.")
    except Exception as e:
        print(f"\n❌ {CLIENT_ID}: Training failed: {e}")
