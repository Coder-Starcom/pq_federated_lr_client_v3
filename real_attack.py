import requests
import json
import numpy as np
import base64

# Configuration for separate services
COORDINATOR_URL = "https://pq-federated-coordinator-v3.onrender.com"
KEY_MANAGER_URL = "https://pq-federated-key-manager-v3.onrender.com"

def run_test_header(name):
    print(f"\n--- [TEST] {name} ---")

def test_key_manager_direct():
    run_test_header("Direct Key Manager Access")
    try:
        # Testing the Key Manager directly to bypass Coordinator proxy
        r = requests.get(f"{KEY_MANAGER_URL}/keys/kyber", timeout=10)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print("✅ SUCCESS: Key Manager is online and serving Kyber PK.")
            print(json.dumps(r.json(), indent=2))
        else:
            print(f"❌ FAILED: Key Manager returned {r.status_code}")
    except Exception as e:
        print(f"⚠️ CONNECTION ERROR: Could not reach Key Manager at {KEY_MANAGER_URL}\n{e}")

def fetch_global_model():
    run_test_header("Fetching Global Model from Coordinator")
    try:
        r = requests.get(f"{COORDINATOR_URL}/global_model")
        if r.status_code != 200:
            print(f"❌ Status {r.status_code}: Model access denied.")
            return None
        
        data = r.json()
        print(f"✅ SUCCESS: Current Round: {data['round']}")
        return data
    except Exception as e:
        print(f"⚠️ CONNECTION ERROR: Could not reach Coordinator.\n{e}")
        return None

def test_signature_enforcement(model):
    run_test_header("Attack: Unsigned Malicious Update")
    # This payload LACKS the "signature" key entirely
    payload = {
        "round": model["round"],
        "client_id": "unsigned_attacker",
        "weights": [0.1] * 30,
        "bias": 0.5
    }
    r = requests.post(f"{COORDINATOR_URL}/submit_update", json=payload)
    
    if r.status_code >= 400:
        print(f"✅ DEFENSE ACTIVE: Coordinator rejected unsigned payload (Status {r.status_code})")
    else:
        print(f"❌ VULNERABILITY: Coordinator accepted unsigned update! (Status {r.status_code})")

def test_fake_signature_integrity(model):
    run_test_header("Attack: Fake PQC Signature")
    # Providing a signature that is NOT a valid Dilithium signature
    payload = {
        "round": model["round"],
        "client_id": "fake_sig_client",
        "weights": model["weights"],
        "bias": model["bias"],
        "signature": base64.b64encode(b"invalid_pqc_signature_data").decode()
    }
    r = requests.post(f"{COORDINATOR_URL}/submit_update", json=payload)
    
    if r.status_code == 401:
        print("✅ DEFENSE ACTIVE: Coordinator/KeyManager rejected invalid PQC signature.")
    elif r.status_code == 500:
        print("⚠️ SERVER ERROR (500): Coordinator/KeyManager crashed on malformed PQC data.")
    else:
        print(f"❌ VULNERABILITY: Fake signature bypassed verification (Status {r.status_code})")

def test_replay_attack(model):
    run_test_header("Attack: Replay Valid Round 0 Update")
    # We simulate capturing a valid update from an earlier round (Round 0)
    # and trying to submit it now while the server is at Round 1+
    payload = {
        "round": 0, # OLD ROUND
        "client_id": "pc_node_05",
        "weights": model["weights"],
        "bias": model["bias"],
        "signature": "valid_but_old_signature_from_logs" # Capture from real client
    }
    r = requests.post(f"{COORDINATOR_URL}/submit_update", json=payload)
    
    if r.status_code == 400:
        print(f"✅ DEFENSE ACTIVE: Coordinator rejected stale round (Status 400)")
    else:
        print(f"❌ VULNERABILITY: Stale update accepted! (Status {r.status_code})")

def test_signed_poisoning(model):
    run_test_header("Attack: Signed Model Poisoning (NaN/Inf)")
    
    # Poisoned weights designed to break the average (Aggregation)
    poisoned_weights = [float('nan')] * 30 
    
    # We must sign this so the Coordinator doesn't block it at the gate
    # Note: In a real test, you'd use the Dilithium2.sign from your client logic here
    payload = {
        "round": model["round"],
        "client_id": "pc_node_05", # USES LEGIT ID
        "weights": poisoned_weights,
        "bias": 999999.9,
        "signature": "legit_signature_of_poison_data" 
    }
    r = requests.post(f"{COORDINATOR_URL}/submit_update", json=payload)
    
    if r.status_code == 400 or r.status_code == 422:
        print(f"✅ DEFENSE ACTIVE: Poisoned data caught by validator (Status {r.status_code})")
    else:
        print(f"❌ VULNERABILITY: Global model poisoned! Check server logs for crashes.")

def test_identity_spoofing(model):
    run_test_header("Attack: Identity Spoofing")
    payload = {
        "round": model["round"],
        "client_id": "victim_client_01", # SPOOFED ID
        "weights": model["weights"],
        "bias": model["bias"],
        "signature": "attacker_valid_signature" # Signed by attacker, not victim
    }
    r = requests.post(f"{COORDINATOR_URL}/submit_update", json=payload)
    
    if r.status_code == 401:
        print(f"✅ DEFENSE ACTIVE: Identity/Signature mismatch caught.")
    else:
        print(f"❌ VULNERABILITY: Identity spoofed successfully.")

if __name__ == "__main__":
    print("="*65)
    print("SFFD-DAA: PQC FEDERATED LEARNING REAL ATTACK SCRIPT (V3)")
    print("="*65)
    
    # 1. Test Key Manager directly
    test_key_manager_direct()
    
    # 2. Get state from Coordinator
    model_data = fetch_global_model()

    if model_data:
        # 3. Execute Attack simulations
        test_signature_enforcement(model_data)
        test_fake_signature_integrity(model_data)

        test_replay_attack(model_data)
        #test_signed_poisoning(model_data)
        test_identity_spoofing(model_data)
    
    print("\n" + "="*65)
    print("TEST SUITE COMPLETE")
    print("="*65)
