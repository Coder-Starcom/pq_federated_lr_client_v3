import requests
import base64
# Ensure you have installed: pip install kyber-py dilithium-py requests
from kyber_py.kyber import Kyber512
from dilithium_py.dilithium import Dilithium2

BASE_URL = "http://127.0.0.1:8000"
CLIENT_ID = "test_client_001"

def test_pqc_flow():
    print("--- Starting PQC Service Test ---")

    # 1. Test Kyber Key Fetch & Decapsulation
    print("[*] Fetching Server Kyber Public Key...")
    resp = requests.get(f"{BASE_URL}/keys/kyber")
    resp.raise_for_status()
    server_pk_kyber = base64.b64decode(resp.json()["public_key"])

    # Client-side: Encapsulate (using .encaps as required by kyber-py)
    print("[*] Encapsulating shared secret...")
    c_shared_secret, ciphertext = Kyber512.encaps(server_pk_kyber)
    
    # Server-side: Decapsulate
    print("[*] Sending ciphertext for decapsulation...")
    decaps_payload = {"ciphertext": base64.b64encode(ciphertext).decode()}
    decaps_resp = requests.post(f"{BASE_URL}/keys/decapsulate", json=decaps_payload)
    
    if decaps_resp.status_code == 200:
        s_shared_secret = base64.b64decode(decaps_resp.json()["shared_secret"])
        if c_shared_secret == s_shared_secret:
            print("[+] Kyber Success: Shared secrets match!")
        else:
            print("[-] Kyber Failure: Secret mismatch.")
    else:
        print(f"[-] Kyber Error: {decaps_resp.text}")

    # 2. Test Client Registration (Dilithium)
    print("\n[*] Generating Client Dilithium keys...")
    client_pk_dil, client_sk_dil = Dilithium2.keygen()
    
    print("[*] Registering client...")
    reg_payload = {
        "client_id": CLIENT_ID,
        "public_key": base64.b64encode(client_pk_dil).decode()
    }
    reg_resp = requests.post(f"{BASE_URL}/clients/register", json=reg_payload)
    print(f"[*] Registration response: {reg_resp.json()}")

    # 3. Test Signature Verification
    print("[*] Signing test message...")
    message = b"Quantum-resistant handshake test"
    # dilithium-py uses .sign(sk, message)
    signature = Dilithium2.sign(client_sk_dil, message)

    verify_payload = {
        "client_id": CLIENT_ID,
        "message": base64.b64encode(message).decode(),
        "signature": base64.b64encode(signature).decode()
    }
    verify_resp = requests.post(f"{BASE_URL}/verify", json=verify_payload)
    
    is_valid = verify_resp.json().get("valid")
    if is_valid:
        print("[+] Dilithium Success: Signature verified by server!")
    else:
        print("[-] Dilithium Failure: Signature rejected (valid=False).")

if __name__ == "__main__":
    try:
        test_pqc_flow()
    except Exception as e:
        print(f"[!] Test Error: {type(e).__name__} - {e}")
