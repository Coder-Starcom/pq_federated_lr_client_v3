from communication.message import ModelUpdateRequest

req = ModelUpdateRequest(
    round=1,
    client_id="test_client",
    weights=[0.1, 0.2, 0.3],
    bias=0.1,
    signature="abc123",
    kyber_ciphertext="xyz456",
    client_pubkey_id="client_001"
)

print(req)
print("Schema loaded successfully")