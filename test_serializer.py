from communication.serializer import Serializer

data = b"quantum_safe_test"

encoded = Serializer.bytes_to_base64(data)
decoded = Serializer.base64_to_bytes(encoded)

print("Encoded:", encoded)
print("Decoded:", decoded)

assert decoded == data

print("Serializer PQC test passed")