from crypto.post_quantum.dilithium_signature import DilithiumSignature

d = DilithiumSignature()
pub, priv = d.generate_keypair()

print(pub)