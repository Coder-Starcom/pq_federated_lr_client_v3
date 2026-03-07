# crypto/encoding.py

from typing import List

# Number of decimal places to preserve
SCALE = 10**6


def encode_float(value: float) -> int:
    """
    Convert float to scaled integer.

    Example:
        0.123456 -> 123456
    """
    return int(round(value * SCALE))


def decode_float(value: int) -> float:
    """
    Convert scaled integer back to float.
    """
    return value / SCALE


def encode_vector(values: List[float]) -> List[int]:
    """
    Encode list of floats into scaled integers.
    """
    return [encode_float(v) for v in values]


def decode_vector(values: List[int]) -> List[float]:
    """
    Decode list of scaled integers into floats.
    """
    return [decode_float(v) for v in values]