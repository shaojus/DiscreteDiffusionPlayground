VALID_ENCODINGS = frozenset({"binary", "gray"})


def normalize_encoding(encoding):
    normalized = str(encoding).lower()
    if normalized not in VALID_ENCODINGS:
        choices = ", ".join(sorted(VALID_ENCODINGS))
        raise ValueError(f"Unknown encoding {encoding!r}; expected one of: {choices}")
    return normalized


def binary_to_gray(values):
    return values ^ (values >> 1)


def gray_to_binary(values, n_bits):
    n_bits = int(n_bits)
    if n_bits <= 0:
        raise ValueError(f"n_bits must be positive, got {n_bits}")

    binary = values
    shift = 1
    while shift < n_bits:
        binary = binary ^ (binary >> shift)
        shift <<= 1
    return binary
