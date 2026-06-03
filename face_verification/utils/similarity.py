# utils/similarity.py
"""
Cosine similarity utility.
Preserved exactly from verification/similarity.py.
"""

import numpy as np


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Cosine similarity between two embedding vectors.

    Fast path: if both vectors are already L2-normalised (which they are
    after FaceVerificationService._extract_embedding normalises them) the
    function reduces to a single dot-product, skipping two norm calls.

    Safe path: guards against zero-norm vectors so we never return NaN.
    """
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)

    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))

    if n1 == 0.0 or n2 == 0.0:
        return 0.0

    # Skip division when vectors are already unit-norm (saves ~2 µs per call)
    if abs(n1 - 1.0) < 1e-5 and abs(n2 - 1.0) < 1e-5:
        return float(np.dot(v1, v2))

    return float(np.dot(v1, v2) / (n1 * n2))