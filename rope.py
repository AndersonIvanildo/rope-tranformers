"""Uma cabeça de self-attention, sem treino, para comparar posição e conteúdo."""

import hashlib
import re
import unicodedata

import numpy as np

MAX_TOKENS = 48
MAX_CHARS = 2000
METHODS = ("Sem posição", "Fixa senoidal", "RoPE")


def tokenize(text: str) -> list[str]:
    if len(text) > MAX_CHARS:
        raise ValueError(f"Use até {MAX_CHARS} caracteres.")
    tokens = re.findall(r"\w+|[^\w\s]", unicodedata.normalize("NFC", text))
    if not tokens:
        raise ValueError("Digite uma frase para começar.")
    if len(tokens) > MAX_TOKENS:
        raise ValueError(f"Use até {MAX_TOKENS} tokens, incluindo a pontuação.")
    return tokens


def frequencies(dim: int) -> np.ndarray:
    if not isinstance(dim, int) or dim < 2 or dim > 64 or dim % 2:
        raise ValueError("A dimensão deve ser par, entre 2 e 64.")
    return 10000.0 ** (-np.arange(0, dim, 2) / dim)


def embeddings(tokens: list[str], dim: int) -> np.ndarray:
    frequencies(dim)
    vectors = []
    for token in tokens:
        digest = hashlib.sha256(token.casefold().encode("utf-8")).digest()
        vector = np.random.default_rng(int.from_bytes(digest[:8], "little")).normal(size=dim)
        vectors.append(vector * np.sqrt(dim) / np.linalg.norm(vector))
    return np.array(vectors)


def sinusoidal(positions: np.ndarray, dim: int) -> np.ndarray:
    angles = np.asarray(positions)[:, None] * frequencies(dim)
    result = np.empty((len(positions), dim))
    result[:, 0::2] = np.sin(angles)
    result[:, 1::2] = np.cos(angles)
    return result


def rotate(vectors: np.ndarray, positions: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if vectors.ndim != 2 or positions.shape != (len(vectors),):
        raise ValueError("Use um vetor e uma posição por token.")
    angles = positions[:, None] * frequencies(vectors.shape[1])
    even, odd = vectors[:, 0::2], vectors[:, 1::2]
    result = np.empty_like(vectors)
    result[:, 0::2] = even * np.cos(angles) - odd * np.sin(angles)
    result[:, 1::2] = even * np.sin(angles) + odd * np.cos(angles)
    return result


def projections(dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies(dim)
    rng = np.random.default_rng(42)
    return tuple(np.linalg.qr(rng.normal(size=(dim, dim)))[0] for _ in range(3))


def attention(
    tokens: list[str], dim: int = 16, offset: int = 0, causal: bool = False
) -> dict[str, dict[str, np.ndarray]]:
    if not 1 <= len(tokens) <= MAX_TOKENS:
        raise ValueError(f"Use entre 1 e {MAX_TOKENS} tokens.")
    if not isinstance(offset, int) or not 0 <= offset <= 1000:
        raise ValueError("A posição inicial deve estar entre 0 e 1000.")
    x = embeddings(tokens, dim)
    wq, wk, wv = projections(dim)
    positions = np.arange(len(tokens)) + offset
    results = {}
    for method in METHODS:
        inputs = x + sinusoidal(positions, dim) if method == "Fixa senoidal" else x
        q, k, v = inputs @ wq, inputs @ wk, inputs @ wv
        if method == "RoPE":
            q, k = rotate(q, positions), rotate(k, positions)
        scores = q @ k.T / np.sqrt(dim)
        masked = scores
        if causal:
            future = np.triu(np.ones_like(scores, dtype=bool), 1)
            masked = np.where(future, -np.inf, scores)
        weights = np.exp(masked - masked.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        results[method] = {
            "q": q, "k": k, "v": v, "scores": scores,
            "weights": weights, "output": weights @ v,
        }
    return results
