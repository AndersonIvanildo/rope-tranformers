import unittest

import numpy as np

from rope import METHODS, attention, embeddings, rotate, sinusoidal, tokenize


class PositionTests(unittest.TestCase):
    def test_known_rotation_and_norm(self):
        q = np.array([[1.0, 0.0], [0.0, 2.0]])
        rotated = rotate(q, np.array([np.pi / 2, np.pi]))
        np.testing.assert_allclose(rotated, [[0, 1], [0, -2]], atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(rotated, axis=1), np.linalg.norm(q, axis=1))

    def test_sinusoidal_definition(self):
        np.testing.assert_allclose(sinusoidal(np.array([0]), 4), [[0, 1, 0, 1]])
        np.testing.assert_allclose(sinusoidal(np.array([1]), 4),
                                   [[np.sin(1), np.cos(1), np.sin(0.01), np.cos(0.01)]])

    def test_relative_position_identity(self):
        q = np.array([[1.0, 2.0, -3.0, 4.0]])
        k = np.array([[2.0, -1.0, 0.5, 3.0]])
        left = np.sum(rotate(q, np.array([3])) * rotate(k, np.array([8])))
        right = np.sum(q * rotate(k, np.array([5])))
        self.assertAlmostEqual(left, right)

    def test_shift_invariance_and_fixed_difference(self):
        tokens = tokenize("O gato viu o gato.")
        for dim in (8, 16, 32, 64):
            for causal in (False, True):
                original = attention(tokens, dim, causal=causal)
                shifted = attention(tokens, dim, offset=100, causal=causal)
                for method in ("RoPE", "Sem posição"):
                    for tensor in ("scores", "weights", "output"):
                        np.testing.assert_allclose(original[method][tensor], shifted[method][tensor], atol=1e-12)
                self.assertGreater(np.max(np.abs(original["Fixa senoidal"]["weights"] - shifted["Fixa senoidal"]["weights"])), 1e-4)

    def test_attention_normalization_mask_and_output(self):
        for n in (1, 48):
            for causal in (False, True):
                result = attention(["gato"] * n, causal=causal)
                for method in METHODS:
                    data = result[method]
                    np.testing.assert_allclose(data["weights"].sum(axis=1), 1)
                    np.testing.assert_allclose(data["output"], data["weights"] @ data["v"])
                    self.assertTrue(np.isfinite(data["output"]).all())
                    if causal:
                        np.testing.assert_array_equal(np.triu(data["weights"], 1), 0)
                np.testing.assert_array_equal(result["RoPE"]["v"], result["Sem posição"]["v"])

    def test_no_position_is_permutation_equivariant(self):
        tokens = ["o", "gato", "correu"]
        order = [2, 0, 1]
        original = attention(tokens)["Sem posição"]
        permuted = attention([tokens[i] for i in order])["Sem posição"]
        np.testing.assert_allclose(permuted["weights"], original["weights"][np.ix_(order, order)])
        np.testing.assert_allclose(permuted["output"], original["output"][order])

    def test_tokenization_and_determinism(self):
        self.assertEqual(tokenize("Olá, ação!"), ["Olá", ",", "ação", "!"])
        self.assertEqual(tokenize("ac\u0327a\u0303o"), ["ação"])
        repeated = embeddings(tokenize("Gato gato"), 16)
        np.testing.assert_array_equal(repeated[0], repeated[1])
        np.testing.assert_array_equal(repeated, embeddings(tokenize("Gato gato"), 16))
        np.testing.assert_array_equal(repeated[0], embeddings(["gato", "novo"], 16)[0])

    def test_invalid_inputs(self):
        for text in ("", "  \n", "a " * 49, "a" * 2001):
            with self.assertRaises(ValueError):
                tokenize(text)
        for dim in (0, 3, 66):
            with self.assertRaises(ValueError):
                attention(["gato"], dim=dim)
        for offset in (-1, 1001):
            with self.assertRaises(ValueError):
                attention(["gato"], offset=offset)
        with self.assertRaises(ValueError):
            rotate(np.ones((2, 4)), np.array([1]))


if __name__ == "__main__":
    unittest.main()
