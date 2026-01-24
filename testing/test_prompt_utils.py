import unittest

import torch

from backend.prompt_utils import parse_prompt_attention_a1111
from backend.prompt_utils import build_prompt_embeddings_a1111


class PromptParserA1111Tests(unittest.TestCase):
    def assert_segments_almost_equal(self, actual, expected, places=6):
        self.assertEqual(len(actual), len(expected))
        for (actual_text, actual_weight), (expected_text, expected_weight) in zip(actual, expected):
            self.assertEqual(actual_text, expected_text)
            self.assertAlmostEqual(actual_weight, expected_weight, places=places)

    def test_plain_text(self):
        self.assert_segments_almost_equal(
            parse_prompt_attention_a1111("normal text"),
            [("normal text", 1.0)],
        )

    def test_parentheses_emphasis(self):
        self.assert_segments_almost_equal(
            parse_prompt_attention_a1111("an (important) word"),
            [("an ", 1.0), ("important", 1.1), (" word", 1.0)],
        )

    def test_explicit_weight(self):
        self.assert_segments_almost_equal(
            parse_prompt_attention_a1111("a (red:1.5) cat"),
            [("a ", 1.0), ("red", 1.5), (" cat", 1.0)],
        )

    def test_square_brackets(self):
        self.assert_segments_almost_equal(
            parse_prompt_attention_a1111("a [red] cat"),
            [("a ", 1.0), ("red", 1 / 1.1), (" cat", 1.0)],
        )

    def test_unbalanced_paren(self):
        self.assert_segments_almost_equal(
            parse_prompt_attention_a1111("(unbalanced"),
            [("unbalanced", 1.1)],
        )

    def test_escapes(self):
        self.assert_segments_almost_equal(
            parse_prompt_attention_a1111(r"\(literal\]"),
            [("(literal]", 1.0)],
        )

    def test_merge_same_weight(self):
        self.assert_segments_almost_equal(
            parse_prompt_attention_a1111("(unnecessary)(parens)"),
            [("unnecessaryparens", 1.1)],
        )

    def test_complex_example(self):
        prompt = "a (((house:1.3)) [on] a (hill:0.5), sun, (((sky)))."
        expected = [
            ("a ", 1.0),
            ("house", 1.5730000000000004),
            (" ", 1.1),
            ("on", 1.0),
            (" a ", 1.1),
            ("hill", 0.55),
            (", sun, ", 1.1),
            ("sky", 1.4641000000000006),
            (".", 1.1),
        ]
        self.assert_segments_almost_equal(parse_prompt_attention_a1111(prompt), expected)

    def test_break_token(self):
        self.assert_segments_almost_equal(
            parse_prompt_attention_a1111("a BREAK b"),
            [("a", 1.0), ("BREAK", -1.0), ("b", 1.0)],
        )


class _FakeTokenizeResult:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class _FakeTokenizer:
    def __init__(self, *, model_max_length: int = 6):
        self.model_max_length = model_max_length
        self.bos_token_id = 101
        self.eos_token_id = 102

    def __call__(self, text, add_special_tokens=False):
        if text is None:
            text = ""
        ids = [ord(ch) for ch in str(text) if ch != " "]
        return _FakeTokenizeResult(input_ids=ids)


class _FakeFinalLayerNorm:
    def __call__(self, x):
        return x * 10


class _FakeTextModel:
    def __init__(self):
        self.final_layer_norm = _FakeFinalLayerNorm()


class _FakeTextEncoder:
    def __init__(self, *, device=None, hidden_dim: int = 4, num_hidden_states: int = 3):
        self.device = device or torch.device("cpu")
        self.hidden_dim = hidden_dim
        self.num_hidden_states = num_hidden_states
        self.text_model = _FakeTextModel()
        self.config = type("Cfg", (), {"use_attention_mask": False})()

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
        base = input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 1, self.hidden_dim)
        if not output_hidden_states:
            return (base,)

        hidden_states = tuple(base + float(i) for i in range(self.num_hidden_states))
        return (None, None, hidden_states)


class _FakePipe:
    def __init__(self, *, model_max_length: int = 6, hidden_dim: int = 4):
        self.tokenizer = _FakeTokenizer(model_max_length=model_max_length)
        self.text_encoder = _FakeTextEncoder(hidden_dim=hidden_dim)


class PromptEmbeddingBuilderTests(unittest.TestCase):
    def test_lora_scale_forces_embedding_path(self):
        pipe = _FakePipe(model_max_length=6, hidden_dim=2)

        pos, neg, used = build_prompt_embeddings_a1111(
            pipe,
            "ab",
            "",
            lora_scale=0.5,
        )
        self.assertTrue(used)
        self.assertEqual(pos.shape, (1, 6, 2))
        self.assertEqual(neg.shape, (1, 6, 2))

    def test_concat_chunking_pads_to_max_length(self):
        pipe = _FakePipe(model_max_length=6, hidden_dim=2)

        # 5 non-space chars => 2 chunks (4 + 1). Each chunk is padded to 6 tokens => 12 total.
        pos, neg, used = build_prompt_embeddings_a1111(
            pipe,
            "abcde",
            "",
        )
        self.assertTrue(used)
        self.assertEqual(pos.shape, (1, 12, 2))
        self.assertEqual(neg.shape, (1, 12, 2))

    def test_clip_skip_selects_hidden_state_and_applies_final_ln(self):
        pipe = _FakePipe(model_max_length=6, hidden_dim=1)

        pos, neg, used = build_prompt_embeddings_a1111(
            pipe,
            "aaaaaaaaa",  # long enough to force embeddings
            "",
            clip_skip=1,
        )
        self.assertTrue(used)

        # With clip_skip=1 and 3 hidden states, we pick hidden_states[-2] = base+1, then *10 via final_layer_norm.
        # First token is BOS (101) at position 0 in each chunk.
        expected_bos = (101.0 + 1.0) * 10.0
        self.assertAlmostEqual(float(pos[0, 0, 0]), expected_bos, places=5)

    def test_weighting_scales_token_vectors(self):
        pipe = _FakePipe(model_max_length=6, hidden_dim=1)

        # Force embeddings by using a weight.
        pos, neg, used = build_prompt_embeddings_a1111(
            pipe,
            "(ab:2.0)",
            "",
            normalize_weights=False,
        )
        self.assertTrue(used)

        # Layout for first chunk (max_length=6):
        # [BOS, 'a', 'b', EOS, EOS, EOS] where 'a'=97, 'b'=98, EOS=102
        # Weights apply to 'a' and 'b' only, so positions 1 and 2 are doubled.
        self.assertEqual(float(pos[0, 1, 0]), 97.0 * 2.0)
        self.assertEqual(float(pos[0, 2, 0]), 98.0 * 2.0)
        self.assertEqual(float(pos[0, 0, 0]), 101.0)  # BOS not scaled


if __name__ == "__main__":
    unittest.main()
