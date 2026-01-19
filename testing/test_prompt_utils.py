import unittest

from backend.prompt_utils import parse_prompt_attention_a1111


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


if __name__ == "__main__":
    unittest.main()
