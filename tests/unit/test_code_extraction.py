"""Tests for code extraction — parity-critical path.

The extract_python_block regex must exactly match arc_memo's
concept_mem.utils.common.extract_python_block behavior.
"""
from mem2.utils.code_execution import extract_python_block


class TestExtractPythonBlock:
    def test_standard_python_block(self):
        text = "Some text\n```python\ndef transform(x):\n    return x\n```\nMore text"
        code, err = extract_python_block(text)
        assert err is None
        assert code == "def transform(x):\n    return x"

    def test_generic_code_block_fallback(self):
        text = "Some text\n```\ndef transform(x):\n    return x\n```\nMore text"
        code, err = extract_python_block(text)
        assert err is None
        assert code == "def transform(x):\n    return x"

    def test_python_block_preferred_over_generic(self):
        text = (
            "```\ngeneric code\n```\n"
            "```python\ndef transform(x):\n    return x\n```"
        )
        code, err = extract_python_block(text)
        assert err is None
        assert code == "def transform(x):\n    return x"

    def test_no_code_block(self):
        text = "Just some text without any code block"
        code, err = extract_python_block(text)
        assert code is None
        assert err is not None
        assert "no python code block" in err.lower()

    def test_empty_completion(self):
        code, err = extract_python_block("")
        assert code is None
        assert err is not None

    def test_null_completion(self):
        code, err = extract_python_block(None)
        assert code is None
        assert err is not None

    def test_strips_whitespace(self):
        text = "```python\n  \ndef transform(x):\n    return x\n  \n```"
        code, err = extract_python_block(text)
        assert err is None
        assert code.startswith("def transform")

    def test_multiline_code(self):
        text = (
            "Here's a solution:\n"
            "```python\n"
            "import numpy as np\n"
            "\n"
            "def transform(input_grid: np.ndarray) -> np.ndarray:\n"
            "    output = np.copy(input_grid)\n"
            "    output[output == 1] = 2\n"
            "    return output\n"
            "```\n"
            "This should work."
        )
        code, err = extract_python_block(text)
        assert err is None
        assert "import numpy" in code
        assert "def transform" in code
        assert "return output" in code

    def test_no_newline_after_fence_fails(self):
        """Verify that ```python without a newline does NOT match (parity with arc_memo)."""
        text = "```python def transform(x): return x```"
        code, err = extract_python_block(text)
        assert code is None

    def test_capital_python_does_not_match(self):
        """Verify case sensitivity (parity with arc_memo regex)."""
        text = "```Python\ndef transform(x):\n    return x\n```"
        code, err = extract_python_block(text)
        # Should fall through to generic ``` block, which also won't match here
        # because the opening fence is ```Python\n not ```\n
        assert code is None
