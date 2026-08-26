from __future__ import annotations

import pytest

from dfm_evals.tasks.long_context import _token_f1


def test_token_f1_counts_repeated_tokens() -> None:
    assert _token_f1("red red blue", "red blue blue") == pytest.approx(2 / 3)


def test_token_f1_is_zero_for_empty_input() -> None:
    assert _token_f1("", "answer") == 0.0
    assert _token_f1("answer", "") == 0.0
