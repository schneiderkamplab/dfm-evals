from __future__ import annotations

import pytest

from dfm_evals.tasks import long_context
from dfm_evals.tasks.long_context import _token_f1


def test_token_f1_counts_repeated_tokens() -> None:
    assert _token_f1("red red blue", "red blue blue") == pytest.approx(2 / 3)


def test_token_f1_is_zero_for_empty_input() -> None:
    assert _token_f1("", "answer") == 0.0
    assert _token_f1("answer", "") == 0.0


def test_prepared_cache_key_includes_requested_example_cap(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(long_context, "PREPARED_CACHE_ROOT", tmp_path)

    smoke = long_context._prepared_cache_path("longalign", language="en", max_examples=10)
    production = long_context._prepared_cache_path(
        "longalign", language="en", max_examples=5000
    )

    assert smoke != production
    assert "_max10_" in smoke.name
    assert "_max5000_" in production.name
