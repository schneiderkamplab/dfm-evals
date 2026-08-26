from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class LengthEstimator(Protocol):
    def count_tokens(self, text: str) -> int: ...


class SimpleLengthEstimator:
    def count_tokens(self, text: str) -> int:
        return len(_TOKEN_PATTERN.findall(text))


class TiktokenLengthEstimator:
    def __init__(self, encoding_name: str) -> None:
        import tiktoken

        self._encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))


class HFLengthEstimator:
    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            _normalize_model_name(model_name),
            trust_remote_code=True,
        )

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))


class TokenizersLengthEstimator:
    def __init__(self, model_name: str) -> None:
        import tokenizers
        from tokenizers import Tokenizer

        path = Path(_normalize_model_name(model_name))
        tokenizer_path = path / "tokenizer.json" if path.is_dir() else path
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Tokenizer JSON not found: {tokenizer_path}")
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        config_path = tokenizer_path.parent / "tokenizer_config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("fix_mistral_regex") is True:
                split = tokenizers.pre_tokenizers.Split(
                    pattern=tokenizers.Regex(
                        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
                    ),
                    behavior="isolated",
                )
                current = self._tokenizer.pre_tokenizer
                if isinstance(current, tokenizers.pre_tokenizers.Sequence):
                    current[0] = split
                else:
                    if isinstance(current, tokenizers.pre_tokenizers.Metaspace):
                        current = tokenizers.pre_tokenizers.ByteLevel(
                            add_prefix_space=False,
                            use_regex=False,
                        )
                    self._tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
                        [split, current]
                    )

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


@lru_cache(maxsize=16)
def build_length_estimator(
    backend: str = "auto", model_name: str | None = None
) -> LengthEstimator:
    normalized = backend.strip().lower()

    if normalized == "auto":
        if model_name:
            try:
                return HFLengthEstimator(model_name)
            except Exception:
                pass
        try:
            return TiktokenLengthEstimator("cl100k_base")
        except Exception:
            return SimpleLengthEstimator()

    if normalized == "hf":
        if not model_name:
            raise ValueError("`tokenizer_model` is required when tokenizer_backend=hf.")
        return HFLengthEstimator(model_name)

    if normalized == "tokenizers":
        if not model_name:
            raise ValueError("`tokenizer_model` is required when tokenizer_backend=tokenizers.")
        return TokenizersLengthEstimator(model_name)

    if normalized in {"tiktoken", "cl100k_base"}:
        return TiktokenLengthEstimator("cl100k_base")

    if normalized == "simple":
        return SimpleLengthEstimator()

    raise ValueError(
        f"Unsupported tokenizer_backend {backend!r}. "
        "Supported values: auto, hf, tokenizers, tiktoken, cl100k_base, simple."
    )


def _normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip()
    for prefix in ("vllm/", "openai/"):
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized
