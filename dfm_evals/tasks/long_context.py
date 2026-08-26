"""Long-context evaluation tasks used by the HRM scheduler.

The loaders cap each task at a deterministic maximum number of examples and
keep inputs below the 8K serving budget by applying a conservative character
cap. These are evaluation-only tasks; they are not added to training data.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate
from datasets import load_dataset
from rouge_score import rouge_scorer

from ._sharding import shard_sequence

MAX_EXAMPLES = 5000
MAX_CONTEXT_CHARS = 30000
MODEL_CONTEXT_TOKENS = 8192
# Reserve room for the Gemma chat template and tokenizer-specific framing.  The
# serving limit applies after the template is rendered, not only to Sample.input.
# The Gemma chat renderer can add 1,025 tokens on the largest prompts. Keep a
# larger margin so the request plus max_new_tokens never exceeds the 8K server.
# Runtime fallback only. Production LongAlign caches are fitted and verified
# against the exact HF tokenizer and deployed chat template by
# scripts/prepare_long_context_eval_cache.py.
CHAT_TEMPLATE_OVERHEAD_TOKENS = 2600
PREPARED_CACHE_VERSION = "v8"
PREPARED_CACHE_ROOT = Path(
    os.environ.get(
        "DFM_EVAL_CACHE_DIR",
        "/work/dfm/HRM-Text/data/eval_cache/long_context",
    )
)
LONGALIGN_DATASET = "zai-org/LongAlign-10k"
MARATHON_DATASET = "Hambaobao/Marathon"
QMSUM_DATASET = "pszemraj/qmsum-cleaned"
DANISH_SUMMARIZATION_DATASET = "oliverkinch/danish-summarization"


@lru_cache(maxsize=1)
def _tokenizer():
    from tokenizers import Tokenizer

    path = os.environ.get(
        "DFM_EVAL_TOKENIZER_PATH",
        "/work/dfm/HRM-Text/data_io/trained_tokenizers/bpe/tokenizer.json",
    )
    return Tokenizer.from_file(path)


def _token_count(text: str) -> int:
    return len(_tokenizer().encode(text).ids)


def _fit_prompt(
    context: str,
    prefix: str,
    suffix: str,
    *,
    max_gen_toks: int = 512,
) -> str:
    """Fit a rendered prompt below the serving limit, preserving both ends."""
    context = context[:MAX_CONTEXT_CHARS]
    budget = MODEL_CONTEXT_TOKENS - CHAT_TEMPLATE_OVERHEAD_TOKENS - max_gen_toks
    tokenizer = _tokenizer()
    prefix_ids = tokenizer.encode(prefix).ids
    suffix_ids = tokenizer.encode(suffix).ids
    context_ids = tokenizer.encode(context).ids
    if len(prefix_ids) + len(context_ids) + len(suffix_ids) <= budget:
        return f"{prefix}{context}{suffix}"

    marker_ids = tokenizer.encode("\n[... context truncated ...]\n").ids
    available = max(1, budget - len(prefix_ids) - len(suffix_ids) - len(marker_ids))
    head_len = max(1, int(available * 0.65))
    tail_len = max(0, available - head_len)
    shortened_ids = (
        prefix_ids
        + context_ids[:head_len]
        + marker_ids
        + (context_ids[-tail_len:] if tail_len else [])
        + suffix_ids
    )
    # Decoding can merge boundary tokens; trim until the serialized prompt is
    # within budget rather than relying on the approximate token arithmetic.
    text = tokenizer.decode(shortened_ids)
    while _token_count(text) > budget and tail_len > 0:
        tail_len -= 1
        shortened_ids = prefix_ids + context_ids[:head_len] + marker_ids + context_ids[-tail_len:] + suffix_ids
        text = tokenizer.decode(shortened_ids)
    return text


def _jsonl_rows(payload: bytes) -> list[dict[str, Any]]:
    """Read usable LongBench rows while tolerating its known broken lines."""
    rows: list[dict[str, Any]] = []
    for line in payload.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # LongBench's TriviaQA-E file contains a few literal newlines in
            # JSON strings. Those records cannot be recovered without changing
            # their text; skip them rather than failing the entire suite.
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _samples_task(
    samples: list[Sample],
    *,
    name: str,
    num_shards: int,
    shard_index: int,
) -> MemoryDataset:
    if len(samples) > MAX_EXAMPLES:
        samples = samples[:MAX_EXAMPLES]
    return MemoryDataset(
        samples=shard_sequence(samples, num_shards=num_shards, shard_index=shard_index),
        name=name,
        location=name,
    )


def _prepared_cache_path(name: str, *, language: str | None = None, max_gen_toks: int = 512) -> Path:
    suffix = f"_{language}" if language else ""
    return PREPARED_CACHE_ROOT / f"{PREPARED_CACHE_VERSION}_{name}{suffix}_gen{max_gen_toks}.jsonl"


def _read_prepared_cache(path: Path) -> list[Sample] | None:
    if not path.is_file():
        return None
    samples: list[Sample] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                samples.append(Sample(id=str(row["id"]), input=str(row["input"]), target=row["target"], metadata=row.get("metadata", {})))
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Invalid prepared long-context cache {path} at line "
            f"{locals().get('line_number', 'unknown')}; rebuild it explicitly"
        ) from exc
    return samples


def _write_prepared_cache(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temporary = Path(f.name)
            for sample in samples:
                f.write(json.dumps({"id": sample.id, "input": sample.input, "target": sample.target, "metadata": sample.metadata}, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def _longbench_zip() -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download("zai-org/LongBench", "data.zip", repo_type="dataset"))


@task(name="longbench_en")
def longbench_en(
    max_examples: int = MAX_EXAMPLES,
    num_shards: int = 1,
    shard_index: int = 0,
    max_gen_toks: int = 512,
) -> Task:
    """English LongBench/LongBench-E examples, capped at 5,000 rows."""
    cache_path = _prepared_cache_path("longbench_en", max_gen_toks=max_gen_toks)
    cached = _read_prepared_cache(cache_path)
    if cached is not None:
        return Task(
            dataset=_samples_task(cached, name="LongBench-English", num_shards=num_shards, shard_index=shard_index),
            solver=[generate(max_tokens=max_gen_toks, temperature=0.0)],
            scorer=long_context_scorer(),
        )
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(_longbench_zip()) as archive:
        for name in sorted(archive.namelist()):
            if not name.startswith("data/") or not name.endswith("_e.jsonl"):
                continue
            rows.extend(_jsonl_rows(archive.read(name)))
    rows = rows[: min(max_examples, MAX_EXAMPLES)]
    samples = []
    for row in rows:
        context = str(row.get("context", ""))
        question = str(row.get("input", "")).strip()
        prompt = _fit_prompt(
            context,
            "Context:\n",
            f"\n\nQuestion or task:\n{question}\n\nAnswer:",
            max_gen_toks=max_gen_toks,
        )
        answers = row.get("answers") or []
        if not isinstance(answers, list):
            answers = [str(answers)]
        samples.append(
            Sample(
                id=str(row.get("_id", len(samples))),
                input=prompt,
                target=[str(answer) for answer in answers],
                metadata={"dataset": row.get("dataset", "longbench"), "kind": "longbench"},
            )
        )
    _write_prepared_cache(cache_path, samples)
    return Task(
        dataset=_samples_task(samples, name="LongBench-English", num_shards=num_shards, shard_index=shard_index),
        solver=[generate(max_tokens=max_gen_toks, temperature=0.0)],
        scorer=long_context_scorer(),
    )


@task(name="longalign")
def longalign(
    language: str = "en",
    max_examples: int = MAX_EXAMPLES,
    num_shards: int = 1,
    shard_index: int = 0,
    max_gen_toks: int = 512,
) -> Task:
    """LongAlign instruction examples filtered by detected user language."""
    cache_path = _prepared_cache_path("longalign", language=language, max_gen_toks=max_gen_toks)
    cached = _read_prepared_cache(cache_path)
    if cached is not None:
        return Task(
            dataset=_samples_task(cached, name=f"LongAlign-{language}", num_shards=num_shards, shard_index=shard_index),
            solver=[generate(max_tokens=max_gen_toks, temperature=0.0)],
            scorer=long_context_scorer(),
        )
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
    raw = load_dataset(LONGALIGN_DATASET, split="train")
    samples = []
    for row in raw:
        messages = row.get("messages") or []
        if len(messages) < 2:
            continue
        user = str(messages[0].get("content", ""))
        assistant = str(messages[1].get("content", ""))
        try:
            detected = detect(user[:4000])
        except Exception:
            continue
        if detected != language:
            continue
        samples.append(
            Sample(
                id=f"{language}-{len(samples)}-{row.get('id', len(samples))}",
                input=_fit_prompt(user, "", "", max_gen_toks=max_gen_toks),
                target=[assistant],
                metadata={"dataset": f"longalign_{language}", "kind": "longalign"},
            )
        )
        if len(samples) >= min(max_examples, MAX_EXAMPLES):
            break
    _write_prepared_cache(cache_path, samples)
    return Task(
        dataset=_samples_task(samples, name=f"LongAlign-{language}", num_shards=num_shards, shard_index=shard_index),
        solver=[generate(max_tokens=max_gen_toks, temperature=0.0)],
        scorer=long_context_scorer(),
    )


@task(name="marathon")
def marathon(
    max_examples: int = MAX_EXAMPLES,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Task:
    raw = load_dataset(MARATHON_DATASET, split="test")
    samples = []
    for row in raw.select(range(min(len(raw), max_examples, MAX_EXAMPLES))):
        options = row["options"]
        labels = list(options.keys())
        option_text = "\n".join(f"{label}: {options[label]}" for label in labels)
        samples.append(
            Sample(
                id=str(row["id"]),
                input=_fit_prompt(
                    str(row["context"]),
                    "Context:\n",
                    f"\n\nQuestion:\n{row['question']}\n\nOptions:\n{option_text}\n\n"
                    "Answer with one option letter.",
                    max_gen_toks=16,
                ),
                target="",
                metadata={"dataset": "marathon", "kind": "marathon", "labels": labels},
            )
        )
    # Marathon's public test conversion does not expose an answer column in
    # the HF schema; its score is format-only rather than claimed accuracy.
    return Task(
        dataset=_samples_task(samples, name="Marathon", num_shards=num_shards, shard_index=shard_index),
        solver=[generate(max_tokens=16, temperature=0.0)],
        scorer=long_context_scorer(),
    )


@task(name="qmsum_cleaned")
def qmsum_cleaned(
    max_examples: int = MAX_EXAMPLES,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Task:
    # The cleaned test split currently has empty output fields. Validation has
    # usable reference summaries and is the reproducible evaluation split here.
    raw = load_dataset(QMSUM_DATASET, split="validation")
    samples = [
        Sample(
            id=str(row["id"]),
            input=_fit_prompt(str(row["input"]), "", ""),
            target=[str(row["output"])],
            metadata={"dataset": "qmsum_cleaned", "kind": "summarization"},
        )
        for row in raw
        if str(row.get("output", "")).strip()
    ][: min(max_examples, MAX_EXAMPLES)]
    return Task(
        dataset=_samples_task(samples, name="QMSum", num_shards=num_shards, shard_index=shard_index),
        solver=[generate(max_tokens=512, temperature=0.0)],
        scorer=long_context_scorer(),
    )


@task(name="danish_summarization_eur_lex")
def danish_summarization_eur_lex(
    max_examples: int = MAX_EXAMPLES,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Task:
    raw = load_dataset(DANISH_SUMMARIZATION_DATASET, "eur_lex", split="train")
    samples = [
        Sample(
            id=str(index),
            input=_fit_prompt(
                str(row["document"]),
                "Lav et kort, præcist referat af dokumentet.\n\nDokument:\n",
                "\n\nReferat:",
            ),
            target=[str(row["summary"])],
            metadata={"dataset": "danish_summarization_eur_lex", "kind": "summarization"},
        )
        for index, row in enumerate(raw.select(range(min(len(raw), max_examples, MAX_EXAMPLES))))
    ]
    return Task(
        dataset=_samples_task(samples, name="DanishSummarization-EUR-Lex", num_shards=num_shards, shard_index=shard_index),
        solver=[generate(max_tokens=512, temperature=0.0)],
        scorer=long_context_scorer(),
    )


@task(name="danish_summarization")
def danish_summarization(
    dataset_config: str = "nordjylland",
    max_examples: int = MAX_EXAMPLES,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Task:
    """Danish summarization, with a selectable HF config and a 5K cap."""
    raw = load_dataset(
        DANISH_SUMMARIZATION_DATASET,
        dataset_config,
        split="train",
    )
    rows = raw.select(range(min(len(raw), max_examples, MAX_EXAMPLES)))
    samples = [
        Sample(
            id=str(index),
            input=_fit_prompt(
                str(row["document"]),
                "Lav et kort, præcist referat af dokumentet.\n\nDokument:\n",
                "\n\nReferat:",
            ),
            target=[str(row["summary"])],
            metadata={
                "dataset": f"danish_summarization_{dataset_config}",
                "kind": "summarization",
            },
        )
        for index, row in enumerate(rows)
    ]
    return Task(
        dataset=_samples_task(
            samples,
            name=f"DanishSummarization-{dataset_config}",
            num_shards=num_shards,
            shard_index=shard_index,
        ),
        solver=[generate(max_tokens=512, temperature=0.0)],
        scorer=long_context_scorer(),
    )


@scorer(metrics=[mean(), stderr()])
def long_context_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion or ""
        references = target.target if isinstance(target.target, list) else [target.target]
        references = [str(value) for value in references if str(value).strip()]
        kind = str(state.metadata.get("kind", ""))
        if kind == "marathon":
            value = float(bool(re.fullmatch(r"\s*[A-D](?:\s|[.):]|$).*", prediction, re.IGNORECASE)))
            return Score(value=value, answer=prediction, explanation="format_only_no_public_gold")
        if not references:
            return Score(value=0.0, answer=prediction, explanation="no reference")
        reference = references[0]
        if kind == "summarization":
            rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            value = float(rouge.score(reference, prediction)["rougeL"].fmeasure)
        else:
            value = _token_f1(reference, prediction)
        return Score(value=value, answer=prediction, explanation=f"kind={kind}")

    return score


def _token_f1(reference: str, prediction: str) -> float:
    ref = re.findall(r"\w+", reference.lower())
    pred = re.findall(r"\w+", prediction.lower())
    if not ref or not pred:
        return 0.0
    ref_counts: dict[str, int] = {}
    pred_counts: dict[str, int] = {}
    for token in ref:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in pred:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    overlap = sum(min(count, pred_counts.get(token, 0)) for token, count in ref_counts.items())
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
