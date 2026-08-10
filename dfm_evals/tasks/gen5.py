"""Gen5 missing tasks: CoQA, NQ-Open, TriviaQA.

These three generative QA benchmarks are part of the FlexOlmo Gen5 aggregate
but are not available in the installed inspect_evals version.  They follow the
same pattern as inspect_evals/squad (zero-shot, ``generate`` solver,
``f1`` / ``exact`` scorers).
"""

from __future__ import annotations

import random
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import exact, f1, mean, stderr
from inspect_ai.solver import generate, system_message

from inspect_evals.utils.huggingface import hf_dataset, load_dataset

from ._sharding import shard_sequence

# ---------------------------------------------------------------------------
# NQ-Open
# ---------------------------------------------------------------------------

NQ_OPEN_DATASET_PATH = "nq_open"
NQ_OPEN_REVISION = "5dd9790a83002ad084ddeb7c420dc716852c6f28"

NQ_OPEN_SYSTEM_MESSAGE = (
    "Answer the following question in as few words as possible."
)


@task(name="nq_open")
def nq_open(shuffle: bool = True, num_shards: int = 1, shard_index: int = 0) -> Task:
    """Natural Questions open-domain QA (3 610 validation samples)."""
    raw = load_dataset(
        NQ_OPEN_DATASET_PATH,
        split="validation",
        revision=NQ_OPEN_REVISION,
    )
    samples: list[Sample] = []
    for idx, record in enumerate(raw):
        samples.append(_nq_open_record_to_sample(record, idx))
    if shuffle:
        random.Random(4242).shuffle(samples)
    if num_shards > 1:
        samples = shard_sequence(samples, num_shards=num_shards, shard_index=shard_index)
    dataset = MemoryDataset(
        samples=samples,
        name=f"nq_open-shard-{shard_index}-of-{num_shards}",
        location=NQ_OPEN_DATASET_PATH,
        shuffled=shuffle,
    )
    return Task(
        dataset=dataset,
        solver=[system_message(NQ_OPEN_SYSTEM_MESSAGE), generate()],
        scorer=[f1(), exact()],
        metrics=[mean(), stderr()],
    )


def _nq_open_record_to_sample(record: dict[str, Any], idx: int = 0) -> Sample:
    return Sample(
        input=f"{record['question']}",
        target=record["answer"],
        id=f"nq_{idx}",
    )


# ---------------------------------------------------------------------------
# TriviaQA
# ---------------------------------------------------------------------------

TRIVIAQA_DATASET_PATH = "mandarjoshi/trivia_qa"
TRIVIAQA_REVISION = "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"

TRIVIAQA_SYSTEM_MESSAGE = (
    "Answer the following question in as few words as possible."
)


@task(name="triviaqa")
def triviaqa(shuffle: bool = True, num_shards: int = 1, shard_index: int = 0) -> Task:
    """TriviaQA open-domain QA (17 944 validation samples, no-context config)."""
    raw = load_dataset(
        TRIVIAQA_DATASET_PATH,
        name="rc.nocontext",
        split="validation",
        revision=TRIVIAQA_REVISION,
    )
    samples: list[Sample] = []
    for idx, record in enumerate(raw):
        samples.append(_triviaqa_record_to_sample(record, idx))
    if shuffle:
        random.Random(4242).shuffle(samples)
    if num_shards > 1:
        samples = shard_sequence(samples, num_shards=num_shards, shard_index=shard_index)
    dataset = MemoryDataset(
        samples=samples,
        name=f"triviaqa-shard-{shard_index}-of-{num_shards}",
        location=TRIVIAQA_DATASET_PATH,
        shuffled=shuffle,
    )
    return Task(
        dataset=dataset,
        solver=[system_message(TRIVIAQA_SYSTEM_MESSAGE), generate()],
        scorer=[f1(), exact()],
        metrics=[mean(), stderr()],
    )


def _triviaqa_record_to_sample(record: dict[str, Any], idx: int = 0) -> Sample:
    answer = record["answer"]
    targets = [answer["value"]]
    if "aliases" in answer:
        targets.extend(a for a in answer["aliases"] if a not in targets)
    return Sample(
        input=f"{record['question']}",
        target=targets,
        id=f"triviaqa_{idx}",
    )


# ---------------------------------------------------------------------------
# CoQA
# ---------------------------------------------------------------------------

COQA_DATASET_PATH = "stanfordnlp/coqa"
COQA_REVISION = "0d9e9952f1ef6e5415492d3d84b5873259137e3c"

COQA_SYSTEM_MESSAGE = (
    "Read the passage and answer the question in as few words as possible."
)


@task(name="coqa")
def coqa(shuffle: bool = True, seed: int = 4242, num_shards: int = 1, shard_index: int = 0) -> Task:
    """Conversational QA (500 validation documents, flattened to ~5 000 Q-A pairs).

    Each CoQA record contains a story with multiple follow-up questions.
    We flatten to one sample per question, using the story as shared context
    (conversational history is omitted for comparability with extractive QA).
    """
    raw = load_dataset(COQA_DATASET_PATH, split="validation", revision=COQA_REVISION)
    samples: list[Sample] = []
    for idx, record in enumerate(raw):
        story = record["story"]
        record_id = str(idx)
        questions = record["questions"]
        answers = record["answers"]
        answer_texts = answers["input_text"]
        answer_starts = answers.get("answer_start", [None] * len(questions))
        answer_ends = answers.get("answer_end", [None] * len(questions))
        for turn_idx, (q, ans_text, start, end) in enumerate(
            zip(questions, answer_texts, answer_starts, answer_ends, strict=True)
        ):
            span_text = story[start:end] if start is not None and end is not None else ""
            targets = list({ans_text, span_text} - {""}) or [ans_text]
            samples.append(
                Sample(
                    input=f"Passage: {story}\nQuestion: {q}",
                    target=targets,
                    id=f"coqa_{idx}_turn_{turn_idx}",
                )
            )
    if shuffle:
        random.Random(seed).shuffle(samples)
    if num_shards > 1:
        samples = shard_sequence(samples, num_shards=num_shards, shard_index=shard_index)
    dataset = MemoryDataset(
        samples=samples,
        name="coqa",
        location=COQA_DATASET_PATH,
        shuffled=shuffle,
    )
    return Task(
        dataset=dataset,
        solver=[system_message(COQA_SYSTEM_MESSAGE), generate()],
        scorer=[f1(), exact()],
        metrics=[mean(), stderr()],
        config=GenerateConfig(max_tokens=128),
    )
