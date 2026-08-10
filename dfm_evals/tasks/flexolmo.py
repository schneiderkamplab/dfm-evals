"""FlexOlmo comparison tasks: MMLU-Pro, AGIEval, BBH.

These wrap the upstream ``inspect_evals`` tasks (which don't support sharding)
so the eval scheduler can split them across GPUs.
"""

from __future__ import annotations

import re as _re

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.scorer import choice
from inspect_ai.solver import generate, multiple_choice

from ._sharding import shard_sequence

# ---------------------------------------------------------------------------
# Monkey-patch inspect_ai's parse_answers to also recognise \boxed{LETTER}
# and bare-letter formats that the model frequently produces but the upstream
# scorer silently ignores.
# ---------------------------------------------------------------------------
_original_parse_answers = None


def _patch_parse_answers() -> None:
    global _original_parse_answers
    if _original_parse_answers is not None:
        return
    import inspect_ai.solver._multiple_choice as _mc

    _original_parse_answers = _mc.parse_answers

    _boxed_letter_re = _re.compile(r"\\boxed\{\s*(?:\\text\{)?\s*([A-Za-z])\s*(?:\})?\s*\}")

    def _patched(state, multiple_correct: bool) -> set[str]:
        answers = _original_parse_answers(state, multiple_correct)
        if answers:
            return answers
        completion = state.output.completion or ""
        allowed = set(
            _mc.answer_character(i) for i in range(len(state.choices))
        )
        m = _boxed_letter_re.search(completion)
        if m:
            letter = m.group(1).upper()
            if letter in allowed:
                return {letter}
        lines = [ln.strip() for ln in completion.strip().splitlines() if ln.strip()]
        if lines and len(lines[-1]) == 1 and lines[-1].isalpha():
            letter = lines[-1].upper()
            if letter in allowed:
                return {letter}
        return set()

    _patched.__dfm_evals_patched__ = True
    _mc.parse_answers = _patched


_patch_parse_answers()


def _shard_samples(samples, *, num_shards: int, shard_index: int, name: str) -> MemoryDataset:
    if num_shards > 1:
        samples = shard_sequence(samples, num_shards=num_shards, shard_index=shard_index)
    return MemoryDataset(
        samples=list(samples),
        name=f"{name}-shard-{shard_index}-of-{num_shards}" if num_shards > 1 else name,
        location=name,
    )


@task(name="mmlu_pro")
def mmlu_pro(num_shards: int = 1, shard_index: int = 0) -> Task:
    from inspect_evals.mmlu_pro.mmlu_pro import mmlu_pro as _upstream

    upstream = _upstream()
    dataset = _shard_samples(
        list(upstream.dataset),
        num_shards=num_shards,
        shard_index=shard_index,
        name="mmlu_pro",
    )
    return Task(
        dataset=dataset,
        solver=upstream.solver,
        scorer=upstream.scorer,
        config=upstream.config,
    )


@task(name="agieval")
def agieval(num_shards: int = 1, shard_index: int = 0) -> Task:
    from inspect_evals.agieval.agieval import (
        agie_aqua_rat,
        agie_logiqa_en,
        agie_lsat_ar,
        agie_lsat_lr,
        agie_lsat_rc,
        agie_sat_en,
        agie_sat_en_without_passage,
        agie_sat_math,
    )

    all_samples = []
    for func in (
        agie_aqua_rat,
        agie_logiqa_en,
        agie_lsat_ar,
        agie_lsat_lr,
        agie_lsat_rc,
        agie_sat_en,
        agie_sat_en_without_passage,
        agie_sat_math,
    ):
        t = func()
        all_samples.extend(t.dataset)
    dataset = _shard_samples(
        all_samples,
        num_shards=num_shards,
        shard_index=shard_index,
        name="agieval",
    )
    return Task(
        dataset=dataset,
        solver=multiple_choice(),
        scorer=choice(),
    )


_BBH_SUBSETS = [
    "date_understanding",
    "disambiguation_qa",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
]


@task(name="bbh")
def bbh(num_shards: int = 1, shard_index: int = 0) -> Task:
    from inspect_evals.bbh.bbh import bbh as _bbh_single
    from inspect_evals.bbh.bbh import bbh_scorer, bbh_solver

    all_samples = []
    for subset in _BBH_SUBSETS:
        t = _bbh_single(subset_name=subset)
        all_samples.extend(t.dataset)
    dataset = _shard_samples(
        all_samples,
        num_shards=num_shards,
        shard_index=shard_index,
        name="bbh",
    )
    return Task(
        dataset=dataset,
        solver=bbh_solver(),
        scorer=bbh_scorer(),
    )


@task(name="squad")
def squad(num_shards: int = 1, shard_index: int = 0) -> Task:
    from inspect_evals.squad.squad import squad as _upstream

    upstream = _upstream()
    dataset = _shard_samples(
        list(upstream.dataset),
        num_shards=num_shards,
        shard_index=shard_index,
        name="squad",
    )
    return Task(
        dataset=dataset,
        solver=upstream.solver,
        scorer=upstream.scorer,
        config=upstream.config,
    )
