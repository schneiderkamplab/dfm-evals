"""Code4 missing tasks: MBPP (configurable sandbox), HumanEvalPlus, MBPPPlus.

The upstream ``inspect_evals/mbpp`` hardcodes a Docker sandbox.  This wrapper
exposes a ``sandbox`` parameter (default ``"local"``) so scoring can run
without Docker.  HumanEvalPlus and MBPPPlus are EvalPlus variants with more
comprehensive test suites, loaded from the ``evalplus`` HuggingFace datasets.
"""

from __future__ import annotations

import textwrap
from typing import Any

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate, prompt_template
from inspect_ai.util import ExecResult, sandbox

from inspect_evals.humaneval.humaneval import VERIFY_TIMEOUT, find_code
from inspect_evals.mbpp.mbpp import (
    DATASET_PATH as MBPP_DATASET_PATH,
    EVAL_VERSION as MBPP_EVAL_VERSION,
    MBPP_DATASET_REVISION,
    NUM_EPOCHS as MBPP_NUM_EPOCHS,
    PROMPT_TEMPLATE as MBPP_PROMPT_TEMPLATE,
    record_to_sample as mbpp_record_to_sample,
    verify as mbpp_verify,
)
from inspect_evals.utils.huggingface import hf_dataset, load_dataset

from ._sharding import shard_sequence

# ---------------------------------------------------------------------------
# MBPP (configurable-sandbox wrapper around upstream inspect_evals task)
# ---------------------------------------------------------------------------


@task(name="mbpp")
def mbpp(
    temperature: float = 0.5,
    sandbox: str = "local",
    num_shards: int = 1,
    shard_index: int = 0,
) -> Task:
    """MBPP via upstream inspect_evals logic but with configurable sandbox.

    The upstream task hardcodes ``sandbox="docker"``.  This wrapper preserves
    the 3-shot prompt, 5-epoch pass@k evaluation, and verify scorer while
    allowing a non-Docker sandbox.
    """
    template = MBPP_PROMPT_TEMPLATE
    template += "\n\nFor example:\n\n"

    few_shot_dataset = load_dataset(
        MBPP_DATASET_PATH,
        "full",
        split="prompt",
        revision=MBPP_DATASET_REVISION,
    )
    few_shot_ids = [2, 3, 4]
    few_shot_dataset = few_shot_dataset.filter(
        lambda row: row["task_id"] in few_shot_ids
    )

    for i, sample in enumerate(few_shot_dataset):
        test_cases = "\n".join(sample["test_list"])
        template += "".join(
            [
                f"## Prompt {i + 1}\n",
                "```python\n",
                f"{sample['text']}\n",
                "```\n\n",
                f"## Test Case {i + 1}\n",
                "```python\n",
                f"{test_cases}\n```\n\n",
                f"## Completion {i + 1}\n",
                "```python\n",
                f"{sample['code']}\n```\n\n",
            ]
        )

    template += textwrap.dedent(
        """
        # Now, do it for the following task.

        ## Prompt:
        ```python
        {prompt}
        ```

        ## Test Case:
        ```python
        {test_list_str}
        ```

        ## Completion:
        """
    )

    dataset = hf_dataset(
        path=MBPP_DATASET_PATH,
        name="sanitized",
        sample_fields=mbpp_record_to_sample,
        split="test",
        revision=MBPP_DATASET_REVISION,
    )
    if num_shards > 1:
        dataset = MemoryDataset(
            samples=shard_sequence(
                dataset,
                num_shards=num_shards,
                shard_index=shard_index,
            ),
            name=f"mbpp-shard-{shard_index}-of-{num_shards}",
            location=MBPP_DATASET_PATH,
        )

    return Task(
        dataset=dataset,
        epochs=Epochs(MBPP_NUM_EPOCHS, ["mean", "pass_at_1", "pass_at_2", "pass_at_5"]),
        solver=[
            prompt_template(template),
            generate(),
        ],
        scorer=mbpp_verify(),
        config=GenerateConfig(temperature=temperature),
        sandbox=sandbox,
        version=MBPP_EVAL_VERSION.comparability_version,
        metadata=MBPP_EVAL_VERSION.to_metadata(),
    )


# ---------------------------------------------------------------------------
# HumanEvalPlus
# ---------------------------------------------------------------------------

HUMANEVALPLUS_DATASET_PATH = "evalplus/humanevalplus"
HUMANEVALPLUS_REVISION = "d32357cf319e50e9c8d8dab5ea876c72b0fd321b"


@task(name="humaneval_plus")
def humaneval_plus(
    sandbox: str = "local",
    max_gen_toks: int = 512,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Task:
    """HumanEval+ code-generation benchmark (164 test samples).

    Uses the EvalPlus test suite which augments HumanEval with more edge cases.
    Verification executes the generated code against the EvalPlus test string,
    which defines a ``check(candidate)`` function like the original HumanEval.
    """
    dataset = hf_dataset(
        path=HUMANEVALPLUS_DATASET_PATH,
        split="test",
        sample_fields=_humaneval_plus_record_to_sample,
        revision=HUMANEVALPLUS_REVISION,
    )
    if num_shards > 1:
        dataset = MemoryDataset(
            samples=shard_sequence(
                dataset,
                num_shards=num_shards,
                shard_index=shard_index,
            ),
            name=f"humaneval_plus-shard-{shard_index}-of-{num_shards}",
            location=HUMANEVALPLUS_DATASET_PATH,
        )
    return Task(
        dataset=dataset,
        solver=generate(max_tokens=max_gen_toks),
        scorer=verify_humaneval_plus(),
        sandbox=sandbox,
    )


def _humaneval_plus_record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        id=record["task_id"],
        input=record["prompt"],
        target=record["canonical_solution"],
        metadata={
            "prompt": record["prompt"],
            "test": record["test"],
            "entry_point": record["entry_point"],
        },
    )


@scorer(metrics=[accuracy(), stderr()])
def verify_humaneval_plus() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        answer = find_code(state.output.completion)
        if "\x00" in answer:
            return Score(
                value=INCORRECT,
                answer=answer.replace("\x00", ""),
                explanation="Generated code contained NUL bytes.",
            )
        code = [
            state.metadata["prompt"],
            answer,
            "\n",
            state.metadata["test"],
            "\n",
            "".join(["check(", state.metadata["entry_point"], ")"]),
        ]
        full_code = "".join(code)
        try:
            await sandbox().write_file("solution.py", full_code)
            result = await sandbox().exec(
                cmd=["python", "solution.py"],
                timeout=VERIFY_TIMEOUT,
            )
        except TimeoutError:
            result = ExecResult(False, 1, "", "Verification timed out.")
        except ValueError as exc:
            result = ExecResult(False, 1, "", f"Verification failed: {exc}")
        return Score(
            value=CORRECT if result.success else INCORRECT,
            answer=answer,
            explanation="".join(
                ["```python\n\n", *code, "\n```\n", f"\n{result.stderr}"]
                if not result.success
                else [""]
            ),
        )

    return score


# ---------------------------------------------------------------------------
# MBPPPlus
# ---------------------------------------------------------------------------

MBPPPLUS_DATASET_PATH = "evalplus/mbppplus"
MBPPPLUS_REVISION = "b2d74c91837c3f2a20c1299ae98133cbe7cfa077"


@task(name="mbpp_plus")
def mbpp_plus(
    temperature: float = 0.5,
    sandbox: str = "local",
    num_shards: int = 1,
    shard_index: int = 0,
) -> Task:
    """MBPP+ code-generation benchmark (378 test samples).

    Uses the EvalPlus test suite which augments MBPP with more edge cases.
    Verification appends assert statements from ``test_list`` to the generated
    code and executes, matching the upstream MBPP verify scorer pattern.
    """
    dataset = hf_dataset(
        path=MBPPPLUS_DATASET_PATH,
        split="test",
        sample_fields=_mbpp_plus_record_to_sample,
        revision=MBPPPLUS_REVISION,
    )
    if num_shards > 1:
        dataset = MemoryDataset(
            samples=shard_sequence(
                dataset,
                num_shards=num_shards,
                shard_index=shard_index,
            ),
            name=f"mbpp_plus-shard-{shard_index}-of-{num_shards}",
            location=MBPPPLUS_DATASET_PATH,
        )
    template = MBPP_PROMPT_TEMPLATE + "\n\n"
    template += textwrap.dedent(
        """
        ## Prompt:
        ```python
        {prompt}
        ```

        ## Test Case:
        ```python
        {test_list_str}
        ```

        ## Completion:
        """
    )
    return Task(
        dataset=dataset,
        epochs=Epochs(MBPP_NUM_EPOCHS, ["mean", "pass_at_1", "pass_at_2", "pass_at_5"]),
        solver=[
            prompt_template(template),
            generate(),
        ],
        scorer=verify_mbpp_plus(),
        config=GenerateConfig(temperature=temperature),
        sandbox=sandbox,
    )


def _mbpp_plus_record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        id=record["task_id"],
        input=record["prompt"],
        target=record["test_list"],
        metadata={
            "prompt": record["prompt"],
            "test_list": record["test_list"],
            "test_list_str": "\n".join(record["test_list"]),
            "test_imports": record.get("test_imports", []),
        },
    )


@scorer(metrics=[accuracy(), stderr()])
def verify_mbpp_plus() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        from inspect_evals.mbpp.mbpp import find_code as mbpp_find_code

        raw = state.output.completion
        generated = mbpp_find_code(raw)
        code = generated
        for test_case in target.target:
            code += f"\n{test_case}, {repr(test_case[len('assert '):])}"
        try:
            await sandbox().write_file("solution.py", code)
            result = await sandbox().exec(
                cmd=["python", "solution.py"],
                timeout=VERIFY_TIMEOUT,
            )
        except TimeoutError:
            result = ExecResult(False, 1, "", "Verification timed out.")
        return Score(
            value=CORRECT if result.success else INCORRECT,
            answer=raw,
            explanation=code if not result.success else "",
        )

    return score
