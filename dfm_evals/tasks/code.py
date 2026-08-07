from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset
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
from inspect_ai.solver import Solver, TaskState, generate
from inspect_ai.util import ExecResult, sandbox
from inspect_evals.humaneval.humaneval import (
    VERIFY_TIMEOUT,
    find_code,
    humaneval as inspect_humaneval,
)
from ._sharding import shard_sequence


@task(name="humaneval")
def humaneval(
    solver: Solver | None = None,
    instruction_prompt: str | None = None,
    scorer: Scorer | list[Scorer] | None = None,
    sandbox: str = "docker",
    max_gen_toks: int = 512,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Task:
    """HumanEval code-generation benchmark via inspect-evals.

    The scorer executes generated Python against unit tests, so the default
    sandbox is Docker. Keep that default for real runs unless the execution
    environment has an explicitly approved alternative sandbox.
    """

    kwargs = {
        "solver": solver or generate(max_tokens=max_gen_toks),
        "scorer": scorer or verify_sanitized(),
        "sandbox": sandbox,
    }
    if instruction_prompt is not None:
        kwargs["instruction_prompt"] = instruction_prompt
    task_obj = inspect_humaneval(**kwargs)
    if num_shards > 1:
        task_obj.dataset = MemoryDataset(
            samples=shard_sequence(
                task_obj.dataset,
                num_shards=num_shards,
                shard_index=shard_index,
            ),
            name=f"humaneval-shard-{shard_index}-of-{num_shards}",
            location="openai_humaneval",
        )
    return task_obj


@scorer(metrics=[accuracy(), stderr()])
def verify_sanitized() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        answer = find_code(state.output.completion)
        if "\x00" in answer:
            return Score(
                value=INCORRECT,
                answer=answer.replace("\x00", ""),
                explanation="The generated code contained embedded NUL bytes and was marked incorrect without execution.",
            )

        code = [
            state.metadata["prompt"],
            answer,
            "\n",
            state.metadata["test"],
            "\n",
            "".join(["check(", state.metadata["entry_point"], ")"]),
        ]

        try:
            result = await sandbox().exec(
                cmd=["python", "-c", "".join(code)],
                timeout=VERIFY_TIMEOUT,
            )
        except TimeoutError:
            result = ExecResult(False, 1, "", "Verification timed out.")
        except ValueError as exc:
            result = ExecResult(False, 1, "", f"Verification failed before execution: {exc}")

        return Score(
            value=CORRECT if result.success else INCORRECT,
            answer=answer,
            explanation="".join(
                ["The following verification code was executed:\n\n"]
                + ["```python\n\n"]
                + code
                + ["\n```\n"]
                + [f"\nThe submission was incorrect\n\n{result.stderr}"]
                if not result.success
                else [""]
            ),
        )

    return score
