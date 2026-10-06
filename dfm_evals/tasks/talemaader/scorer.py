"""Judgment-only Talemaader scorer; helpers are reusable for saved-answer grading."""

from inspect_ai.model import GenerateConfig, Model, get_model
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

from dfm_evals.tasks.talemaader.prompts import (
    JUDGE_INSTRUCTIONS_V2 as JUDGE_INSTRUCTIONS_V2,
)
from dfm_evals.tasks.talemaader.prompts import (
    InvalidJudgmentError,
    build_judge_prompt,
    parse_judgment,
)

MAX_JUDGE_ATTEMPTS = 2


@scorer(metrics=[accuracy(), stderr()], name="model_graded_fact_v2")
def model_graded_fact_v2(
    model: str | Model | None = None,
    model_role: str | None = "grader",
) -> Scorer:
    """Grade with at most two 64-token calls; invalid judgments fail the sample."""

    async def score(state: TaskState, target: Target) -> Score:
        judge = get_model(model) if model is not None else get_model(role=model_role)
        prompt = build_judge_prompt(
            talemaade_udtryk=state.metadata["talemaade_udtryk"],
            criterion=target.text,
            answer=state.output.completion,
        )
        for attempt in range(1, MAX_JUDGE_ATTEMPTS + 1):
            output = await judge.generate(
                prompt, config=GenerateConfig(max_tokens=64, temperature=0)
            )
            try:
                if (
                    output.error
                    or len(output.choices) != 1
                    or output.choices[0].stop_reason != "stop"
                ):
                    raise InvalidJudgmentError(
                        "Judge output was not a complete stopped answer"
                    )
                value = parse_judgment(output.completion)
            except InvalidJudgmentError as exc:
                if attempt == MAX_JUDGE_ATTEMPTS:
                    raise InvalidJudgmentError(
                        f"Invalid judgment after {MAX_JUDGE_ATTEMPTS} attempts"
                    ) from exc
                prompt += "\nSvar kun med én af: GRADE: C, GRADE: P, GRADE: I."
                continue
            return Score(
                value=value,
                answer=state.output.completion,
                metadata={
                    "judgment": output.completion.strip(),
                    "judge_attempts": attempt,
                },
            )
        raise AssertionError("Unreachable judge retry state")

    return score
