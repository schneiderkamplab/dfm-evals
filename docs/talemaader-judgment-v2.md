# Talemaader Judgment V2

`generative-talemaader` now uses `model_graded_fact_v2`. The original Inspect
`model_graded_fact` scorer and historical metrics are unchanged; the new scorer
name distinguishes new results. Source records, splits, targets, evaluated-model
prompt and its generation limit are unchanged. The existing Danish semantic
judge template and correctness criteria are retained; only the judge output
instructions change to judgment-only.

Reusable CPU rejudge helpers in `dfm_evals/tasks/talemaader/prompts.py`
(also imported by the scorer):

- `build_judge_prompt(*, talemaade_udtryk: str, criterion: str, answer: str) -> str`
- `parse_judgment(text: str) -> float`

This file uses only the standard library. Environments without Inspect can load
it without importing the package's task initializer:

```python
import importlib.util

spec = importlib.util.spec_from_file_location("talemaader_judgment", prompts_path)
judgment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judgment)
value = judgment.parse_judgment("GRADE: P")
```

The parser accepts exactly `GRADE: C`, `GRADE: P` or `GRADE: I`, with optional
surrounding whitespace. Scores are1.0,0.5,0.0 respectively. Explanations, multiple
grades and other text raise `InvalidJudgmentError` rather than returning zero.

The scorer calls the configured judge with
`GenerateConfig(max_tokens=64, temperature=0)` on every attempt. It requires one
error-free choice with stop reason `stop`; truncated or otherwise incomplete
outputs are invalid. One invalid-output retry adds only a format reminder, then
exhaustion raises `InvalidJudgmentError`. Transport errors propagate; Inspect's
provider-level transport retry policy is separate. Saved-answer clients must
likewise check output completion before calling the text parser.

No explanation is requested or stored in the score. Metadata retains the parsed
judgment and attempt count, while `Score.answer` remains the evaluated answer.
No GPU calls are needed for `tests/test_talemaader_scorer.py`.
