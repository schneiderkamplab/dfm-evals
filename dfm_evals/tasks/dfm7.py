from __future__ import annotations

import json
import re
import string
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    mean,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate

from dfm_evals.tasks._sharding import shard_samples
from dfm_evals.tasks.bfcl.bfcl import _collect_assistant_calls
from dfm_evals.tasks.bfcl.solver import bfcl_solver
from dfm_evals.tasks.ifeval_da import instruction_following, record_to_sample as ifeval_record_to_sample
from dfm_evals.tasks.piqa import _extract_choice as extract_piqa_choice


GSM_DA_PROMPT = """Løs regneopgaven. Svar kort, og afslut med den endelige numeriske værdi.

Opgave:
{question}"""

MCQ_DA_PROMPT = """Vælg det korrekte svar. Svar kun med bogstavet for den bedste svarmulighed.

Spørgsmål:
{question}

Svarmuligheder:
{choices}"""

QA_DA_PROMPT = """Besvar spørgsmålet kort og præcist på dansk.

Spørgsmål:
{question}"""

SQL_DA_PROMPT = """Skriv en SQL-forespørgsel, der løser opgaven. Returner kun SQL.

Opgave:
{instruction}"""

LINGUISTIC_QUALITY_PROMPT = """Vurder den sproglige kvalitet af teksten.
Svar kun med den korrekte label fra opgaven.

Tekst:
{text}"""

_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:[.,]\d+)?|\d{1,3}(?:[.,]\d{3})+)")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_CHOICE_RE = re.compile(r"\b([A-ZÆØÅ])\b", re.IGNORECASE)


@task(name="multi-ifeval")
def multi_ifeval(
    dataset_id: str = "danish-foundation-models/multi-ifeval",
    language: str = "da",
    split: str = "test",
    shuffle: bool = False,
    seed: int | None = None,
    limit: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Task:
    """Multilingual IFEval wrapper; keep language slices separate in metrics."""
    samples = _load_hf_samples(
        dataset_id=dataset_id,
        name=language,
        split=split,
        mapper=ifeval_record_to_sample,
        shuffle=shuffle,
        seed=seed,
        limit=limit,
    )
    return Task(
        dataset=shard_samples(
            samples,
            name=f"multi-ifeval-{language}",
            location=f"{dataset_id}:{language}:{split}",
            num_shards=num_shards,
            shard_index=shard_index,
            shuffled=shuffle,
        ),
        solver=[generate()],
        scorer=instruction_following(),
    )


@task(name="gsm8k-da")
def gsm8k_da(
    dataset_id: str = "synquid/gsm8k-da",
    split: str = "test",
    max_gen_toks: int = 512,
    temperature: float = 0.0,
    limit: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Task:
    samples = _load_hf_samples(
        dataset_id=dataset_id,
        name=None,
        split=split,
        mapper=lambda row: _gsm_sample(row, id_field="id"),
        limit=limit,
    )
    return _generative_task(
        samples=samples,
        name="GSM8K-da",
        location=f"{dataset_id}:{split}",
        max_gen_toks=max_gen_toks,
        temperature=temperature,
        scorer_=numeric_answer_scorer(),
        num_shards=num_shards,
        shard_index=shard_index,
    )


@task(name="gsm-symbolic")
def gsm_symbolic(
    dataset_id: str = "danish-foundation-models/multilingual-gsm-symbolic",
    language: str = "dan",
    split: str = "test_original",
    max_gen_toks: int = 512,
    temperature: float = 0.0,
    limit: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Task:
    samples = _load_hf_samples(
        dataset_id=dataset_id,
        name=language,
        split=split,
        mapper=lambda row: _gsm_sample(row, id_field="source_id", target_field="target"),
        limit=limit,
    )
    return _generative_task(
        samples=samples,
        name=f"GSM-Symbolic-{language}-{split}",
        location=f"{dataset_id}:{language}:{split}",
        max_gen_toks=max_gen_toks,
        temperature=temperature,
        scorer_=numeric_answer_scorer(),
        num_shards=num_shards,
        shard_index=shard_index,
    )


@task(name="kaenguruen")
def kaenguruen(
    dataset_id: str = "danish-foundation-models/kaenguruen",
    split: str = "test",
    max_gen_toks: int = 16,
    temperature: float = 0.0,
    limit: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Task:
    samples = _load_hf_samples(
        dataset_id=dataset_id,
        name=None,
        split=split,
        mapper=_mcq_sample_flexible,
        limit=limit,
    )
    return _generative_task(
        samples=samples,
        name="Kaenguruen",
        location=f"{dataset_id}:{split}",
        max_gen_toks=max_gen_toks,
        temperature=temperature,
        scorer_=mcq_letter_scorer(),
        num_shards=num_shards,
        shard_index=shard_index,
    )


@task(name="global-piqa-da")
def global_piqa_da(
    dataset_id: str = "danish-foundation-models/global-piqa-da",
    split: str = "train",
    max_gen_toks: int = 8,
    temperature: float = 0.0,
    limit: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Task:
    samples = _load_hf_samples(
        dataset_id=dataset_id,
        name=None,
        split=split,
        mapper=_global_piqa_sample,
        limit=limit,
        skip_none=True,
    )
    if not samples:
        raise ValueError(
            f"No accepted/annotated PIQA rows found in {dataset_id}:{split}."
        )
    return _generative_task(
        samples=samples,
        name="Global-PIQA-da",
        location=f"{dataset_id}:{split}",
        max_gen_toks=max_gen_toks,
        temperature=temperature,
        scorer_=piqa_hf_scorer(),
        num_shards=num_shards,
        shard_index=shard_index,
    )


@task(name="linguistic-quality-da")
def linguistic_quality_da(
    dataset_id: str = "danish-foundation-models/linguistic-quality",
    split: str = "train",
    max_gen_toks: int = 16,
    temperature: float = 0.0,
    limit: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Task:
    samples = _load_hf_samples(
        dataset_id=dataset_id,
        name=None,
        split=split,
        mapper=_linguistic_quality_sample,
        limit=limit,
    )
    return _generative_task(
        samples=samples,
        name="Linguistic-quality-da",
        location=f"{dataset_id}:{split}",
        max_gen_toks=max_gen_toks,
        temperature=temperature,
        scorer_=label_scorer(),
        num_shards=num_shards,
        shard_index=shard_index,
    )


@task(name="sdu-daisy")
def sdu_daisy(
    dataset_id: str = "schneiderkamplab/SDU-Daisy",
    split: str = "train",
    max_gen_toks: int = 64,
    temperature: float = 0.0,
    limit: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Task:
    samples = _load_hf_samples(
        dataset_id=dataset_id,
        name=None,
        split=split,
        mapper=_sdu_daisy_sample,
        limit=limit,
    )
    return _generative_task(
        samples=samples,
        name="SDU-Daisy",
        location=f"{dataset_id}:{split}",
        max_gen_toks=max_gen_toks,
        temperature=temperature,
        scorer_=normalized_text_scorer(),
        num_shards=num_shards,
        shard_index=shard_index,
    )


@task(name="da-bird")
def da_bird(
    dataset_id: str = "oliverkinch/da-bird",
    max_gen_toks: int = 512,
    temperature: float = 0.0,
    limit: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Task:
    samples = _load_da_bird_samples(dataset_id=dataset_id, limit=limit)
    return _generative_task(
        samples=samples,
        name="DA-BIRD",
        location=dataset_id,
        max_gen_toks=max_gen_toks,
        temperature=temperature,
        scorer_=sql_exact_scorer(),
        num_shards=num_shards,
        shard_index=shard_index,
    )


@task(name="danish-tool-calling-benchmark")
def danish_tool_calling_benchmark(
    dataset_id: str = "schneiderkamplab/danish-tool-calling-benchmark",
    split: str = "danish",
    limit: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Task:
    samples = _load_hf_samples(
        dataset_id=dataset_id,
        name=None,
        split=split,
        mapper=_tool_call_sample,
        limit=limit,
    )
    return Task(
        dataset=shard_samples(
            samples,
            name=f"tool-calling-{split}",
            location=f"{dataset_id}:{split}",
            num_shards=num_shards,
            shard_index=shard_index,
        ),
        solver=bfcl_solver(),
        scorer=tool_call_scorer(),
    )


def _load_hf_samples(
    *,
    dataset_id: str,
    name: str | None,
    split: str,
    mapper: Any,
    shuffle: bool = False,
    seed: int | None = None,
    limit: int | None = None,
    skip_none: bool = False,
) -> list[Sample]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, name, split=split) if name else load_dataset(dataset_id, split=split)
    rows = [dict(row) for row in dataset]
    if shuffle:
        import random

        random.Random(seed).shuffle(rows)
    if limit is not None:
        rows = rows[:limit]

    samples: list[Sample] = []
    for row in rows:
        sample = mapper(row)
        if sample is None and skip_none:
            continue
        if not isinstance(sample, Sample):
            raise ValueError(f"Could not map row from {dataset_id}:{split}: {row}")
        samples.append(sample)
    return samples


def _generative_task(
    *,
    samples: list[Sample],
    name: str,
    location: str,
    max_gen_toks: int,
    temperature: float,
    scorer_: Scorer,
    num_shards: int,
    shard_index: int,
) -> Task:
    return Task(
        dataset=shard_samples(
            samples,
            name=name,
            location=location,
            num_shards=num_shards,
            shard_index=shard_index,
        ),
        solver=[generate(max_tokens=max_gen_toks, temperature=temperature)],
        scorer=scorer_,
    )


def _gsm_sample(
    row: Mapping[str, Any],
    *,
    id_field: str,
    target_field: str | None = None,
) -> Sample:
    question = _require_any_string(row, ["question", "prompt", "problem"])
    target = str(row.get(target_field or "target") or _extract_gold_number(str(row.get("answer", ""))) or "").strip()
    if not target:
        raise ValueError(f"Could not extract numeric target from row: {row}")
    return Sample(
        id=_optional_id(row, id_field),
        input=GSM_DA_PROMPT.format(question=question),
        target=target,
        metadata={"gold_answer": row.get("answer"), "target": target},
    )


def _mcq_sample_flexible(row: Mapping[str, Any]) -> Sample:
    question = _require_any_string(row, ["question", "prompt", "problem", "input"])
    choices = _extract_choices(row)
    answer = _extract_mcq_answer(row, choices)
    choice_text = "\n".join(f"{letter}. {text}" for letter, text in choices)
    return Sample(
        id=_optional_id(row, "id"),
        input=MCQ_DA_PROMPT.format(question=question, choices=choice_text),
        target=answer,
        metadata={"choices": choices},
    )


def _global_piqa_sample(row: Mapping[str, Any]) -> Sample | None:
    question = _first_response(row, "question.responses")
    correct = _first_response(row, "correct_answer.responses")
    wrong = _first_response(row, "wrong_answer.responses")
    status = str(row.get("status") or "").lower()
    if not question or not correct or not wrong or status not in {"accepted", "completed", "validated"}:
        return None
    return Sample(
        id=_optional_id(row, "id"),
        input=(
            "Du får et spørgsmål om fysisk hverdagsfornuft og to svarmuligheder.\n"
            "Vælg den bedste løsning.\n\n"
            f"Spørgsmål:\n{question}\n\nMulighed A:\n{correct}\n\nMulighed B:\n{wrong}\n\n"
            "Svar kun med A eller B."
        ),
        target="A",
        metadata={"solution0": correct, "solution1": wrong},
    )


def _linguistic_quality_sample(row: Mapping[str, Any]) -> Sample:
    text = _require_any_string(row, ["text", "sentence", "input", "prompt"])
    label = _require_any_string(row, ["label", "target", "answer", "quality"])
    return Sample(
        id=_optional_id(row, "id"),
        input=LINGUISTIC_QUALITY_PROMPT.format(text=text),
        target=_normalize_label(label),
        metadata={"raw_label": label},
    )


def _sdu_daisy_sample(row: Mapping[str, Any]) -> Sample:
    question = _require_any_string(row, ["Question", "question"])
    answer = _require_any_string(row, ["Answer", "answer"])
    return Sample(
        id=_optional_id(row, "id"),
        input=QA_DA_PROMPT.format(question=question),
        target=answer,
        metadata={"subject": row.get("Subject")},
    )


def _load_da_bird_samples(dataset_id: str, limit: int | None) -> list[Sample]:
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo_id=dataset_id, repo_type="dataset"))
    task_dirs = sorted(path for path in root.iterdir() if (path / "instruction.md").is_file())
    if limit is not None:
        task_dirs = task_dirs[:limit]

    samples: list[Sample] = []
    for task_dir in task_dirs:
        instruction = (task_dir / "instruction.md").read_text(encoding="utf-8").strip()
        gold_sql = (task_dir / "tests" / "gold.sql").read_text(encoding="utf-8").strip()
        samples.append(
            Sample(
                id=task_dir.name,
                input=SQL_DA_PROMPT.format(instruction=instruction),
                target=gold_sql,
                metadata={"task_dir": str(task_dir)},
            )
        )
    return samples


def _tool_call_sample(row: Mapping[str, Any]) -> Sample:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("tool-calling row requires messages list")
    user = next((m for m in messages if isinstance(m, dict) and m.get("role") == "user"), None)
    assistant = next((m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"), None)
    if not isinstance(user, dict) or not isinstance(assistant, dict):
        raise ValueError("tool-calling row requires user and assistant messages")
    expected_calls = _normalize_expected_tool_calls(assistant.get("tool_calls"))
    tools = _tool_schemas_for_calls(expected_calls)
    return Sample(
        input=[ChatMessageUser(content=str(user.get("content") or ""))],
        target=json.dumps(expected_calls, sort_keys=True, ensure_ascii=False),
        metadata={"tools": tools, "expected_calls": expected_calls},
    )


@scorer(metrics=[accuracy()])
def numeric_answer_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        expected = _normalize_number(target.text)
        predicted = _normalize_number(_extract_predicted_number(state.output.completion))
        return Score(
            value=CORRECT if predicted == expected and expected else INCORRECT,
            answer=predicted,
            explanation=f"predicted={predicted!r}, expected={expected!r}",
        )

    return score


@scorer(metrics=[accuracy()])
def mcq_letter_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        predicted = _extract_choice_letter(state.output.completion)
        expected = target.text.strip().upper()
        return Score(
            value=CORRECT if predicted == expected else INCORRECT,
            answer=predicted or "",
            explanation=f"predicted={predicted!r}, expected={expected!r}",
        )

    return score


@scorer(metrics=[accuracy()])
def piqa_hf_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        predicted = extract_piqa_choice(
            state.output.completion,
            str(state.metadata["solution0"]),
            str(state.metadata["solution1"]),
        )
        expected = target.text.strip().upper()
        return Score(
            value=CORRECT if predicted == expected else INCORRECT,
            answer=predicted or "",
            explanation=f"predicted={predicted!r}, expected={expected!r}",
        )

    return score


@scorer(metrics=[accuracy()])
def label_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        predicted = _normalize_label(state.output.completion)
        expected = _normalize_label(target.text)
        return Score(value=CORRECT if predicted == expected else INCORRECT, answer=predicted)

    return score


@scorer(metrics={"exact": [mean(), stderr()]})
def normalized_text_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        prediction = _normalize_text(state.output.completion)
        references = [_normalize_text(item) for item in target if item]
        return Score(value={"exact": float(prediction in references)}, answer=state.output.completion)

    return score


@scorer(metrics=[accuracy()])
def sql_exact_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        predicted = _normalize_sql(state.output.completion)
        expected = _normalize_sql(target.text)
        return Score(
            value=CORRECT if predicted == expected and expected else INCORRECT,
            answer=state.output.completion,
            explanation=f"normalized_predicted={predicted!r}, normalized_expected={expected!r}",
        )

    return score


@scorer(metrics=[accuracy()])
def tool_call_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        expected = state.metadata["expected_calls"]
        assistant_messages = [m for m in state.messages if isinstance(m, ChatMessageAssistant)]
        _, actual = _collect_assistant_calls(state)
        expected_norm = _normalize_call_list(expected)
        actual_norm = _normalize_call_list(actual)
        return Score(
            value=CORRECT if actual_norm == expected_norm else INCORRECT,
            answer=json.dumps(actual_norm, sort_keys=True, ensure_ascii=False),
            explanation=(
                f"assistant_messages={len(assistant_messages)}, "
                f"expected={json.dumps(expected_norm, sort_keys=True, ensure_ascii=False)}"
            ),
        )

    return score


def _require_any_string(row: Mapping[str, Any], fields: list[str]) -> str:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"None of {fields} contained a non-empty string in row: {row}")


def _optional_id(row: Mapping[str, Any], field: str) -> str | None:
    value = row.get(field)
    return None if value is None else str(value)


def _extract_gold_number(answer: str) -> str:
    if "####" in answer:
        return answer.rsplit("####", 1)[1].strip()
    return _extract_predicted_number(answer)


def _extract_predicted_number(text: str) -> str:
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1]
    matches = _NUMBER_RE.findall(text)
    return matches[-1] if matches else ""


def _normalize_number(text: str) -> str:
    text = text.strip().replace(" ", "")
    text = text.replace(",", ".")
    return text.rstrip(".")


def _extract_choice_letter(text: str) -> str | None:
    matches = [m.group(1).upper() for m in _CHOICE_RE.finditer(text)]
    unique = list(dict.fromkeys(matches))
    return unique[0] if len(unique) == 1 else None


def _extract_choices(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    raw_choices = row.get("choices") or row.get("options") or row.get("answers")
    if isinstance(raw_choices, list):
        out = []
        for index, value in enumerate(raw_choices):
            letter = string.ascii_uppercase[index]
            out.append((letter, str(value).strip()))
        if out:
            return out
    out = []
    for letter in string.ascii_uppercase[:8]:
        for key in (letter, letter.lower(), f"option_{letter}", f"choice_{letter}"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                out.append((letter, value.strip()))
                break
    if out:
        return out
    raise ValueError(f"Could not extract MCQ choices from row: {row}")


def _extract_mcq_answer(row: Mapping[str, Any], choices: list[tuple[str, str]]) -> str:
    for field in ("answer", "target", "label", "correct", "correct_answer"):
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, int) and 0 <= value < len(choices):
            return choices[value][0]
        text = str(value).strip()
        letter = text[:1].upper()
        if letter in {choice[0] for choice in choices}:
            return letter
        for choice_letter, choice_text in choices:
            if _normalize_text(text) == _normalize_text(choice_text):
                return choice_letter
    raise ValueError(f"Could not extract MCQ answer from row: {row}")


def _first_response(row: Mapping[str, Any], field: str) -> str | None:
    value = row.get(field)
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _normalize_label(text: str) -> str:
    return _normalize_text(text).replace(" ", "_")


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _normalize_sql(text: str) -> str:
    cleaned = text.strip().strip("`")
    if cleaned.lower().startswith("sql\n"):
        cleaned = cleaned[4:]
    return re.sub(r"\s+", " ", cleaned).strip().rstrip(";").lower()


def _normalize_expected_tool_calls(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    calls = []
    for call in raw:
        if not isinstance(call, dict):
            continue
        args = call.get("arguments")
        if isinstance(args, dict):
            args = {k: v for k, v in args.items() if v is not None}
        else:
            args = {}
        calls.append({"function": str(call.get("name") or call.get("function") or ""), "arguments": args})
    return calls


def _tool_schemas_for_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    schemas = []
    for call in calls:
        properties = {}
        for key, value in call["arguments"].items():
            properties[key] = {"type": "string" if isinstance(value, str) else "number"}
        schemas.append(
            {
                "name": call["function"],
                "description": f"Tool function {call['function']}.",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                },
            }
        )
    return schemas


def _normalize_call_list(calls: Any) -> list[dict[str, Any]]:
    if not isinstance(calls, list):
        return []
    normalized = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        normalized.append(
            {
                "function": str(call.get("function") or call.get("name") or ""),
                "arguments": _normalize_json(call.get("arguments") or {}),
            }
        )
    return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))


def _normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_json(v) for k, v in sorted(value.items()) if v is not None}
    if isinstance(value, list):
        return [_normalize_json(v) for v in value]
    return value
