from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Target

from dfm_evals.tasks.talemaader import scorer as module
from dfm_evals.tasks.talemaader.prompts import JUDGE_TEMPLATE_DA


def test_standalone_helpers_without_site_packages():
    import subprocess
    import sys
    from pathlib import Path

    path = Path(module.__file__).with_name("prompts.py")
    program = """
import importlib.util
import sys
spec = importlib.util.spec_from_file_location('standalone_judgment', sys.argv[1])
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
assert helper.parse_judgment('GRADE: P') == 0.5
assert 'meaning' in helper.build_judge_prompt(
    talemaade_udtryk='idiom', criterion='meaning', answer='answer')
try:
    helper.parse_judgment('explanation GRADE: C')
except helper.InvalidJudgmentError:
    pass
else:
    raise AssertionError('Invalid judgment accepted')
assert 'inspect_ai' not in sys.modules
assert 'dfm_evals' not in sys.modules
"""
    subprocess.run([sys.executable, "-I", "-S", "-c", program, str(path)], check=True)


@pytest.mark.parametrize("grade,value", [("C", 1), ("P", 0.5), ("I", 0)])
def test_parse(grade, value):
    assert module.parse_judgment(f"\nGRADE: {grade}\n") == value


@pytest.mark.parametrize(
    "text",
    [
        "",
        "C",
        "GRADE: c",
        "GRADE: X",
        "GRADE: CP",
        "GRADE: C\nGRADE: I",
        "Reason\nGRADE: C",
        "GRADE: C because",
        "```GRADE: C```",
    ],
)
def test_invalid_is_not_incorrect(text):
    with pytest.raises(module.InvalidJudgmentError):
        module.parse_judgment(text)


def test_prompt_preserves_template_and_criteria():
    assert module.build_judge_prompt(
        talemaade_udtryk="idiom", criterion="truth", answer="reply"
    ) == (
        JUDGE_TEMPLATE_DA.format(
            talemaade_udtryk="idiom",
            criterion="truth",
            answer="reply",
            instructions=module.JUDGE_INSTRUCTIONS_V2,
        )
    )


def state():
    return SimpleNamespace(
        metadata={"talemaade_udtryk": "idiom"},
        output=ModelOutput.from_content("evaluated", "saved answer"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("grade,value", [("C", 1), ("P", 0.5), ("I", 0)])
async def test_mock_judge_config_and_scores(monkeypatch, grade, value):
    judge = SimpleNamespace(
        generate=AsyncMock(
            return_value=ModelOutput.from_content("judge", f"GRADE: {grade}")
        )
    )
    calls = []
    monkeypatch.setattr(
        module, "get_model", lambda *a, **k: calls.append((a, k)) or judge
    )
    score = await module.model_graded_fact_v2(model="mockllm/judge")(
        state(), Target("truth")
    )
    assert score.value == value
    assert score.answer == "saved answer"
    assert score.explanation is None
    assert calls == [(("mockllm/judge",), {})]
    config = judge.generate.call_args.kwargs["config"]
    assert config.max_tokens == 64 and config.temperature == 0
    assert judge.generate.await_count == 1


@pytest.mark.asyncio
async def test_invalid_retries_once_then_succeeds(monkeypatch):
    judge = SimpleNamespace(
        generate=AsyncMock(
            side_effect=[
                ModelOutput.from_content("judge", "explanation GRADE: C"),
                ModelOutput.from_content("judge", "GRADE: P"),
            ]
        )
    )
    monkeypatch.setattr(module, "get_model", lambda **k: judge)
    result = await module.model_graded_fact_v2()(state(), Target("truth"))
    assert result.value == 0.5 and result.metadata["judge_attempts"] == 2
    assert judge.generate.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content,stop", [("invalid", "stop"), ("GRADE: C", "max_tokens")]
)
async def test_invalid_or_truncated_exhaustion_raises(monkeypatch, content, stop):
    judge = SimpleNamespace(
        generate=AsyncMock(
            return_value=ModelOutput.from_content("judge", content, stop_reason=stop)
        )
    )
    monkeypatch.setattr(module, "get_model", lambda **k: judge)
    with pytest.raises(module.InvalidJudgmentError, match="after 2 attempts"):
        await module.model_graded_fact_v2()(state(), Target("truth"))
    assert judge.generate.await_count == 2


@pytest.mark.asyncio
async def test_transport_error_not_converted_to_grade(monkeypatch):
    judge = SimpleNamespace(generate=AsyncMock(side_effect=RuntimeError("unavailable")))
    monkeypatch.setattr(module, "get_model", lambda **k: judge)
    with pytest.raises(RuntimeError, match="unavailable"):
        await module.model_graded_fact_v2()(state(), Target("truth"))
    assert judge.generate.await_count == 1


def test_task_changes_only_judge(monkeypatch):
    from importlib import import_module

    from inspect_ai._util.registry import registry_info
    from inspect_ai.dataset import MemoryDataset

    task_module = import_module("dfm_evals.tasks.talemaader.task")
    sample = task_module.record_to_sample(
        {"id": "example", "talemaade_udtryk": "idiom", "ddo_definition": "meaning"}
    )
    monkeypatch.setattr(
        task_module, "_memory_dataset", lambda **kwargs: MemoryDataset([sample])
    )
    result = task_module._talemaader_task(judge_model="mockllm/judge")
    assert sample.input == task_module.PROMPT_TEMPLATE_DA.format(
        talemaade_udtryk="idiom"
    )
    assert sample.target == ["meaning"]
    assert sample.metadata == {"talemaade_udtryk": "idiom"}
    assert registry_info(result.scorer[0]).name.endswith("model_graded_fact_v2")
