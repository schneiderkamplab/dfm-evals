from __future__ import annotations

import json
from pathlib import Path

from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from dfm_evals.tasks.andersen_modernization import (
    andersen_modernization,
    load_records,
    record_to_sample,
)


def example_record() -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": "Modernisér teksten."},
            {"role": "user", "content": "Han saae en Fugl."},
            {"role": "assistant", "content": "Han så en fugl."},
        ],
        "id": "historien",
        "title": "Historien",
        "year": "1850",
        "genre": "tale",
        "chunk_idx": 2,
    }


def test_record_to_sample_preserves_system_and_user_messages() -> None:
    sample = record_to_sample(example_record())
    assert sample.id == "historien:2"
    assert isinstance(sample.input[0], ChatMessageSystem)
    assert isinstance(sample.input[1], ChatMessageUser)
    assert sample.input[0].content == "Modernisér teksten."
    assert sample.input[1].content == "Han saae en Fugl."
    assert sample.target == ["Han så en fugl."]


def test_task_is_zero_shot_and_uses_validation_rows(tmp_path: Path) -> None:
    path = tmp_path / "validation.jsonl"
    path.write_text(json.dumps(example_record(), ensure_ascii=False) + "\n")
    task = andersen_modernization(data_path=str(path))
    assert len(task.dataset) == 1
    assert len(task.solver) == 1
    assert [scorer.__registry_info__.name for scorer in task.scorer] == [
        "gleu",
        "chrf3pp",
    ]


def test_load_records_rejects_non_chat_rows(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"messages": []}) + "\n")
    try:
        load_records(path)
    except ValueError as exc:
        assert "expected exactly three messages" in str(exc)
    else:
        raise AssertionError("invalid row was accepted")
