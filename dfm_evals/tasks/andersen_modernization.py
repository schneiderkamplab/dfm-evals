from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import generate

from ..scorers.chrf import chrf3pp
from ..scorers.gleu import gleu
from ._sharding import shard_samples

DEFAULT_DATA_PATH = "/work/dfm/andersen/pairs_chunked_val.jsonl"
DEFAULT_MAX_GEN_TOKS = 1536
DEFAULT_TEMPERATURE = 0.0
EXPECTED_ROLES = ("system", "user", "assistant")


@task(name="andersen-modernization")
def andersen_modernization(
    data_path: str = DEFAULT_DATA_PATH,
    max_gen_toks: int = DEFAULT_MAX_GEN_TOKS,
    temperature: float = DEFAULT_TEMPERATURE,
    limit: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Task:
    """Evaluate zero-shot modernization of held-out H.C. Andersen passages."""

    if max_gen_toks < 1:
        raise ValueError("`max_gen_toks` must be >= 1.")
    records = load_records(Path(data_path))
    if limit is not None:
        records = records[:limit]
    samples = [record_to_sample(record) for record in records]
    return Task(
        dataset=shard_samples(
            samples,
            name="Andersen modernization",
            location=data_path,
            num_shards=num_shards,
            shard_index=shard_index,
        ),
        # There are deliberately no demonstrations or few-shot solvers here.
        solver=[generate(max_tokens=max_gen_toks, temperature=temperature)],
        scorer=[gleu(ignore_case=False), chrf3pp()],
    )


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            validate_record(row, path=path, line_number=line_number)
            records.append(row)
    return records


def validate_record(record: dict[str, Any], *, path: Path, line_number: int) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"{path}:{line_number}: expected exactly three messages")
    roles = tuple(message.get("role") for message in messages)
    if roles != EXPECTED_ROLES:
        raise ValueError(f"{path}:{line_number}: unexpected roles {roles}")
    for message in messages:
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"{path}:{line_number}: empty message content")


def record_to_sample(record: dict[str, Any]) -> Sample:
    system, user, assistant = record["messages"]
    story_id = str(record.get("id") or "unknown")
    chunk_idx = int(record.get("chunk_idx", 0))
    target = assistant["content"].strip()
    return Sample(
        id=f"{story_id}:{chunk_idx}",
        input=[
            ChatMessageSystem(content=system["content"].strip()),
            ChatMessageUser(content=user["content"].strip()),
        ],
        target=[target],
        metadata={
            "story_id": story_id,
            "title": record.get("title"),
            "year": record.get("year"),
            "genre": record.get("genre"),
            "chunk_idx": chunk_idx,
            "reference": target,
        },
    )
