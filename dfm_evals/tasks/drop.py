from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.scorer import f1, mean, stderr
from inspect_evals.constants import DEFAULT_FEWSHOT_SEED
from inspect_evals.drop.drop import (
    DATASET_PATH,
    DROP_DATASET_REVISION,
    EVAL_VERSION,
    _sample_is_not_known_duplicate,
    drop_solver,
    extract_answer,
    record_to_sample,
)
from inspect_evals.utils.huggingface import hf_dataset

from ._sharding import shard_samples


@task(name="drop")
def drop(
    fewshot: int = 3,
    fewshot_seed: int = DEFAULT_FEWSHOT_SEED,
    fewshot_shuffle: bool = True,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Task:
    """Inspect DROP task with deterministic, disjoint scheduler sharding."""

    dataset = hf_dataset(
        path=DATASET_PATH,
        split="validation",
        sample_fields=record_to_sample,
        revision=DROP_DATASET_REVISION,
    ).filter(_sample_is_not_known_duplicate)
    dataset = shard_samples(
        dataset,
        name="drop",
        location=DATASET_PATH,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    return Task(
        dataset=dataset,
        solver=drop_solver(
            fewshot=fewshot,
            fewshot_seed=fewshot_seed,
            fewshot_shuffle=fewshot_shuffle,
        ),
        scorer=f1(extract_answer),
        metrics=[mean(), stderr(cluster="passage_hash")],
        version=EVAL_VERSION.comparability_version,
        metadata=EVAL_VERSION.to_metadata(),
    )
