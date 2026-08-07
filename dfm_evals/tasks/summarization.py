from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU, CHRF
from ._sharding import shard_samples

DEFAULT_GOVREPORT_DATASET_ID = "ccdv/govreport-summarization"
DEFAULT_GOVREPORT_CONFIG = "document"
DEFAULT_GOVREPORT_SPLIT = "test"
DEFAULT_NORDJYLLANDNEWS_PATH = (
    "data/downloads/datasets/danish_dynaword/data/nordjyllandnews/"
    "nordjyllandnews.parquet"
)

GOVREPORT_PROMPT_TEMPLATE = """Summarize the following government report.

Report:
{{document}}

Summary:"""

NORDJYLLANDNEWS_PROMPT_TEMPLATE = """Lav et kort referat af nedenstående nyhedsartikel.

Artikel:
{{document}}

Referat:"""


@task(name="govreport")
def govreport(
    dataset_id: str = DEFAULT_GOVREPORT_DATASET_ID,
    dataset_config: str = DEFAULT_GOVREPORT_CONFIG,
    split: str = DEFAULT_GOVREPORT_SPLIT,
    prompt_template: str = GOVREPORT_PROMPT_TEMPLATE,
    max_gen_toks: int = 512,
    max_report_chars: int | None = None,
    temperature: float = 0.0,
    limit: int | None = None,
    include_bertscore: bool = True,
    bertscore_model: str = "xlm-roberta-large",
    bertscore_device: str = "auto",
    num_shards: int = 1,
    shard_index: int = 0,
) -> Task:
    samples = []
    for index, row in enumerate(
        _load_hf_records(dataset_id=dataset_id, config=dataset_config, split=split)
    ):
        report = _require_string(row, "report")
        if max_report_chars is not None and max_report_chars > 0:
            report = report[:max_report_chars]
        samples.append(
            Sample(
                id=str(index),
                input=prompt_template.replace("{{document}}", report),
                target=[_require_string(row, "summary")],
                metadata={"dataset_id": dataset_id, "split": split},
            )
        )
    if limit is not None:
        samples = samples[:limit]

    return Task(
        dataset=shard_samples(
            samples=samples,
            name="GovReport",
            location=f"{dataset_id}:{dataset_config}:{split}",
            num_shards=num_shards,
            shard_index=shard_index,
        ),
        solver=[generate(max_tokens=max_gen_toks, temperature=temperature)],
        scorer=[
            summarization_scorer(
                include_bertscore=include_bertscore,
                bertscore_model=bertscore_model,
                bertscore_lang="en",
                bertscore_device=bertscore_device,
            )
        ],
    )


@task(name="nordjyllandnews")
def nordjyllandnews(
    data_path: str = DEFAULT_NORDJYLLANDNEWS_PATH,
    prompt_template: str = NORDJYLLANDNEWS_PROMPT_TEMPLATE,
    max_gen_toks: int = 128,
    temperature: float = 0.0,
    max_samples: int | None = 1000,
    include_bertscore: bool = True,
    bertscore_model: str = "xlm-roberta-large",
    bertscore_device: str = "auto",
    num_shards: int = 1,
    shard_index: int = 0,
) -> Task:
    pairs = _evenly_spaced(_load_nordjyllandnews_pairs(Path(data_path)), max_samples)
    samples = [
        Sample(
            id=str(index),
            input=prompt_template.replace("{{document}}", article),
            target=[summary],
            metadata={"source": "nordjyllandnews"},
        )
        for index, (article, summary) in enumerate(pairs)
    ]

    return Task(
        dataset=shard_samples(
            samples=samples,
            name="NordjyllandNews",
            location=data_path,
            num_shards=num_shards,
            shard_index=shard_index,
        ),
        solver=[generate(max_tokens=max_gen_toks, temperature=temperature)],
        scorer=[
            summarization_scorer(
                include_bertscore=include_bertscore,
                bertscore_model=bertscore_model,
                bertscore_lang="da",
                bertscore_device=bertscore_device,
            )
        ],
    )


@scorer(
    metrics={
        "rouge1": [mean(), stderr()],
        "rouge2": [mean(), stderr()],
        "rougeL": [mean(), stderr()],
        "rougeLsum": [mean(), stderr()],
        "bleu": [mean(), stderr()],
        "chrf3": [mean(), stderr()],
        "chrf3pp": [mean(), stderr()],
        "bertscore_precision": [mean(), stderr()],
        "bertscore_recall": [mean(), stderr()],
        "bertscore_f1": [mean(), stderr()],
    },
    name="summarization",
)
def summarization_scorer(
    *,
    include_bertscore: bool = True,
    bertscore_model: str = "xlm-roberta-large",
    bertscore_lang: str = "da",
    bertscore_device: str = "auto",
) -> Scorer:
    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
    )

    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion.strip()
        references = [item.strip() for item in target if item.strip()]
        reference = references[0] if references else ""

        values = _sentence_summarization_scores(
            rouge=rouge,
            prediction=prediction,
            reference=reference,
        )
        if include_bertscore:
            precision, recall, f1 = await asyncio.to_thread(
                _score_bertscore,
                prediction,
                reference,
                bertscore_model,
                bertscore_lang,
                bertscore_device,
            )
        else:
            precision, recall, f1 = 0.0, 0.0, 0.0
        values.update(
            {
                "bertscore_precision": precision,
                "bertscore_recall": recall,
                "bertscore_f1": f1,
            }
        )

        return Score(
            value=values,
            answer=prediction,
            metadata={"reference": reference},
        )

    return score


def _load_hf_records(
    *, dataset_id: str, config: str, split: str
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, config, split=split)
    return [dict(row) for row in dataset]


def _load_nordjyllandnews_pairs(path: Path) -> list[tuple[str, str]]:
    from datasets import load_dataset

    dataset = load_dataset("parquet", data_files=str(path), split="train")
    pairs: list[tuple[str, str]] = []
    for row in dataset:
        text = row.get("text")
        if not isinstance(text, str) or "\n\nReferat:\n" not in text:
            continue

        article, summary = text.split("\n\nReferat:\n", 1)
        if article.startswith("Lav et referat af nedenstående tekst:\n\nTekst:\n"):
            article = article.split("\n\nTekst:\n", 1)[1]

        article = article.strip()
        summary = summary.strip()
        if article and summary:
            pairs.append((article, summary))
    return pairs


def _evenly_spaced(
    items: list[tuple[str, str]], max_samples: int | None
) -> list[tuple[str, str]]:
    if max_samples is None or len(items) <= max_samples:
        return items
    if max_samples < 1:
        raise ValueError("`max_samples` must be positive or None.")
    if max_samples == 1:
        return [items[0]]

    last_index = len(items) - 1
    return [items[round(i * last_index / (max_samples - 1))] for i in range(max_samples)]


def _sentence_summarization_scores(
    *, rouge: rouge_scorer.RougeScorer, prediction: str, reference: str
) -> dict[str, float]:
    rouge_scores = rouge.score(reference, prediction) if reference else {}
    bleu = BLEU(effective_order=True).sentence_score(prediction, [reference]).score
    chrf3 = CHRF(beta=3, word_order=0).sentence_score(prediction, [reference]).score
    chrf3pp = CHRF(beta=3, word_order=2).sentence_score(prediction, [reference]).score
    return {
        "rouge1": float(rouge_scores.get("rouge1", _ZERO_ROUGE).fmeasure),
        "rouge2": float(rouge_scores.get("rouge2", _ZERO_ROUGE).fmeasure),
        "rougeL": float(rouge_scores.get("rougeL", _ZERO_ROUGE).fmeasure),
        "rougeLsum": float(rouge_scores.get("rougeLsum", _ZERO_ROUGE).fmeasure),
        "bleu": float(bleu),
        "chrf3": float(chrf3),
        "chrf3pp": float(chrf3pp),
    }


class _ZeroRouge:
    fmeasure = 0.0


_ZERO_ROUGE = _ZeroRouge()


_BERTSCORER_CACHE: dict[tuple[str, str, str], Any] = {}
_BERTSCORER_LOCK = Lock()


def _score_bertscore(
    prediction: str,
    reference: str,
    model_type: str,
    lang: str,
    device: str,
) -> tuple[float, float, float]:
    if not prediction or not reference:
        return 0.0, 0.0, 0.0

    scorer_instance = _get_bertscorer(
        model_type=model_type,
        lang=lang,
        device=_resolve_bertscore_device(device),
    )
    precision, recall, f1 = scorer_instance.score([prediction], [reference])
    return float(precision[0]), float(recall[0]), float(f1[0])


def _get_bertscorer(*, model_type: str, lang: str, device: str) -> Any:
    key = (model_type, lang, device)
    with _BERTSCORER_LOCK:
        scorer_instance = _BERTSCORER_CACHE.get(key)
        if scorer_instance is None:
            from bert_score import BERTScorer

            scorer_instance = BERTScorer(
                model_type=model_type,
                lang=lang,
                device=device,
                rescale_with_baseline=False,
            )
            _BERTSCORER_CACHE[key] = scorer_instance
        return scorer_instance


def _resolve_bertscore_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized != "auto":
        return normalized
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _require_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Record field '{field}' must be a non-empty string.")
    return value
