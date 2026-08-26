from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

QADatasetName = Literal["squad", "hotpotqa"]

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
HOTPOTQA_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
    "refs%2Fconvert%2Fparquet/distractor/validation/0000.parquet"
)


@dataclass(frozen=True)
class QADocument:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class QAExample:
    question: str
    answers: list[str]
    documents: list[QADocument]


@dataclass(frozen=True)
class QABundle:
    examples: list[QAExample]
    distractor_documents: list[QADocument]


@lru_cache(maxsize=4)
def load_qa_bundle(dataset: QADatasetName) -> QABundle:
    suffix = ".parquet" if dataset == "hotpotqa" else ".json"
    cache_file = _cache_dir() / f"{dataset}{suffix}"
    if not cache_file.is_file():
        _download_dataset(dataset, cache_file)

    if dataset == "squad":
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        return _parse_squad_bundle(payload)
    if dataset == "hotpotqa":
        return _parse_hotpotqa_parquet(cache_file)
    raise ValueError(f"Unsupported QA dataset {dataset!r}")


def _cache_dir() -> Path:
    override = os.environ.get("DFM_EVALS_RULER_QA_CACHE")
    if override:
        root = Path(override).expanduser()
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
        root = base / "dfm_evals" / "ruler" / "qa"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _download_dataset(dataset: QADatasetName, target: Path) -> None:
    url = SQUAD_URL if dataset == "squad" else HOTPOTQA_URL
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "dfm-evals-ruler/1.0"},
    )
    with urllib.request.urlopen(request) as response:
        data = response.read()
    temporary = target.with_suffix(f"{target.suffix}.part")
    temporary.write_bytes(data)
    temporary.replace(target)


def _parse_squad_bundle(payload: object) -> QABundle:
    if not isinstance(payload, dict):
        raise ValueError("Invalid SQuAD payload: expected an object.")

    raw_articles = payload.get("data")
    if not isinstance(raw_articles, list):
        raise ValueError("Invalid SQuAD payload: missing `data` list.")

    examples: list[QAExample] = []
    documents: list[QADocument] = []

    for article_index, article in enumerate(raw_articles):
        if not isinstance(article, dict):
            continue
        title = _clean_text(article.get("title")) or f"SQuAD Article {article_index + 1}"
        raw_paragraphs = article.get("paragraphs")
        if not isinstance(raw_paragraphs, list):
            continue

        for paragraph_index, paragraph in enumerate(raw_paragraphs):
            if not isinstance(paragraph, dict):
                continue
            context = _clean_text(paragraph.get("context"))
            if not context:
                continue

            document = QADocument(
                id=f"squad:{article_index}:{paragraph_index}",
                title=title,
                text=context,
            )
            documents.append(document)

            raw_qas = paragraph.get("qas")
            if not isinstance(raw_qas, list):
                continue

            for qa in raw_qas:
                if not isinstance(qa, dict):
                    continue
                if qa.get("is_impossible") is True:
                    continue

                question = _clean_text(qa.get("question"))
                if not question:
                    continue

                raw_answers = qa.get("answers")
                if not isinstance(raw_answers, list):
                    continue

                answers = _dedupe_preserve_order(
                    _clean_text(answer.get("text"))
                    for answer in raw_answers
                    if isinstance(answer, dict)
                )
                if not answers:
                    continue

                examples.append(
                    QAExample(
                        question=question,
                        answers=answers,
                        documents=[document],
                    )
                )

    return QABundle(examples=examples, distractor_documents=documents)


def _parse_hotpotqa_bundle(payload: object) -> QABundle:
    if not isinstance(payload, list):
        raise ValueError("Invalid HotpotQA payload: expected a list.")

    examples: list[QAExample] = []
    documents: list[QADocument] = []

    for example_index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue

        question = _clean_text(item.get("question"))
        answer = _clean_text(item.get("answer"))
        raw_context = item.get("context")
        if not question or not answer or not isinstance(raw_context, list):
            continue

        example_documents: list[QADocument] = []
        for document_index, raw_document in enumerate(raw_context):
            if (
                not isinstance(raw_document, list)
                or len(raw_document) != 2
                or not isinstance(raw_document[0], str)
                or not isinstance(raw_document[1], list)
            ):
                continue

            title = raw_document[0].strip() or f"Hotpot Document {document_index + 1}"
            sentences = [
                sentence.strip()
                for sentence in raw_document[1]
                if isinstance(sentence, str) and sentence.strip()
            ]
            if not sentences:
                continue

            document = QADocument(
                id=f"hotpot:{example_index}:{document_index}",
                title=title,
                text=" ".join(sentences),
            )
            example_documents.append(document)
            documents.append(document)

        if not example_documents:
            continue

        examples.append(
            QAExample(
                question=question,
                answers=[answer],
                documents=example_documents,
            )
        )

    return QABundle(examples=examples, distractor_documents=documents)


def _parse_hotpotqa_parquet(path: Path) -> QABundle:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("HotpotQA RULER tasks require pyarrow.") from exc

    table = parquet.read_table(
        path,
        columns=["id", "question", "answer", "context"],
    )
    examples: list[QAExample] = []
    documents: list[QADocument] = []

    for example_index, item in enumerate(table.to_pylist()):
        question = _clean_text(item.get("question"))
        answer = _clean_text(item.get("answer"))
        context = item.get("context")
        if not question or not answer or not isinstance(context, dict):
            continue

        titles = context.get("title")
        sentence_groups = context.get("sentences")
        if not isinstance(titles, list) or not isinstance(sentence_groups, list):
            continue

        example_documents: list[QADocument] = []
        for document_index, (raw_title, raw_sentences) in enumerate(
            zip(titles, sentence_groups, strict=False)
        ):
            title = _clean_text(raw_title) or f"Hotpot Document {document_index + 1}"
            if not isinstance(raw_sentences, list):
                continue
            sentences = [
                sentence.strip()
                for sentence in raw_sentences
                if isinstance(sentence, str) and sentence.strip()
            ]
            if not sentences:
                continue

            document = QADocument(
                id=f"hotpot:{example_index}:{document_index}",
                title=title,
                text=" ".join(sentences),
            )
            example_documents.append(document)
            documents.append(document)

        if example_documents:
            examples.append(
                QAExample(
                    question=question,
                    answers=[answer],
                    documents=example_documents,
                )
            )

    return QABundle(examples=examples, distractor_documents=documents)


def _clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _dedupe_preserve_order(values: object) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        output.append(cleaned)
    return output
