import threading
import time

from dfm_evals.vllm_patches import (
    apply_instance_method_rlock_patch,
)


class FakeFastTokenizer:
    def __init__(self, backend=None) -> None:
        self._tokenizer = backend or self
        self._borrow_lock = getattr(self._tokenizer, "_borrow_lock", threading.Lock())
        self.call_calls = 0
        self.encode_calls = 0
        self.encode_plus_calls = 0
        self.batch_encode_plus_calls = 0
        self.set_truncation_and_padding_calls = 0
        self.decode_calls = 0
        self.batch_decode_calls = 0

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        return self._borrow("call_calls", {"input_ids": [len(text)]})

    def encode(self, text: str, **_: object) -> list[int]:
        return self._borrow("encode_calls", [len(text)])

    def encode_plus(self, text: str, **_: object) -> dict[str, list[int]]:
        return self._borrow("encode_plus_calls", {"input_ids": [len(text)]})

    def batch_encode_plus(
        self,
        texts: list[str],
        **_: object,
    ) -> dict[str, list[list[int]]]:
        return self._borrow(
            "batch_encode_plus_calls",
            {"input_ids": [[len(text)] for text in texts]},
        )

    def _batch_encode_plus(
        self,
        texts: list[str],
        **_: object,
    ) -> dict[str, list[list[int]]]:
        return self.batch_encode_plus(texts, **_)

    def set_truncation_and_padding(self, **_: object) -> None:
        self._borrow("set_truncation_and_padding_calls", None)

    def decode(self, token_ids: list[int]) -> str:
        return self._borrow("decode_calls", str(token_ids[0]))

    def batch_decode(self, token_ids: list[list[int]]) -> list[str]:
        return self._borrow("batch_decode_calls", [str(ids[0]) for ids in token_ids])

    def _borrow(self, counter_name: str, result):
        if not self._borrow_lock.acquire(blocking=False):
            raise RuntimeError("Already borrowed")
        try:
            setattr(self, counter_name, getattr(self, counter_name) + 1)
            time.sleep(0.01)
            return result
        finally:
            self._borrow_lock.release()


class FakeFastTokenizerBackend:
    def __init__(self) -> None:
        self._borrow_lock = threading.Lock()


def test_instance_method_rlock_patch_serializes_fast_tokenizer_calls() -> None:
    apply_instance_method_rlock_patch(
        FakeFastTokenizer,
        (
            "__call__",
            "encode",
            "encode_plus",
            "batch_encode_plus",
            "_batch_encode_plus",
            "set_truncation_and_padding",
            "decode",
            "batch_decode",
        ),
    )

    shared_backend = FakeFastTokenizerBackend()
    tokenizers = [FakeFastTokenizer(shared_backend) for _ in range(2)]
    barrier = threading.Barrier(7)
    errors = []
    results = []

    def run(worker) -> None:
        try:
            barrier.wait()
            results.append(worker())
        except Exception as exc:
            errors.append(exc)

    workers = [
        lambda: tokenizers[0]("hello", truncation=True, max_length=4),
        lambda: tokenizers[0].encode("<tool_call>", add_special_tokens=False),
        lambda: tokenizers[0].encode_plus("question", truncation=True, max_length=8),
        lambda: tokenizers[0].batch_encode_plus(["a", "bb"], truncation=True, max_length=4),
        lambda: tokenizers[1].set_truncation_and_padding(max_length=4),
        lambda: tokenizers[1].decode([7]),
        lambda: tokenizers[1].batch_decode([[1], [2]]),
    ]
    threads = [threading.Thread(target=run, args=(worker,)) for worker in workers]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(results) == len(workers)
    assert tokenizers[0].call_calls == 1
    assert tokenizers[0].encode_calls == 1
    assert tokenizers[0].encode_plus_calls == 1
    assert tokenizers[0].batch_encode_plus_calls == 1
    assert tokenizers[1].set_truncation_and_padding_calls == 1
    assert tokenizers[1].decode_calls == 1
    assert tokenizers[1].batch_decode_calls == 1
