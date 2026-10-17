import asyncio
from types import SimpleNamespace

import pytest
from inspect_ai.scorer import Target
from inspect_ai.util import ExecResult
from dfm_evals.tasks import code


@pytest.mark.asyncio
async def test_verification_is_bounded_without_changing_timeout(monkeypatch):
    active = peak = calls = 0

    class Sandbox:
        async def exec(self, *, cmd, timeout):
            nonlocal active, peak, calls
            assert timeout == code.VERIFY_TIMEOUT
            assert cmd[:2] == ['python', '-c']
            active += 1
            peak = max(peak, active)
            calls += 1
            await asyncio.sleep(0.01)
            active -= 1
            return ExecResult(True, 0, '', '')

    monkeypatch.setattr(code, 'sandbox', Sandbox)
    state = SimpleNamespace(output=SimpleNamespace(completion='def f():\n    return 1'),
                            metadata={'prompt': '', 'test': 'def check(f): assert f() == 1',
                                      'entry_point': 'f'})
    verify = code.verify_sanitized()
    results = await asyncio.gather(*(verify(state, Target('')) for _ in range(20)))
    assert peak == 2 and calls == 20
    assert all(r.value == 'C' for r in results)


def test_invalid_concurrency_rejected():
    with pytest.raises(ValueError):
        code.verify_sanitized(0)
