import asyncio

from vectoria_lib.api.routers.openai_compat.router import _locked_stream


def test_locked_stream_holds_lock_until_generator_exhausted():
    """The lock must stay held for the whole SSE stream, not just until the
    StreamingResponse object is constructed — otherwise a second request on
    the same thread could interleave with an in-flight stream."""

    async def main():
        lock = asyncio.Lock()
        await lock.acquire()

        async def inner():
            assert lock.locked()
            yield "a"
            assert lock.locked()
            yield "b"

        chunks = []
        async for chunk in _locked_stream(lock, inner()):
            chunks.append(chunk)
        return chunks, lock.locked()

    chunks, locked_after = asyncio.run(main())
    assert chunks == ["a", "b"]
    assert locked_after is False


def test_locked_stream_releases_lock_even_if_inner_raises():
    async def main():
        lock = asyncio.Lock()
        await lock.acquire()

        async def inner():
            yield "a"
            raise RuntimeError("boom")

        collected = []
        raised = False
        try:
            async for chunk in _locked_stream(lock, inner()):
                collected.append(chunk)
        except RuntimeError:
            raised = True
        return collected, raised, lock.locked()

    collected, raised, locked_after = asyncio.run(main())
    assert collected == ["a"]
    assert raised is True
    assert locked_after is False
