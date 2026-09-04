#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import asyncio

from vectoria_lib.api.tenant_registry import TenantRegistry


def test_get_thread_lock_returns_same_lock_for_same_key():
    registry = TenantRegistry()
    lock1 = registry.get_thread_lock("tenant-a", "thread-1")
    lock2 = registry.get_thread_lock("tenant-a", "thread-1")
    assert lock1 is lock2


def test_get_thread_lock_returns_different_locks_for_different_threads():
    registry = TenantRegistry()
    lock1 = registry.get_thread_lock("tenant-a", "thread-1")
    lock2 = registry.get_thread_lock("tenant-a", "thread-2")
    assert lock1 is not lock2


def test_get_thread_lock_returns_different_locks_for_different_tenants_same_thread_id():
    """Two tenants must never share a lock even if a thread_id string collides."""
    registry = TenantRegistry()
    lock1 = registry.get_thread_lock("tenant-a", "thread-1")
    lock2 = registry.get_thread_lock("tenant-b", "thread-1")
    assert lock1 is not lock2


def test_concurrent_get_thread_lock_calls_converge_on_one_lock():
    """Regression test for the race this lock exists to prevent: many
    coroutines racing to create-or-fetch the lock for the same key must all
    end up holding the SAME lock object, not each creating their own."""

    async def main():
        registry = TenantRegistry()

        async def fetch():
            return registry.get_thread_lock("tenant-a", "thread-1")

        return await asyncio.gather(*[fetch() for _ in range(50)])

    results = asyncio.run(main())
    assert len(set(id(lock) for lock in results)) == 1
