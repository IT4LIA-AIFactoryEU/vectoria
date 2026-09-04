#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
from vectoria_lib.api.routers.openai_compat.models import ChatMessage
from vectoria_lib.api.routers.openai_compat.router import _derive_thread_id


def test_derive_thread_id_is_stable_as_conversation_grows():
    """The regression this guards against: thread_id must stay the same as a
    conversation grows turn over turn (same first message, more messages
    appended later), otherwise a header-less client's checkpoint is never
    found again after turn 1 — total memory loss every turn."""
    turn_1 = [ChatMessage(role="user", content="Ciao")]
    turn_2 = [
        ChatMessage(role="user", content="Ciao"),
        ChatMessage(role="assistant", content="Come posso aiutarti?"),
        ChatMessage(role="user", content="Che tempo fa?"),
    ]
    assert _derive_thread_id("tenant-a", turn_1) == _derive_thread_id("tenant-a", turn_2)


def test_derive_thread_id_is_deterministic_for_identical_history():
    msgs = [
        ChatMessage(role="user", content="Ciao"),
        ChatMessage(role="assistant", content="Come posso aiutarti?"),
        ChatMessage(role="user", content="Che tempo fa?"),
    ]
    assert _derive_thread_id("tenant-a", msgs) == _derive_thread_id("tenant-a", list(msgs))


def test_derive_thread_id_works_for_single_message_conversation():
    """A brand new conversation's very first turn only has one message —
    'all messages except the last' would be empty, so this must fall back to
    hashing that single message instead of crashing or hashing nothing."""
    msgs = [ChatMessage(role="user", content="Ciao")]
    result = _derive_thread_id("tenant-a", msgs)
    assert isinstance(result, str)
    assert len(result) == 32


def test_derive_thread_id_differs_by_tenant():
    msgs = [ChatMessage(role="user", content="Ciao")]
    assert _derive_thread_id("tenant-a", msgs) != _derive_thread_id("tenant-b", msgs)
