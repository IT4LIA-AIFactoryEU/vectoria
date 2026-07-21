from vectoria_lib.api.routers.openai_compat.interrupts import parse_decision


def test_parse_decision_accepts_deliberate_confirm_words():
    for word in ["si", "sì", "s", "yes", "y", "conferma", "confermo", "procedi"]:
        assert parse_decision(word) == "approve", f"{word!r} should approve"


def test_parse_decision_no_longer_treats_ok_or_vai_as_approve():
    """ok/vai are too generic — a common one-word reply to something entirely
    unrelated must not silently approve a stale pending action."""
    assert parse_decision("ok") is None
    assert parse_decision("vai") is None


def test_parse_decision_accepts_reject_words():
    for word in ["no", "n", "annulla", "annullo", "rifiuta", "rifiuto", "cancel", "stop", "ferma"]:
        assert parse_decision(word) == "reject", f"{word!r} should reject"


def test_parse_decision_returns_none_for_unrelated_text():
    assert parse_decision("che tempo fa domani?") is None


def test_build_resume_command_reject_has_no_approve_leakage():
    """Sanity check on the primitive the router's implicit-reject path relies
    on: rejecting must never produce an 'approve' decision."""
    from langgraph.types import Interrupt

    from vectoria_lib.api.routers.openai_compat.interrupts import build_resume_command

    interrupt_obj = Interrupt(
        value={"action_requests": [{"name": "delete_document", "args": {"filename": "x.pdf"}}]},
        id="test-id",
    )
    command = build_resume_command(interrupt_obj, "reject")
    assert command.resume["decisions"] == [
        {"type": "reject", "message": "L'utente ha annullato l'azione."}
    ]
