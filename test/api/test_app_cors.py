import pytest

from vectoria_lib.api.app import create_app


def test_wildcard_origin_with_credentials_is_rejected(monkeypatch):
    monkeypatch.setenv("VECTORIA_ALLOWED_ORIGINS", "*")
    with pytest.raises(RuntimeError, match="cannot contain '\\*'"):
        create_app()


def test_wildcard_among_explicit_origins_is_also_rejected(monkeypatch):
    monkeypatch.setenv("VECTORIA_ALLOWED_ORIGINS", "https://example.com,*")
    with pytest.raises(RuntimeError, match="cannot contain '\\*'"):
        create_app()


def test_explicit_origins_are_accepted(monkeypatch):
    monkeypatch.setenv("VECTORIA_ALLOWED_ORIGINS", "https://example.com")
    app = create_app()
    assert app is not None


def test_no_origins_set_is_accepted(monkeypatch):
    monkeypatch.delenv("VECTORIA_ALLOWED_ORIGINS", raising=False)
    app = create_app()
    assert app is not None
