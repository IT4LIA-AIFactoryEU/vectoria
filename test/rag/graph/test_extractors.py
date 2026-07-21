import builtins
import logging

from vectoria_lib.rag.graph import extractors


def test_load_spacy_warns_when_not_installed(monkeypatch, caplog):
    extractors._SPACY_CACHE.clear()
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "spacy":
            raise ImportError("no module named spacy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with caplog.at_level(logging.WARNING, logger="graph"):
        result = extractors.load_spacy("it_core_news_sm")

    assert result is None
    assert any("spaCy not installed" in r.message for r in caplog.records)
    assert any(r.levelno == logging.WARNING for r in caplog.records)
