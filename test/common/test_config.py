import argparse
from pathlib import Path

import pytest

# ── Existing behaviour (backward compat) ──────────────────────────────────────


def test_get_default_config(config):
    assert config.get("retriever", "top_k") == 5
    assert config.get("inference_engine", "model_name") == "HuggingFaceTB/SmolLM-135M"


def test_load_custom_config(config, write_config_file_for_tests):
    # Reload the same full valid config with a different log_level to verify
    # that load_config() re-reads from disk and validates.
    custom_path = "/tmp/test_custom_log.yaml"
    with open(write_config_file_for_tests) as f:
        base = f.read()
    with open(custom_path, "w") as f:
        f.write(base.replace("log_level: DEBUG", "log_level: ERROR"))
    config.load_config(custom_path)
    assert config.get("log_level") == "ERROR"


def test_get_not_valid_config(config):
    with pytest.raises(KeyError):
        config.get("top_k")
    with pytest.raises(KeyError):
        config.get("not_valid_key")


def test_set_config(config):
    config.set("retriever", "top_k", 10)
    assert config.get("retriever", "top_k") == 10


@pytest.mark.skip(reason="This feature is temporarily disabled")
def test_override_config(config):
    config.update_from_args(argparse.Namespace(top_k=10))
    assert config.get("retriever", "top_k") == 10
    config.set("log_level", "INFO")
    assert config.get("log_level") == "INFO"


# ── Pydantic validation ───────────────────────────────────────────────────────


def test_invalid_log_level_rejected(config, write_config_file_for_tests):
    bad_path = "/tmp/test_bad_log_level.yaml"
    with open(write_config_file_for_tests) as f:
        base = f.read()
    with open(bad_path, "w") as f:
        f.write(base.replace("log_level: DEBUG", "log_level: VERBOSE"))
    with pytest.raises(ValueError, match="Invalid configuration"):
        config.load_config(bad_path)


def test_invalid_system_prompts_lang_rejected(config, write_config_file_for_tests):
    bad_path = "/tmp/test_bad_lang.yaml"
    with open(write_config_file_for_tests) as f:
        base = f.read()
    with open(bad_path, "w") as f:
        f.write(base.replace("system_prompts_lang: eng", "system_prompts_lang: fr"))
    with pytest.raises(ValueError, match="Invalid configuration"):
        config.load_config(bad_path)


def test_config_types_after_load(config):
    """Verify that Pydantic coercion gives the expected Python types."""
    assert isinstance(config.get("vectoria_logs_dir"), Path)
    assert isinstance(config.get("retriever", "enabled"), bool)
    assert isinstance(config.get("retriever", "top_k"), int)
    assert isinstance(config.get("retriever", "lambda_mult"), float)
    assert isinstance(config.get("vector_store"), dict)
    assert isinstance(config.get("inference_engine"), dict)
