from vectoria_lib.core.config import Config

# Order matters: these two tests together prove the autouse `_reset_config_state`
# fixture in conftest.py prevents Config.set() mutations from leaking across tests.


def test_a_mutates_log_level():
    original = Config().get("log_level")
    mutated = "MUTATED_" + str(original)
    Config().set("log_level", value=mutated)
    assert Config().get("log_level") == mutated


def test_b_sees_original_log_level_not_the_previous_tests_mutation():
    assert not str(Config().get("log_level")).startswith("MUTATED_")
