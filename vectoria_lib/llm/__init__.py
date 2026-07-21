from vectoria_lib.core.config import Config
from vectoria_lib.core.logger import setup_logger

logger = setup_logger("llm", Config().get("log_level"), Config().get("vectoria_logs_dir") / "llm.log")

logger.debug("Module llm initialized")
