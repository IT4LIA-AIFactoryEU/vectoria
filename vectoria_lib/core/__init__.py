from vectoria_lib.core.config import Config
from vectoria_lib.core.logger import setup_logger

logger = setup_logger("common", Config().get("log_level"), Config().get("vectoria_logs_dir") / "common.log")

logger.debug("Module common initialized")
