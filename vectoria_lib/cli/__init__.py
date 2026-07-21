from vectoria_lib.core.config import Config
from vectoria_lib.core.logger import setup_logger

logger = setup_logger("tasks", Config().get("log_level"), Config().get("vectoria_logs_dir") / "tasks.log")

logger.debug("Module tasks initialized")
