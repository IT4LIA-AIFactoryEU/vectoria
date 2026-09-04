#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
from vectoria_lib.core.config import Config
from vectoria_lib.core.logger import setup_logger

logger = setup_logger("rag", Config().get("log_level"), Config().get("vectoria_logs_dir") / "rag.log")
logger.debug("Module rag initialized")
