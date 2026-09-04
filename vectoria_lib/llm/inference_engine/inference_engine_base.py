#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#

import logging
from abc import ABC, abstractmethod
from copy import deepcopy

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import BaseLanguageModel


class InferenceEngineBase(ABC):
    def __init__(self, args: dict):
        self.args = args
        self.name = args.pop("name")
        self.logger = logging.getLogger("llm")

    def update_args(self, new_args: dict):
        """Return a copy of this engine with updated args, leaving the cached original unchanged."""
        clone = deepcopy(self)
        for key, value in new_args.items():
            if key in clone.args and clone.args[key] != value:
                clone.logger.info("Updating %s with %s", key, value)
                clone.args[key] = value
        return clone

    @abstractmethod
    def as_langchain_completion_model(self) -> BaseLanguageModel:
        pass

    @abstractmethod
    def as_langchain_chat_model(self) -> BaseLanguageModel:
        pass

    @abstractmethod
    def as_langchain_embeddings_model(self) -> Embeddings:
        pass

    def __repr__(self):
        return f"InferenceEngineBase for {self.name}"
