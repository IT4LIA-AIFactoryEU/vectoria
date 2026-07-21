import logging
from abc import ABC, abstractmethod

from langchain_core.documents import Document


class PostRetrievalStepBase(ABC):
    def __init__(self):
        self.logger = logging.getLogger("rag")

    @abstractmethod
    def post_process(self, chunks: list[Document]) -> list[Document]:
        pass

    def as_langchain_post_retrieval_step(self):
        return self.post_process
