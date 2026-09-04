#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
# ----------------------------------------------------------------------------------------------
import datetime
import json
import logging
import time
import uuid
from pathlib import Path

import numpy as np
from langchain_core.messages import AIMessage


class QAAgent:
    """
    A Question Answering (QA) Agent backed by a LangGraph ReAct graph.
    """

    def __init__(self, *, graph, retriever_tool=None):
        self.logger = logging.getLogger("llm")
        self.graph = graph
        self.retriever_tool = retriever_tool

    # --------------------------------------------------------------------------------
    def ask(self, question: str, thread_id: str = None) -> dict:
        """
        Ask the QAAgent a question and return a result dict:
        {"answer", "retrieved_docs", "messages"}

        Parameters:
        - question   : the user question
        - thread_id  : conversation thread; a new UUID is used if omitted
        """
        return self._ask_graph(question, thread_id)

    def _ask_graph(self, question: str, thread_id: str = None) -> dict:
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        result = self.graph.invoke(
            {"messages": [{"role": "human", "content": question}]},
            config={"configurable": {"thread_id": thread_id}},
        )

        # Extract the last plain AIMessage as the final answer
        answer = ""
        for m in reversed(result["messages"]):
            if isinstance(m, AIMessage) and not m.tool_calls:
                answer = m.content
                break

        retrieved_docs = self.retriever_tool.last_docs if self.retriever_tool else []

        self.logger.debug(
            "\n------%s------- > Question: %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), question
        )
        self.logger.info(
            "\n------%s------- > Answer: %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), answer
        )

        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "messages": result["messages"],
        }

    def _get_correct_context_key(self):
        return "retrieved_docs"

    def _get_questions(self, kwargs):
        if kwargs.get("questions"):
            # Handle questions from CLI
            data = {"question": kwargs.get("questions")}
            self.logger.info("Received questions directly via CLI: %s", kwargs.get("questions"))
        else:
            # Handle questions from the test set JSON
            test_set_path = kwargs.get("test_set_path")
            if not test_set_path:
                raise ValueError("No questions provided. Use '--questions' or '--test-set-path'.")
            with open(test_set_path, encoding="utf-8") as file:
                data = json.load(file)
            self.logger.info("Questions are loaded from test set JSON: %s", test_set_path)
        return data

    def inference(self, kwargs):

        data = self._get_questions(kwargs)
        output_dir = kwargs.get("output_dir")

        times = []
        results = data.copy()
        results["answer"] = []
        retrieved_contexts_key = self._get_correct_context_key()
        results[retrieved_contexts_key] = []

        for q in data["question"]:
            self.logger.info("Answering question: %s", q)

            start_time = time.perf_counter()
            try:
                result = self.ask(q)
            except Exception as e:
                self.logger.error("Error answering question: %s", e)
                return

            took = time.perf_counter() - start_time
            self.logger.info("Time taken to answer question: %.2f seconds", took)
            times.append(took)

            contexts = result[retrieved_contexts_key]
            answer = result["answer"]

            results[retrieved_contexts_key].append(
                [{"page_content": c.page_content, "metadata": c.metadata} for c in contexts]
            )
            results["answer"].append(answer)

        results["times"] = dict(mean=np.mean(times), std=np.std(times))
        self.logger.info(
            "Mean time and std taken to answer questions: %.2f seconds, %.2f seconds",
            results["times"]["mean"],
            results["times"]["std"],
        )
        self._write_inference_results(
            results, output_dir, f"inference_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )

    def _write_inference_results(self, results, output_dir, output_name):
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        output_file = Path(output_dir) / f"{output_name}.json"
        start_time = time.perf_counter()
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

        self.logger.info(
            "Annotated test set saved to %s and took %.2f seconds", output_file, time.perf_counter() - start_time
        )
