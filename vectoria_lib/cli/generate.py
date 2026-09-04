#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import logging
from pathlib import Path

from tqdm import tqdm

from vectoria_lib.core.io.file_io import get_file_io
from vectoria_lib.llm.agents.agent_builder import AgentBuilder
from vectoria_lib.rag.vector_store.discovery import find_index_path


def generate(**kwargs):
    logger = logging.getLogger("tasks")

    test_set_path = Path(kwargs["test_set_path"])
    tenant_path = Path(kwargs["index_path"])
    output_dir = Path(kwargs["output_dir"])

    index_path = find_index_path(tenant_path)
    if index_path is None:
        raise ValueError(f"Nessun indice trovato in: {tenant_path}")

    agent = AgentBuilder.build_qa_agent(index_path=index_path)

    data = get_file_io("json").read(test_set_path)
    questions = data["question"]
    ground_truths = data.get("ground_truth", [None] * len(questions))

    answers = []
    contexts = []
    for q in tqdm(questions, desc="Generating answers"):
        result = agent.ask(q)
        answers.append(result["answer"])
        contexts.append([d.page_content for d in result["retrieved_docs"]])

    answers_file = get_file_io("json").write(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        },
        output_root_path=output_dir,
        name=f"{test_set_path.stem}_answers",
        add_time_stamp=False,
    )
    logger.info("Answers saved to %s", answers_file)
