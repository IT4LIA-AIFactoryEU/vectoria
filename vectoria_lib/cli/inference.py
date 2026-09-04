#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#

from vectoria_lib.llm.agents.agent_builder import AgentBuilder


def inference(**kwargs: dict):
    qa_agent = AgentBuilder.build_qa_agent(index_path=kwargs["index_path"])
    qa_agent.inference(kwargs)
