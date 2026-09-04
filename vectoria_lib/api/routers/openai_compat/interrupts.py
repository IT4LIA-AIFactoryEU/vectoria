#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
"""Human-in-the-loop interrupt handling for the OpenAI-compatible chat endpoint.

HumanInTheLoopMiddleware (see agent_builder.py) pauses the LangGraph run via
langgraph's interrupt() before executing configured tools (e.g. delete_document).
The paused state is persisted in the checkpointer keyed by thread_id.

Since the OpenAI-compatible API is a plain request/response chat protocol (no
notion of "resume this run"), the approval round-trip is folded into the normal
conversation: the pending confirmation is sent as an assistant message, and the
user's next chat message ("yes"/"no") is parsed as the decision and used to
resume the graph via Command(resume=...) instead of injecting it as a new
human message.
"""

from typing import Optional

from langgraph.types import Command, Interrupt

_APPROVE_WORDS = {"si", "sì", "s", "yes", "y", "conferma", "confermo", "procedi"}
_REJECT_WORDS = {"no", "n", "annulla", "annullo", "rifiuta", "rifiuto", "cancel", "stop", "ferma"}


async def get_pending_interrupt(graph, config: dict) -> Optional[Interrupt]:
  """Return the pending Interrupt for this thread, if the graph is paused
  waiting for human approval (HumanInTheLoopMiddleware). None otherwise."""
  state = await graph.aget_state(config)
  return state.interrupts[0] if state.interrupts else None


def parse_decision(text: str) -> Optional[str]:
  """Parse a free-text user reply into 'approve' | 'reject' | None (unrecognized)."""
  normalized = text.strip().lower().strip(".!? ")
  if normalized in _APPROVE_WORDS:
    return "approve"
  if normalized in _REJECT_WORDS:
    return "reject"
  return None


def build_resume_command(interrupt_obj: Interrupt, decision: str) -> Command:
  """Build the Command(resume=...) payload resolving every pending action
  request in the interrupt with the same approve/reject decision."""
  n_actions = len(interrupt_obj.value.get("action_requests", []))
  if decision == "approve":
    decisions = [{"type": "approve"} for _ in range(n_actions)]
  else:
    decisions = [{"type": "reject", "message": "L'utente ha annullato l'azione."} for _ in range(n_actions)]
  return Command(resume={"decisions": decisions})


def format_confirmation_prompt(interrupt_obj: Interrupt) -> str:
  """Build a user-facing message describing the pending action(s) and asking
  for confirmation, from a HumanInTheLoopMiddleware HITLRequest."""
  action_requests = interrupt_obj.value.get("action_requests", [])
  lines = ["⚠️ **Conferma richiesta prima di procedere:**\n"]
  for action in action_requests:
    description = action.get("description")
    if description:
      lines.append(f"- {description}")
    else:
      lines.append(f"- {action.get('name')}({action.get('args')})")
  lines.append("\nRispondi **sì** per confermare o **no** per annullare.")
  return "\n".join(lines)
