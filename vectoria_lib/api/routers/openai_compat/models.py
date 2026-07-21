from typing import Optional
from pydantic import BaseModel


class ChatMessage(BaseModel):
  role: str
  content: str | list

  def text(self) -> str:
    if isinstance(self.content, str):
      return self.content
    return " ".join(
      p.get("text", "") for p in self.content if isinstance(p, dict) and p.get("type") == "text"
    ).strip()


class ChatCompletionRequest(BaseModel):
  model: Optional[str] = "vectoria"
  messages: list[ChatMessage]
  stream: Optional[bool] = False
  chat_id: Optional[str] = None
