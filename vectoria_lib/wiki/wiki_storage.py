#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import json
import logging
from pathlib import Path

logger = logging.getLogger("wiki")

_PENDING_FILE = "pending_updates.json"
_SOURCES_FILE = "page_sources.json"
_PAGES_DIR = "pages"


class WikiStorage:
  def __init__(self, wiki_dir: Path):
    self.wiki_dir = Path(wiki_dir)
    self.pages_dir = self.wiki_dir / _PAGES_DIR
    self.pending_file = self.wiki_dir / _PENDING_FILE
    self.wiki_dir.mkdir(parents=True, exist_ok=True)
    self.pages_dir.mkdir(exist_ok=True)

  def load_pending(self) -> list[dict]:
    if not self.pending_file.exists():
      return []
    try:
      return json.loads(self.pending_file.read_text(encoding="utf-8"))
    except Exception:
      return []

  def save_pending(self, pending: list[dict]):
    self.pending_file.write_text(
      json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8"
    )

  def load_sources(self) -> dict:
    p = self.wiki_dir / _SOURCES_FILE
    if not p.exists():
      return {}
    try:
      return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
      return {}

  def save_sources(self, sources: dict):
    (self.wiki_dir / _SOURCES_FILE).write_text(
      json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8"
    )

  def list_pages(self) -> list[str]:
    if not self.pages_dir.exists():
      return []
    return sorted(p.stem for p in self.pages_dir.glob("*.md"))

  def get_page(self, name: str) -> str | None:
    path = self.page_path(name)
    return path.read_text(encoding="utf-8") if path.exists() else None

  def write_page(self, name: str, content: str):
    self.page_path(name).write_text(content, encoding="utf-8")

  def delete_page(self, name: str):
    self.page_path(name).unlink(missing_ok=True)

  def search_pages(self, query: str) -> str:
    query_lower = query.lower()
    results = []
    for page_file in sorted(self.pages_dir.glob("*.md")):
      content = page_file.read_text(encoding="utf-8")
      if query_lower in content.lower():
        idx = content.lower().find(query_lower)
        start = max(0, idx - 40)
        excerpt = content[start : idx + 200].replace("\n", " ")
        results.append(f"**{page_file.stem}**: ...{excerpt}...")
    if not results:
      return f"No wiki pages match '{query}'."
    return f"Found in {len(results)} page(s):\n" + "\n".join(results[:5])

  def safe_stem(self, name: str) -> str:
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in name)[:60].strip()

  def page_path(self, name: str) -> Path:
    return self.pages_dir / f"{self.safe_stem(name)}.md"
