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
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from vectoria_lib.core.constants import ALLOWED_EXTENSIONS

logger = logging.getLogger("llm")


class _DeleteInput(BaseModel):
    filename: str


class ListDocumentsTool(BaseTool):
    name: str = "list_documents"
    description: str = (
        "List all documents currently indexed and available for search. "
        "Call this tool when the user asks which files or documents are loaded, "
        "available, or can be searched."
    )
    upload_dir: Any  # Path
    upload_url: str = "http://localhost:8080/upload"

    def _run(self, query: str = "") -> str:
        upload_dir = Path(self.upload_dir)
        _no_docs_msg = f"Nessun documento caricato. Per caricare file PDF o DOCX vai su: {self.upload_url}"
        if not upload_dir.exists():
            return _no_docs_msg
        files = sorted(f.name for f in upload_dir.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS)
        if not files:
            return _no_docs_msg

        # Load summaries if available
        meta = {}
        meta_path = upload_dir / "documents_meta.json"
        if meta_path.exists():
            try:
                with meta_path.open(encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass

        lines = []
        for filename in files:
            summary = meta.get(filename, "")
            if summary:
                lines.append(f"- **{filename}**: {summary}")
            else:
                lines.append(f"- **{filename}**")

        return f"Documenti disponibili ({len(files)}):\n" + "\n".join(lines)


class DeleteDocumentTool(BaseTool):
    name: str = "delete_document"
    description: str = (
        "Delete a specific document by filename. "
        "Call this tool when the user explicitly asks to remove or delete a document. "
        "The user must provide the exact filename (e.g. 'report.pdf'). "
        "After deletion the index is automatically rebuilt with the remaining documents."
    )
    args_schema: Any = _DeleteInput
    upload_dir: Any  # Path
    index_dir: Any  # Path
    rebuild_fn: Any  # Callable[[list[Path]], None]
    upload_url: str = "http://localhost:8080/upload"

    def _run(self, filename: str) -> str:
        if not filename or not filename.strip():
            return (
                "Filename non specificato. "
                "Fornisci il nome esatto del file da eliminare (es. 'report.pdf'). "
                "Usa il tool list_documents per vedere i file disponibili."
            )

        upload_dir = Path(self.upload_dir)
        clean = filename.strip()
        if Path(clean).name != clean:
            return f"Filename non valido: '{filename}'. Il nome del file non può contenere separatori di percorso."
        target = upload_dir / clean

        if target.is_dir():
            return f"'{clean}' è una directory, non un file. Specifica il nome esatto del file da eliminare."

        if not target.exists():
            available = sorted(
                f.name for f in upload_dir.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
            )
            hint = ", ".join(available) if available else "nessuno"
            return (
                f"File '{clean}' non trovato. "
                f"Documenti disponibili: {hint}. "
                "Usa il tool list_documents per vedere l'elenco completo."
            )

        try:
            target.unlink()
        except FileNotFoundError:
            return f"File '{clean}' non trovato al momento dell'eliminazione (potrebbe essere già stato rimosso)."
        logger.info("Deleted document: %s", target)

        remaining = [f for f in upload_dir.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS]

        if remaining:
            try:
                self.rebuild_fn(clean, remaining)
                return f"File '{clean}' eliminato. Indice aggiornato con {len(remaining)} documento/i rimanente/i."
            except Exception as exc:
                logger.error("Rebuild after deletion failed: %s", exc)
                return (
                    f"File '{clean}' eliminato, ma la reindicizzazione ha fallito: {exc}. "
                    "Ricarica un documento per ripristinare l'indice."
                )
        else:
            # No files left — clear the index
            try:
                self.rebuild_fn(clean, [])
            except Exception as exc:
                logger.error("Index cleanup after deleting last document failed: %s", exc)
                return (
                    f"File '{clean}' eliminato, ma la pulizia dell'indice ha fallito: {exc}. "
                    "L'indice potrebbe essere in uno stato inconsistente."
                )
            return (
                f"File '{clean}' eliminato. "
                "Non ci sono altri documenti: il tenant è stato rimosso. "
                f"Per ricominciare carica nuovi documenti su {self.upload_url}"
            )
