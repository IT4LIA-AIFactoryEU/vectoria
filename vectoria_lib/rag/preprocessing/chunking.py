#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli, Mattia Buzzoni
#
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vectoria_lib.core.config import Config
from vectoria_lib.rag.metadata_keys import DOC_FILE_NAME, PARAGRAPH_NUMBER, SEQ_ID

logger = logging.getLogger("rag")


@lru_cache(maxsize=4)
def _load_tokenizer(tokenizer_name: str):
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer: %s", tokenizer_name)
    return AutoTokenizer.from_pretrained(tokenizer_name)


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str, device: str):
    """Load an embedding model for semantic chunking.

    Root cause: transformers 4.57 + accelerate forces init_empty_weights regardless
    of low_cpu_mem_usage, leaving BOTH parameters and non-persistent buffers on the
    meta device after from_pretrained.  Any .to(device) call then fails.

    Fix: bypass from_pretrained entirely.
      1. AutoModel.from_config() creates the architecture with real CPU tensors
         (no from_pretrained, no accelerate, no meta device).
      2. Load the checkpoint weights directly from the HuggingFace cache using
         safetensors/torch.load, bypassing the accelerate loading pipeline.
      3. load_state_dict() + .to(device) — both safe because no meta tensors exist.
    """
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from transformers.utils import cached_file

    logger.info("Loading embedding model for semantic chunking: %s (device=%s)", model_name, device)
    _tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Step 1 — build architecture with real CPU tensors (no accelerate involved)
    _config = AutoConfig.from_pretrained(model_name)
    _model = AutoModel.from_config(_config)  # random weights, real CPU tensors

    # Step 2 — load checkpoint weights from cache (safetensors preferred, fallback to bin)
    _state_dict = None
    for _fname in ("model.safetensors", "pytorch_model.bin"):
        try:
            _ckpt = cached_file(model_name, _fname, local_files_only=True)
            if _fname.endswith(".safetensors"):
                from safetensors.torch import load_file as _st_load

                _state_dict = _st_load(_ckpt, device="cpu")
            else:
                _state_dict = torch.load(_ckpt, map_location="cpu", weights_only=True)
            logger.info("Loaded weights from cache: %s", _ckpt)
            break
        except Exception as _e:
            logger.debug("Could not load %s: %s", _fname, _e)

    if _state_dict is not None:
        _missing, _unexpected = _model.load_state_dict(_state_dict, strict=False)
        if _missing:
            logger.debug("Missing keys in state dict: %s", _missing[:5])
        if _unexpected:
            logger.debug("Unexpected keys in state dict: %s", _unexpected[:5])
    else:
        logger.warning("Checkpoint not found in cache for %s; semantic chunking uses random weights", model_name)

    # Step 3 — move to target device (safe: all tensors are real CPU tensors)
    _model = _model.to(device)
    _model.eval()

    class _MeanPoolEncoder:
        """Minimal encode() interface compatible with SentenceTransformer, using mean pooling."""

        def encode(
            self,
            sentences: list,
            normalize_embeddings: bool = True,
            show_progress_bar: bool = False,
            batch_size: int = 64,
        ) -> "np.ndarray":
            all_embeddings = []
            for start in range(0, len(sentences), batch_size):
                batch = sentences[start : start + batch_size]
                encoded = _tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    output = _model(**encoded)
                token_emb = output.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
                emb = torch.sum(token_emb * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
                if normalize_embeddings:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                all_embeddings.append(emb.cpu().numpy())
            return np.vstack(all_embeddings)

    return _MeanPoolEncoder()


@lru_cache(maxsize=4)
def _build_splitter(
    chunk_size: int,
    chunk_overlap: int,
    separators: tuple,
    is_separator_regex: tuple,
    tokenizer_name: str | None,
) -> RecursiveCharacterTextSplitter:
    splitter_kwargs = dict(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators),
        is_separator_regex=list(is_separator_regex),
    )
    if tokenizer_name:
        tokenizer = _load_tokenizer(tokenizer_name)
        logger.debug("Chunking with tokenizer %s (chunk_size=%d tokens)", tokenizer_name, chunk_size)
        return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, **splitter_kwargs)
    logger.debug("Chunking by character count (chunk_size=%d chars)", chunk_size)
    return RecursiveCharacterTextSplitter(length_function=len, **splitter_kwargs)


def recursive_character_text_splitter(
    doc: Document,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] = ["\n\n", "\n", " ", ""],
    is_separator_regex: list[bool] = [False, False, False, False],
    dump_chunks_on_file: bool = False,
    tokenizer_name: str = None,
) -> list[Document]:
    """
    Split the input text into chunks using RecursiveCharacterTextSplitter.

    When tokenizer_name is set, chunk_size and chunk_overlap are expressed in
    tokens (using the specified HuggingFace tokenizer). Otherwise they are
    expressed in characters.
    """
    text_splitter = _build_splitter(
        chunk_size,
        chunk_overlap,
        tuple(separators),
        tuple(is_separator_regex),
        tokenizer_name,
    )

    chunks = text_splitter.split_documents([doc])

    for seq_id, chunk in enumerate(chunks):
        chunk.metadata[SEQ_ID] = seq_id

    if dump_chunks_on_file:
        output_name = f"{doc.metadata[DOC_FILE_NAME]}_{doc.metadata[PARAGRAPH_NUMBER]}"
        _log_chunks_on_file(chunks, Path(Config().get("vectoria_logs_dir")) / "chunks" / f"{output_name}_chunks.txt")

    return chunks


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------


def semantic_text_splitter(
    doc: Document,
    embedding_model_name: str,
    breakpoint_threshold_type: Literal["percentile", "standard_deviation", "interquartile", "gradient"] = "percentile",
    breakpoint_threshold_amount: float = 70.0,
    device: str = "cpu",
    max_chunk_size: int | None = None,
    chunk_overlap: int = 0,
    tokenizer_name: str | None = None,
    dump_chunks_on_file: bool = False,
) -> list[Document]:
    """
    Split a document into semantically coherent chunks.

    Sentences are embedded using a SentenceTransformer model. A breakpoint is
    placed between two consecutive sentences when their cosine similarity drops
    below a threshold, indicating a topic change. Sentences between breakpoints
    are merged into a single chunk.

    Parameters
    ----------
    embedding_model_name : SentenceTransformer model used to embed sentences.
        Defaults to the vector_store model so that chunking and retrieval use
        the same embedding space.
    breakpoint_threshold_type : strategy for detecting breakpoints:
        - "percentile"         : split where similarity < (100 - amount)-th percentile.
                                 amount=70 → split the bottom 30% similarity gaps.
        - "standard_deviation" : split where similarity < mean - amount * std.
        - "interquartile"      : split where similarity < Q1 - amount * IQR.
        - "gradient"           : split where the *change* in similarity is in the
                                 top `amount` percentile (sudden topic shifts).
    breakpoint_threshold_amount : numeric threshold for the chosen strategy.
    device : device for the embedding model ("cpu", "cuda", "mps").
    max_chunk_size : if set, semantic chunks exceeding this size (tokens if
        tokenizer_name is given, otherwise words) are further split with the
        recursive splitter as a fallback.
    chunk_overlap : overlap used only when max_chunk_size fallback is applied.
    tokenizer_name : HuggingFace tokenizer for token-based size counting.
    dump_chunks_on_file : write chunks to a log file for inspection.
    """
    import numpy as np

    text = doc.page_content.strip()
    if not text:
        return []

    sentences = _split_into_sentences(text)
    if not sentences:
        return [doc]

    if len(sentences) == 1:
        chunk = Document(page_content=sentences[0], metadata={**doc.metadata, "seq_id": 0})
        return [chunk]

    # Embed all sentences at once (batched)
    model = _load_sentence_transformer(embedding_model_name, device)
    embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)

    # Cosine similarities between consecutive sentence embeddings
    # (dot product of normalized vectors == cosine similarity)
    similarities = [float(np.dot(embeddings[i], embeddings[i + 1])) for i in range(len(embeddings) - 1)]

    breakpoints = _find_breakpoints(similarities, breakpoint_threshold_type, breakpoint_threshold_amount)
    logger.debug(
        "Semantic chunking: %d sentences, %d breakpoints (type=%s, amount=%s)",
        len(sentences),
        len(breakpoints),
        breakpoint_threshold_type,
        breakpoint_threshold_amount,
    )

    chunks = _build_semantic_chunks(sentences, breakpoints, doc, max_chunk_size, chunk_overlap, tokenizer_name)

    if dump_chunks_on_file:
        output_name = f"{doc.metadata.get(DOC_FILE_NAME, 'doc')}_{doc.metadata.get(PARAGRAPH_NUMBER, 0)}"
        _log_chunks_on_file(chunks, Path(Config().get("vectoria_logs_dir")) / "chunks" / f"{output_name}_chunks.txt")

    return chunks


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Splits on:
    - Paragraph boundaries (two or more newlines).
    - Sentence-ending punctuation (. ! ?) followed by whitespace and an
      uppercase letter (handles both ASCII and Latin-extended accented capitals).
    """
    sentences: list[str] = []
    for paragraph in re.split(r"\n{2,}", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\u00C0-\u024F])", paragraph)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


def _find_breakpoints(
    similarities: list[float],
    threshold_type: str,
    threshold_amount: float,
) -> list[int]:
    """Return indices i where a breakpoint should be placed between sentence i and i+1."""
    import numpy as np

    if not similarities:
        return []

    arr = np.array(similarities)

    if threshold_type == "percentile":
        # Split where similarity is below the (100 - amount)-th percentile.
        # amount=70 → threshold at 30th percentile → bottom 30% of gaps are splits.
        threshold = float(np.percentile(arr, 100.0 - threshold_amount))
        return [i for i, s in enumerate(arr) if s < threshold]

    elif threshold_type == "standard_deviation":
        threshold = float(arr.mean() - threshold_amount * arr.std())
        return [i for i, s in enumerate(arr) if s < threshold]

    elif threshold_type == "interquartile":
        q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        threshold = q1 - threshold_amount * (q3 - q1)
        return [i for i, s in enumerate(arr) if s < threshold]

    elif threshold_type == "gradient":
        # Split where the *drop* in similarity is unusually large.
        if len(arr) < 2:
            return []
        gradients = np.abs(np.diff(arr))
        grad_threshold = float(np.percentile(gradients, threshold_amount))
        return [i + 1 for i, g in enumerate(gradients) if g >= grad_threshold]

    else:
        raise ValueError(
            f"Unknown breakpoint_threshold_type: '{threshold_type}'. "
            "Choose one of: percentile, standard_deviation, interquartile, gradient."
        )


def _count_tokens(text: str, tokenizer_name: str | None) -> int:
    """Count tokens using a HuggingFace tokenizer, or words as fallback."""
    if tokenizer_name:
        return len(_load_tokenizer(tokenizer_name).encode(text))
    return len(text.split())


def _build_semantic_chunks(
    sentences: list[str],
    breakpoints: list[int],
    source_doc: Document,
    max_chunk_size: int | None,
    chunk_overlap: int,
    tokenizer_name: str | None,
) -> list[Document]:
    """Merge sentences into chunks at breakpoint positions."""
    breakpoint_set = set(breakpoints)
    groups: list[str] = []
    current: list[str] = [sentences[0]]

    for i in range(1, len(sentences)):
        if (i - 1) in breakpoint_set:
            groups.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        groups.append(" ".join(current))

    # Build Document objects
    chunks: list[Document] = []
    for seq_id, text in enumerate(groups):
        chunks.append(
            Document(
                page_content=text,
                metadata={**source_doc.metadata, "seq_id": seq_id},
            )
        )

    # Optional: split oversized chunks with recursive splitter as fallback
    if max_chunk_size is not None:
        refined: list[Document] = []
        for chunk in chunks:
            if _count_tokens(chunk.page_content, tokenizer_name) > max_chunk_size:
                logger.debug(
                    "Semantic chunk exceeds max_chunk_size=%d — applying recursive split fallback",
                    max_chunk_size,
                )
                sub_chunks = recursive_character_text_splitter(
                    chunk,
                    chunk_size=max_chunk_size,
                    chunk_overlap=chunk_overlap,
                    tokenizer_name=tokenizer_name,
                )
                refined.extend(sub_chunks)
            else:
                refined.append(chunk)
        # Re-assign seq_ids after potential expansion
        for i, c in enumerate(refined):
            c.metadata["seq_id"] = i
        return refined

    return chunks


def _log_chunks_on_file(documents, file_path: Path):
    Path(file_path.parent).mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for doc in documents:
            print(doc, file=f)
