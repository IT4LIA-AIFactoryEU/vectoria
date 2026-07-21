#
# VECTORIA
#
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator

# ── Shared ────────────────────────────────────────────────────────────────────


class RegexConfig(BaseModel):
    name: str
    pattern: str
    replace_with: Optional[str] = None


# ── Inference engine (shared by main LLM, reranker, evaluation) ───────────────


class InferenceEngineConfig(BaseModel):
    """
    Covers both OpenAI-compatible and HuggingFace engines.
    extra="allow" preserves engine-specific fields (openai_api_base, device_map, etc.)
    """

    model_config = ConfigDict(extra="allow")

    name: Literal["openai", "huggingface", "llamacpp", "anthropic"]
    model_name: str

    # Common optional fields
    url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 150

    # HuggingFace specific
    device: Optional[str] = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    max_new_tokens: int = 150
    trust_remote_code: bool = False
    device_map: str = "auto"
    do_sample: bool = False

    # OpenAI-compatible specific
    base_url: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_api_key: Optional[str] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    n: Optional[int] = None
    request_timeout: Optional[float] = None
    max_retries: Optional[int] = None
    seed: Optional[Any] = None
    streaming: bool = False
    batch_size: Optional[int] = None

    # llama.cpp specific (model_name is used as model_path, i.e. path to .gguf file)
    n_ctx: int = 512
    n_gpu_layers: Optional[int] = None
    n_threads: Optional[int] = None
    n_batch: int = 8
    use_mlock: bool = False
    use_mmap: bool = True
    repeat_penalty: float = 1.1


# ── Data ingestion ─────────────────────────────────────────────────────────────


class ExtractionConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    format: Literal["pdf", "docx", "txt"] = "pdf"
    backend: Literal["native", "docling", "auto"] = "native"
    dump_doc_structure_on_file: bool = False
    regexes_for_metadata_extraction: list[RegexConfig] = []


class ChunkingConfig(BaseModel):
    # Chunking strategy: "recursive" (fixed-size) or "semantic" (topic-aware).
    method: Literal["recursive", "semantic"] = "recursive"

    # ── Recursive chunking parameters ─────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 256
    separators: list[str] = ["\n\n", "\n", " ", ""]
    is_separator_regex: list[bool] = [False, False, False, False]
    dump_chunks_on_file: bool = False
    # When set, chunk_size/chunk_overlap are in tokens (HuggingFace tokenizer).
    # Leave null to use character count.
    tokenizer_name: Optional[str] = None

    # ── Semantic chunking parameters (method: semantic) ────────────────────────
    # Embedding model for sentence similarity. Defaults to vector_store.model_name.
    semantic_embedding_model: Optional[str] = None
    # Device for the embedding model ("cpu", "cuda", "mps").
    semantic_device: str = "cpu"
    # How to detect breakpoints (topic changes):
    #   percentile         — split where similarity < (100-amount)-th percentile
    #   standard_deviation — split where similarity < mean - amount * std
    #   interquartile      — split where similarity < Q1 - amount * IQR
    #   gradient           — split where drop in similarity is in top-amount percentile
    semantic_breakpoint_threshold_type: Literal["percentile", "standard_deviation", "interquartile", "gradient"] = (
        "percentile"
    )
    # Numeric value for the chosen threshold type.
    # For "percentile": 70.0 → split the bottom 30% similarity gaps (more splits = higher value).
    semantic_breakpoint_threshold_amount: float = 70.0
    # Optional max chunk size (tokens if tokenizer_name set, otherwise words).
    # Semantic chunks exceeding this limit are further split with the recursive splitter.
    semantic_max_chunk_size: Optional[int] = None

    @field_validator("chunk_size", "chunk_overlap")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v


class DataIngestionConfig(BaseModel):
    multiprocessing: bool = True
    # Max worker processes for the preprocessing pool.
    # None → min(cpu_count(), 4). Set to 1 to force single-process (same as multiprocessing: false).
    multiprocessing_workers: Optional[int] = None
    extraction: ExtractionConfig
    regexes_for_replacement: list[RegexConfig] = []
    chunking: ChunkingConfig


# ── Vector store ───────────────────────────────────────────────────────────────


class VectorStoreConfig(BaseModel):
    name: Literal["faiss", "chroma", "qdrant"] = "chroma"
    model_name: str
    device: str = "cpu"
    normalize_embeddings: bool = True
    embedding_batch_size: int = 64
    collection_name: str = "vectoria"
    # Qdrant-specific (optional)
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_prefer_grpc: bool = False


# ── Retriever ──────────────────────────────────────────────────────────────────


class RetrieverConfig(BaseModel):
    enabled: bool = True
    name: Optional[str] = None
    top_k: int = 5
    search_type: Literal["similarity", "mmr", "hybrid"] = "hybrid"
    multi_query: bool = False
    # mmr-specific
    fetch_k: int = 5
    lambda_mult: float = 0.5
    # hybrid-specific
    bm25_weight: float = 0.5
    # cross-document diversity: minimum chunks per source document in the final result.
    # 0 = disabled. When > 0, top_k should be set high enough to fetch candidates
    # from all sources before diversity enforcement (e.g. top_k >= n_docs * min_per_source * 3).
    min_per_source: int = 0
    # Post-reranker MMR (semantic diversity, reduces intra-document redundancy)
    mmr_enabled: bool = False
    mmr_lambda: float = 0.7
    mmr_top_k: int = 0

    @field_validator("top_k", "fetch_k")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v


# ── Reranker ───────────────────────────────────────────────────────────────────


class RerankerConfig(BaseModel):
    enabled: bool = False
    reranked_top_k: int = 3
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"


# ── Full paragraphs & chat history ────────────────────────────────────────────


class FullParagraphsRetrieverConfig(BaseModel):
    enabled: bool = False


class ChatHistoryConfig(BaseModel):
    enabled: bool = False
    summarize_after_n_messages: int = 6
    max_conv_messages: int = 4
    max_tool_messages: int = 3
    checkpoint_db_path: Optional[str] = None
    # Either a full InferenceEngineConfig dict, or a plain model_name string
    # (in the latter case, endpoint/api_key/etc. are inherited from inference_engine)
    summary_llm: Optional[Any] = None


# ── Agent guardrails ───────────────────────────────────────────────────────────


class ToolCallLimitConfig(BaseModel):
    enabled: bool = True
    # Max tool calls allowed within a single agent turn (one user message → one
    # answer). Kept below the graph's recursion_limit so the agent degrades
    # gracefully (answers with what it has) instead of hitting GraphRecursionError.
    run_limit: int = 8
    # Max tool calls allowed across an entire conversation thread.
    thread_limit: Optional[int] = None
    # "continue": block exceeded tools, let the model answer with what it has.
    # "error": raise ToolCallLimitExceededError. "end": stop immediately.
    exit_behavior: Literal["continue", "error", "end"] = "continue"


class HumanApprovalConfig(BaseModel):
    enabled: bool = True
    # Tool names that require explicit user confirmation before executing.
    tools: list[str] = ["delete_document"]


class AgentGuardrailsConfig(BaseModel):
    tool_call_limit: ToolCallLimitConfig = ToolCallLimitConfig()
    human_approval: HumanApprovalConfig = HumanApprovalConfig()


# ── llama-server auto-start ───────────────────────────────────────────────────


class LlamaServerConfig(BaseModel):
    enabled: bool = False
    executable: str = "./build/bin/llama-server"
    model: str = ""
    port: int = 3002
    ctx_size: int = 8192
    n_predict: Optional[int] = None  # omitted if None
    n_gpu_layers: Optional[int] = None  # omitted if None (CPU-only when absent)
    threads: Optional[int] = None
    batch_size: Optional[int] = None
    ubatch_size: Optional[int] = None
    cache_type_k: Optional[str] = None  # e.g. "q8_0", "f16"
    cache_type_v: Optional[str] = None
    mlock: bool = False
    extra_args: list[str] = []


# ── Graph store ───────────────────────────────────────────────────────────────


class GraphStoreConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    backend: Literal["networkx"] = "networkx"
    extraction_device: Literal["cpu", "cuda"] = "cpu"
    model_path: Optional[str] = None
    entity_types: Optional[list[str]] = None
    min_entity_confidence: float = 0.5
    relation_method: Literal["cooccurrence"] = "cooccurrence"


# ── Wiki ──────────────────────────────────────────────────────────────────────


class WikiConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    pending_threshold: int = 5
    max_topics_per_doc: int = 3
    max_topics_per_batch: int = 8
    language_normalization: bool = True
    similarity_threshold: float = 0.80
    canonical_topic_language: Literal["it", "en"] = "it"
    synthesis_context_chunks: int = 4
    max_page_words: int = 400


# ── Root ──────────────────────────────────────────────────────────────────────


class VectoriaConfig(BaseModel):
    """
    Root schema for Vectoria configuration.
    extra="allow" lets any unknown top-level key pass through so that
    custom/extended configs do not break existing deployments.
    """

    model_config = ConfigDict(extra="allow")

    vectoria_logs_dir: Path
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
    system_prompts_lang: Literal["it", "eng"] = "eng"
    # Base directory for local HuggingFace models (embeddings, reranker, tokenizer).
    # When set, model_name values that are not absolute paths are resolved as
    # subdirectories of models_dir.  Leave null to download from HuggingFace Hub.
    models_dir: Optional[str] = None

    data_ingestion: DataIngestionConfig
    vector_store: VectorStoreConfig
    retriever: RetrieverConfig
    reranker: RerankerConfig
    full_paragraphs_retriever: FullParagraphsRetrieverConfig = FullParagraphsRetrieverConfig()
    chat_history: ChatHistoryConfig = ChatHistoryConfig()
    agent_guardrails: AgentGuardrailsConfig = AgentGuardrailsConfig()
    inference_engine: InferenceEngineConfig
    llama_server: Optional[LlamaServerConfig] = None
    graph_store: Optional[GraphStoreConfig] = None
    wiki: Optional[WikiConfig] = None
