#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import copy
import os
import sys

venv_path = os.environ.get("VIRTUAL_ENV") or sys.prefix
python_version = sys.version[0:4]
lib_path = venv_path + "/lib/python" + python_version + "/site-packages"

# strip it from sys.path if it exists
if lib_path in sys.path:
    sys.path.remove(lib_path)

# restore it in the right position
sys.path.insert(1, lib_path)

import pytest
import requests

from vectoria_lib.core.config import Config
from vectoria_lib.core.io.file_io import get_file_io
from vectoria_lib.core.paths import TEST_DIR


def pytest_addoption(parser):
    pass


@pytest.fixture(autouse=True)
def _reset_config_state():
    """Config is a process-wide Singleton — any test mutating it via .set()/.load_config()
    would otherwise leak state into every later test, regardless of whether it requests
    the `config` fixture. Snapshot/restore keeps each test isolated without touching
    the Singleton itself."""
    snapshot = copy.deepcopy(Config().config)
    yield
    Config().config = snapshot


@pytest.fixture(scope="session")
def data_dir():
    return TEST_DIR / "data"


@pytest.fixture(scope="session")
def docx_test_file(data_dir):
    return data_dir / "docx" / "test_docx.docx"


@pytest.fixture(scope="session")
def index_test_folder(data_dir):
    return data_dir / "index" / "attention_is_all_you_need_index"


@pytest.fixture(scope="session")
def chroma_index_test_folder(tmp_path_factory):
    from langchain_core.documents import Document

    from vectoria_lib.rag.preprocessing.chunking import recursive_character_text_splitter
    from vectoria_lib.rag.vector_store.chroma_vector_store import ChromaVectorStore

    doc = Document(
        page_content=(
            "The Transformer model introduced in this paper eliminates the need for recurrent or "
            "convolutional networks, relying solely on attention mechanisms. "
            "The Transformer model was tested on two major tasks: WMT 2014 English-to-German and "
            "English-to-French translation tasks, as well as English constituency parsing using the "
            "Penn Treebank dataset. The self-attention mechanism computes attention scores through "
            "scaled dot products of queries and keys, followed by a softmax function to weight the values."
        ),
        metadata=dict(paragraph_number="1", paragraph_name="intro"),
    )
    docs = recursive_character_text_splitter(doc, chunk_size=50, chunk_overlap=20)

    output_dir = tmp_path_factory.mktemp("chroma_index")
    vs = ChromaVectorStore(
        model_name="BAAI/bge-m3",
        device="cpu",
        normalize_embeddings=False,
    ).make_index(docs)
    return vs.dump_to_disk(output_dir)


@pytest.fixture()
def eval_data_qa():
    return get_file_io("json").read(TEST_DIR / "data" / "eval" / "qa.json")


@pytest.fixture(scope="function")
def config(request, write_config_file_for_tests, monkeypatch):
    # Loading here only mutates this process's Config singleton. Multiprocessing
    # tests spawn fresh interpreters (mp.get_context("spawn")) that don't inherit
    # it and would otherwise fall back to the real etc/docker/config.yaml — set
    # the env var so spawned workers load the same test config.
    monkeypatch.setenv("VECTORIA_CONFIG_FILE", str(write_config_file_for_tests))
    c = Config().load_config(write_config_file_for_tests)
    c.set("vectoria_logs_dir", value=TEST_DIR / "logs" / request.node.name)
    return c


@pytest.fixture(scope="session", autouse=True)
def openai_server_status_fn():
    return _ping_openai_server


def _ping_openai_server(engine_config: dict):
    try:
        response = requests.get(engine_config.get("openai_api_base").replace("/v1", "/version"))
        return response.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def write_config_file_for_tests(request):
    config_file_for_tests_path = "/tmp/test.yaml"
    with open(config_file_for_tests_path, "w") as f:
        f.write("""
vectoria_logs_dir: /tmp/vectoria_logs
log_level: DEBUG
system_prompts_lang: eng

data_ingestion:
  multiprocessing: true
  
  extraction:
    format: docx
    dump_doc_structure_on_file: true
    regexes_for_metadata_extraction:
      - name: DOC_ID
        pattern: '^Document Title'

  regexes_for_replacement:
    - name: remove_multiple_spaces
      pattern: '[ \t]{2,}'
      replace_with: ' '
    - name: remove_bullets
      pattern: '^\\s*[\u2022\u25aa\u27a2]\\s*'
      replace_with: ''
    - name: remove_ligature_st
      pattern: 'ﬆ'
      replace_with: 'st'
    - name: replace_fi
      pattern: 'ﬁ'
      replace_with: 'fi'
    - name: replace_fl
      pattern: 'ﬂ'
      replace_with: 'fl'
    - name: replace_ffi
      pattern: 'ﬃ'
      replace_with: 'ffi'
    - name: replace_ffl
      pattern: 'ﬄ'
      replace_with: 'ffl'
    - name: replace_ft
      pattern: 'ﬅ'
      replace_with: 'ft'
    - name: replace_st
      pattern: 'ﬆ'
      replace_with: 'st'
    - name: replace_AA
      pattern: 'Ꜳ'
      replace_with: 'AA'
    - name: replace_AE
      pattern: 'Æ'
      replace_with: 'AE'
    - name: replace_aa
      pattern: 'ꜳ'
      replace_with: 'aa'


  chunking:
    chunk_size: 512
    chunk_overlap: 256
    separators: ["\n\n", "\n", " ", ""]
    is_separator_regex: [false, false, false, false]
    dump_chunks_on_file: true

vector_store:
  name: chroma
  model_name: BAAI/bge-m3
  device: cpu
  normalize_embeddings: false
  collection_name: vectoria

retriever:
  enabled: true
  search_type: hybrid
  top_k: 5
  bm25_weight: 0.5

reranker:
  enabled: false
  reranked_top_k: 3
  model_name: BAAI/bge-reranker-v2-m3
  device: cpu

full_paragraphs_retriever:
  enabled: false

chat_history:
  enabled: false

# QA agent
inference_engine:
  name: huggingface
  url: null
  api_key: null
  model_name: HuggingFaceTB/SmolLM-135M
  device: cuda
  load_in_4bit: false
  load_in_8bit: false
  max_new_tokens: 10
  trust_remote_code: false
  device_map: auto
  do_sample: false
  temperature: 0


# Evaluation
evaluation:
  tool: ragas
  
  inference_engine:
    name: openai
    model_name: meta-llama/Llama-3.2-1B-Instruct
    temperature: 0.7
    max_tokens: 256
    top_p: 1
    frequency_penalty: 0
    presence_penalty: 0
    n: 1
    openai_api_key: EMPTY
    openai_api_base: http://localhost:8000/v1
    request_timeout: null
    max_retries: 2
    seed: null
    streaming: false


  embeddings_engine:
    name: openai
    model_name: BAAI/bge-m3
    temperature: 0.7
    max_tokens: 256
    top_p: 1
    frequency_penalty: 0
    presence_penalty: 0
    n: 1
    openai_api_key: EMPTY
    openai_api_base: http://localhost:8000/v1
    batch_size: 20
    request_timeout: null
    max_retries: 2
    seed: null
    streaming: false
""")
    return config_file_for_tests_path
