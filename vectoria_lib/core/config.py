import logging
import os
import re
from pathlib import Path

import yaml
from pydantic import ValidationError

from vectoria_lib.core.config_schema import VectoriaConfig
from vectoria_lib.core.logger import setup_logger
from vectoria_lib.core.paths import ETC_DIR
from vectoria_lib.core.utils import Singleton


class Config(metaclass=Singleton):
    def __init__(self, config_path: Path | str = None):
        self.config_stream_logger = setup_logger("config_logger", "INFO")  # <- stream logger
        self.logger = logging.getLogger("common")
        self.config = {}
        self.load_config(config_path)
        self._disable_tokenizer_parallelism()
        self._optimize_torch()

    @staticmethod
    def _expand_env_variables(obj):
        """Recursively expand ${VAR} and ${VAR:-default} in config strings."""
        if isinstance(obj, str):
            # Full-string placeholder: ${VAR} or ${VAR:-default}
            match_full = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*))?\}", obj)
            if match_full:
                var_name, default = match_full.group(1), match_full.group(2)
                val = os.environ.get(var_name)
                if val is not None:
                    return val
                return default  # None when no default (optional field)

            # Partial placeholders embedded in a string
            def replace_var(match):
                var_name, default = match.group(1), match.group(2)
                val = os.environ.get(var_name)
                if val is not None:
                    return val
                if default is not None:
                    return default
                raise ValueError(f"Environment variable '{var_name}' is referenced in config but not set")

            return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}", replace_var, obj)
        elif isinstance(obj, dict):
            return {k: Config._expand_env_variables(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Config._expand_env_variables(item) for item in obj]
        else:
            return obj

    @staticmethod
    def _load_dotenv():
        from pathlib import Path as _Path
        dotenv = _Path(__file__).parent.parent.parent / ".env"
        if not dotenv.exists():
            return
        with open(dotenv, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)


    @staticmethod
    def _apply_provider(raw: dict) -> dict:
        provider = os.environ.get("LLM_PROVIDER", "custom").lower()
        if provider == "custom":
            return raw
        ie = raw.setdefault("inference_engine", {})
        if provider == "openai":
            ie["name"] = "openai"
            ie["base_url"] = None
            ie["api_key"] = os.environ.get("OPENAI_API_KEY", "")
        elif provider == "anthropic":
            ie["name"] = "anthropic"
            ie["api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
        elif provider == "google":
            ie["name"] = "openai"  # usa OpenAI-compat di Google
            ie["base_url"] = "https://generativelanguage.googleapis.com/v1beta/openai/"
            ie["api_key"] = os.environ.get("GOOGLE_API_KEY", "")
        elif provider == "ollama":
            ie["name"] = "openai"
            ie["base_url"] = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/v1"
            ie["api_key"] = "ollama"
        ie["provider"] = provider
        return raw

    def load_config(self, config_path: Path | str = None):
        self._load_dotenv()
        if config_path is None:
            config_path = os.environ.get("VECTORIA_CONFIG_FILE") or ETC_DIR / "docker" / "config.yaml"
            self.config_stream_logger.info("Loading default configuration: %s", config_path)
        else:
            self.config_stream_logger.debug("Loading configuration from %s", config_path)

        with open(config_path, encoding="utf-8") as file:
            raw = yaml.safe_load(file)

        # Expand environment variables in config (${VAR_NAME} syntax)
        raw = self._expand_env_variables(raw)

        # Apply provider-specific configurations
        raw = self._apply_provider(raw)

        try:
            validated = VectoriaConfig(**raw)
        except ValidationError as e:
            raise ValueError(f"Invalid configuration in '{config_path}':\n{e}") from e

        # model_dump(mode="python") preserves Path objects and other Python types.
        # Backward compat: self.config remains a plain dict so get()/set() are unchanged.
        self.config = validated.model_dump(mode="python")

        self.config["vectoria_logs_dir"] = Path(self.config["vectoria_logs_dir"]).expanduser()
        self.config["vectoria_logs_dir"].mkdir(parents=True, exist_ok=True)

        self._resolve_model_paths()

        _safe = {k: ("***" if "api_key" in k or "password" in k else v) for k, v in self.config.items()}
        self.config_stream_logger.debug("Configuration loaded and validated: %s", _safe)
        return self

    def update_from_args(self, args):
        """Override config parameters from CLI arguments"""
        for key, value in vars(args).items():
            if value is not None:
                self.config[key] = value

    def get(self, key, subkey=None):
        if subkey is None:
            if isinstance(self.config[key], dict):
                return self.config[key].copy()
            return self.config[key]
        else:
            if isinstance(self.config[key][subkey], dict):
                return self.config[key][subkey].copy()
            return self.config[key][subkey]

    def set(self, key, subkey=None, value=None):
        if value is None:
            self.logger.warning("Value is None for key: %s, subkey: %s", key, subkey)
            return
        if subkey is None:
            if key not in self.config:
                raise ValueError(f"Key {key} not found in config")
            self.config[key] = value
        else:
            if key not in self.config:
                raise ValueError(f"Key {key} not found in config")
            if subkey not in self.config[key]:
                raise ValueError(f"Subkey {subkey} not found in key {key} of config")
            self.config[key][subkey] = value

    def _resolve_model_paths(self):
        """Resolve relative model names to absolute paths using models_dir."""
        raw = self.config.get("models_dir")
        if not raw:
            return
        models_dir = Path(raw).expanduser().resolve()

        def resolve(name: str) -> str:
            if not name:
                return name
            p = Path(name)
            if p.is_absolute():
                return name  # already absolute — keep as-is
            candidate = models_dir / name
            if candidate.exists():
                return str(candidate)
            return name  # fallback: treat as HF hub ID

        # vector_store embedding model
        vs = self.config.get("vector_store", {})
        if vs.get("model_name"):
            self.config["vector_store"]["model_name"] = resolve(vs["model_name"])

        # reranker model
        rr = self.config.get("reranker", {})
        if rr.get("model_name"):
            self.config["reranker"]["model_name"] = resolve(rr["model_name"])

        # chunking tokenizer
        try:
            tok = self.config["data_ingestion"]["chunking"].get("tokenizer_name")
            if tok:
                self.config["data_ingestion"]["chunking"]["tokenizer_name"] = resolve(tok)
        except (KeyError, TypeError):
            pass

    def _disable_tokenizer_parallelism(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def _optimize_torch(self):
        import torch
        # Allow TF32 on Ampere+ (A100): ~3× faster matmul with negligible precision loss.
        # No-op on CPU or older GPUs.
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
