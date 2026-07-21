#
# VECTORIA
#
import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger("llm")


class LlamaServerManager:
    """Manages a llama-server subprocess lifecycle.

    Spawned at API startup, terminated at shutdown. Works identically for
    local and Docker deployments — only the paths in the config differ.
    """

    def __init__(self, cfg: dict):
        self.executable = Path(cfg["executable"]).expanduser()
        self.model = Path(cfg["model"]).expanduser()
        self.port = int(cfg.get("port", 3002))
        self.ctx_size = int(cfg.get("ctx_size", 8192))
        self.n_predict = cfg.get("n_predict")
        self.n_gpu_layers = cfg.get("n_gpu_layers")
        self.threads = cfg.get("threads")
        self.batch_size = cfg.get("batch_size")
        self.ubatch_size = cfg.get("ubatch_size")
        self.cache_type_k = cfg.get("cache_type_k")
        self.cache_type_v = cfg.get("cache_type_v")
        self.mlock = bool(cfg.get("mlock", False))
        self.extra_args = list(cfg.get("extra_args") or [])
        self._process: subprocess.Popen | None = None

    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start llama-server and block until it is ready (or raise)."""
        if not self.executable.exists():
            raise FileNotFoundError(
                f"llama-server executable not found: {self.executable}\n"
                "Set llama_server.executable in your config to the correct path."
            )
        if not self.model.exists():
            raise FileNotFoundError(
                f"GGUF model not found: {self.model}\nSet llama_server.model in your config to the correct path."
            )

        cmd = [
            str(self.executable),
            "--model",
            str(self.model),
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.ctx_size),
        ]

        if self.n_predict is not None:
            cmd += ["--n-predict", str(self.n_predict)]
        if self.n_gpu_layers is not None:
            cmd += ["-ngl", str(self.n_gpu_layers)]
        if self.threads is not None:
            cmd += ["--threads", str(self.threads)]
        if self.batch_size is not None:
            cmd += ["--batch-size", str(self.batch_size)]
        if self.ubatch_size is not None:
            cmd += ["--ubatch-size", str(self.ubatch_size)]
        if self.cache_type_k is not None:
            cmd += ["--cache-type-k", self.cache_type_k]
        if self.cache_type_v is not None:
            cmd += ["--cache-type-v", self.cache_type_v]
        if self.mlock:
            cmd += ["--mlock"]

        cmd += self.extra_args

        logger.info("Starting llama-server: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._wait_ready()
        logger.info("llama-server ready on port %d (pid=%d)", self.port, self._process.pid)

    def stop(self) -> None:
        """Terminate llama-server gracefully."""
        if self._process is None or self._process.poll() is not None:
            return
        logger.info("Stopping llama-server (pid=%d)", self._process.pid)
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()

    # ------------------------------------------------------------------

    def _wait_ready(self, timeout: int = 120) -> None:
        import urllib.error
        import urllib.request

        url = f"http://localhost:{self.port}/health"
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            rc = self._process.poll()
            if rc is not None:
                out = self._process.stdout.read().decode(errors="replace") if self._process.stdout else ""
                raise RuntimeError(f"llama-server exited with code {rc}.\n{out[-2000:]}")
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            time.sleep(1)

        self._process.kill()
        raise RuntimeError(f"llama-server did not become ready within {timeout}s on port {self.port}")
