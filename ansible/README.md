# Vectoria — Ansible Deploy (Leonardo HPC)

Fully automated deploy of Vectoria on the Leonardo supercomputer (CINECA). A single playbook syncs the repo, installs Python deps with `uv`, downloads the LLM, builds the vLLM + OpenWebUI Singularity containers, submits the SLURM job, waits for it to start, and opens the SSH tunnels on your laptop.

## Features

- **One command, end to end** — no manual `sbatch`, no manual tunnels.
- Builds vLLM and OpenWebUI Singularity containers on the cluster.
- Generates a SLURM launcher tuned for A100 (4 GPU node).
- GPU-tuned Vectoria config (docling + semantic chunking + GPU embeddings).
- Auto-opens SSH tunnels from your PC to the compute node in the background.
- Re-runs are idempotent: containers and downloaded models are cached.

## Prerequisites

- CINECA account with hours on `boost_usr_prod` and SSH access to `login.leonardo.cineca.it`.
- Valid CINECA SSH certificate (`step ssh certificate ...`) — see [`etc/hpc/leonardo_connect.sh`](../etc/hpc/leonardo_connect.sh) (`cert` subcommand).
- HuggingFace login on Leonardo for gated models (`huggingface-cli login` on the login node).
- Ansible on your local machine (installed via `uv`):

```bash
uv pip install -r requirements.txt
```

## Quick Start

```bash
cd ansible/

# 1. Edit inventory/hosts → set ansible_user=<YOUR_CINECA_USERNAME>
# 2. Edit roles/vectoria/vars/config.yml → set account, api_token, llm_model

ansible-playbook -i inventory deploy.yml
```

That's it. When the playbook returns, the job is `RUNNING` and the tunnels are open on your PC:

- Upload: `http://localhost:8081/upload`
- Chat (OpenWebUI): `http://localhost:8082`

To stop the tunnels: `kill $(cat /tmp/vectoria_tunnel.pid)`. Tunnel logs: `/tmp/vectoria_tunnel.log`.

## What `deploy.yml` Does

The playbook runs the following steps in order, fully automatically:

1. **Sync repo** to `$SCRATCH/vectoria` (excludes `.venv`, `__pycache__`, `.git`, `ansible/`, `data/`).
2. **Install `uv`** on Leonardo (if missing) and run `uv sync --extra docling --extra api --extra observability`.
3. **Cache tiktoken encodings** in `~/.tiktoken_cache` (compute nodes have no internet).
4. **Build vLLM container**: pull sandbox on login node → convert to `.sif` on a compute node via `srun`.
5. **Build OpenWebUI container** (same approach).
6. **Generate configs**: `etc/hpc/vectoria_hpc_launcher.job` (SLURM) and `etc/hpc/default_config.yaml` — written on the remote host, not committed to the repo.
7. **Create data dirs** in `$SCRATCH/vectoria_data/`.
8. **Download the LLM** into `$SCRATCH/hf_models` (`download_LLMs.yml`, async, 30–90 min).
9. **Submit the SLURM job** with `sbatch`.
10. **Wait for `RUNNING`** state (polls `squeue` every 15 s, up to 15 min).
11. **Open SSH tunnels** from your PC to the compute node (`wait_and_tunnel.yml`).

> Container builds take 20–40 min each. If the `.sif` files already exist, the build steps are skipped. Same for cached LLM weights.

## Configuration

Edit [`roles/vectoria/vars/config.yml`](roles/vectoria/vars/config.yml).

Required:

| Variable | Description |
|---|---|
| `account` | CINECA budget account (`saldo -b`) |
| `api_token` | Secret token for API auth (any value) |
| `llm_model` | HuggingFace model ID (e.g. `google/gemma-4-31B-it`) |

Common optional:

| Variable | Default | Description |
|---|---|---|
| `vllm_gpus` | `"0,1"` | GPUs for vLLM |
| `vectoria_gpus` | `"2,3"` | GPUs for Vectoria |
| `vllm_tensor_parallel` | `2` | vLLM tensor parallel size |
| `vllm_image_url` | `docker://vllm/vllm-openai:gemma4` | vLLM container image |
| `vllm_tool_call_parser` | `"gemma4"` | Tool call parser (match model) |
| `vectoria_local_port` | `8081` | Local port for Vectoria |
| `openwebui_local_port` | `8082` | Local port for OpenWebUI |
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedder |
| `reranker_model` | `BAAI/bge-reranker-v2-m3` | Reranker |
| `chunking_method` | `"semantic"` | `semantic` (GPU) or `recursive` |
| `system_prompts_lang` | `"eng"` | `eng` or `it` |

## Reconnecting After Disconnect

If you lose the local tunnels (laptop sleep, disconnect) but the SLURM job is still running, just reopen them — no need to redeploy:

```bash
ansible-playbook -i inventory connect.yml
# or with explicit job id:
ansible-playbook -i inventory connect.yml -e job_id=<SLURM_ID>
```

## Update After Code Changes

```bash
ansible-playbook -i inventory deploy.yml
```

Re-syncs code, regenerates configs, resubmits the job, reopens tunnels. Containers and LLM weights are not re-downloaded.

## See Also

- [`etc/hpc/README.md`](../etc/hpc/README.md) — what runs in the SLURM job, GPU-tuned config, retrieval params, model sizing.
- Root [`README.md`](../README.md) — project overview, local/Docker quick start, agent tools, multi-tenancy.
