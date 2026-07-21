#!/bin/bash
# leonardo_connect.sh — Helper script to connect to Leonardo HPC and manage SSH tunnels.
#
# Usage:
#   ./leonardo_connect.sh cert                 # Renew SSH certificate
#   ./leonardo_connect.sh ssh                  # Open SSH session on Leonardo
#   ./leonardo_connect.sh tunnel <job_id|ip>   # Open vLLM standalone tunnel (port 8001)
#   ./leonardo_connect.sh vectoria [job_id]    # Open Vectoria + OpenWebUI tunnels
#                                              #   → auto-detects job if job_id is omitted
#   ./leonardo_connect.sh wiki-pull <tenant> [local_dir]
#                                              # One-shot: mirror a tenant's Wiki pages from
#                                              #   $SCRATCH to a local directory
#                                              #   (default: data/wiki_sync/<tenant>/)
#   ./leonardo_connect.sh wiki-push <tenant> [local_dir]
#                                              # One-shot: push local Wiki edits back to
#                                              #   $SCRATCH (asks for confirmation first)

# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../../.env"
# ponytail: load .env silently; hard vars still validate below
[[ -f "$ENV_FILE" ]] && set -a && source "$ENV_FILE" && set +a

# Edit these to match your CINECA account, or set them as environment variables.
# LEONARDO_USER="${LEONARDO_USER:-<YOUR_CINECA_USERNAME>}"
LEONARDO_USER="${CINECA_USER:-${LEONARDO_USER:?set CINECA_USER in .env}}"
LEONARDO_HOST="login02-ext.leonardo.cineca.it"
CINECA_EMAIL="${CINECA_EMAIL:?set CINECA_EMAIL in .env}"
CERT_KEY="${CINECA_SSH_KEY:-${CERT_KEY:-$HOME/.ssh/cineca_key}}"

VECTORIA_JOB_NAME="vectoria_hpc"

# Ports forwarded to your PC:
#   8081 → Vectoria API  (port 8080 on compute node)
#   8082 → OpenWebUI     (port 8090 on compute node)
VECTORIA_LOCAL_PORT=8081
VECTORIA_REMOTE_PORT=8080
OPENWEBUI_LOCAL_PORT=8082
OPENWEBUI_REMOTE_PORT=8090
# ──────────────────────────────────────────────────────────────────────────────

usage() {
    echo "Usage: $0 {cert|ssh|tunnel <job_id|ip>|vectoria [job_id]|wiki-pull <tenant> [local_dir]|wiki-push <tenant> [local_dir]}"
    echo ""
    echo "  cert                   Renew CINECA SSH certificate"
    echo "  ssh                    Open SSH session on Leonardo"
    echo "  tunnel  <job_id|ip>    vLLM standalone tunnel  → localhost:8001"
    echo "  vectoria [job_id]      Vectoria + OpenWebUI    → localhost:8081 / localhost:8082"
    echo "                         If job_id is omitted, it is detected automatically."
    echo "  wiki-pull <tenant> [local_dir]"
    echo "                         One-shot: mirror the tenant's Wiki pages from \$SCRATCH to"
    echo "                         local_dir (default: data/wiki_sync/<tenant>/)."
    echo "  wiki-push <tenant> [local_dir]"
    echo "                         One-shot: push local Wiki edits in local_dir back to"
    echo "                         \$SCRATCH. Asks for typed confirmation before overwriting."
    echo ""
    echo "  Note: Ray dashboard not available (vLLM manages Ray internally)."
    exit 1
}

renew_cert() {
    echo "[*] Renewing SSH certificate..."
    step ssh certificate "$CINECA_EMAIL" --provisioner cineca-hpc "$CERT_KEY"
    if [ $? -ne 0 ]; then
        echo "[!] Certificate renewal failed. Run first:"
        echo "    step ca bootstrap --ca-url=https://sshproxy.hpc.cineca.it --fingerprint 2ae1543202304d3f434bdc1a2c92eff2cd2b02110206ef06317e70c1c1735ecd"
        exit 1
    fi
    ssh-add "$CERT_KEY"
    echo "[+] Certificate renewed and added to SSH agent: $CERT_KEY"
}

open_ssh() {
    echo "[*] Connecting to Leonardo ($LEONARDO_USER@$LEONARDO_HOST)..."
    ssh -i "$CERT_KEY" "$LEONARDO_USER@$LEONARDO_HOST"
}

# Resolve node IP for standalone vLLM tunnel (reads from job .out file)
get_node_ip_from_log() {
    local job_id=$1
    local out_file=$2
    echo "[*] Retrieving node IP for job $job_id..." >&2
    local ip
    ip=$(ssh -i "$CERT_KEY" "$LEONARDO_USER@$LEONARDO_HOST" \
        "grep 'Infiniband address' ~/${out_file}-${job_id}.out 2>/dev/null | awk '{print \$NF}'" 2>/dev/null)
    echo "$ip"
}

# Detect a running Vectoria job and return "job_id node"
find_vectoria_job() {
    local job_id="${1:-}"
    local filter=""
    [[ -n "$job_id" ]] && filter="--job $job_id"

    raw=$(ssh -i "$CERT_KEY" "$LEONARDO_USER@$LEONARDO_HOST" \
        "squeue -u $LEONARDO_USER -n $VECTORIA_JOB_NAME $filter -h -o '%i %T %N' 2>/dev/null")

    if [[ -z "$raw" ]]; then
        if [[ -n "$job_id" ]]; then
            echo "[!] Job $job_id not found or does not belong to '$VECTORIA_JOB_NAME'." >&2
        else
            echo "[!] No '$VECTORIA_JOB_NAME' job found for user $LEONARDO_USER." >&2
            echo "    Submit it first with: sbatch vectoria_hpc_launcher.job" >&2
        fi
        return 1
    fi

    # Prefer RUNNING; fall back to the first available entry
    selected=$(echo "$raw" | awk '$2 == "RUNNING"' | head -1)
    [[ -z "$selected" ]] && selected=$(echo "$raw" | head -1)

    local id state node
    id=$(echo "$selected" | awk '{print $1}')
    state=$(echo "$selected" | awk '{print $2}')
    node=$(echo "$selected" | awk '{print $3}')

    if [[ "$state" != "RUNNING" ]]; then
        echo "[!] Job $id found but in state '$state' (not yet RUNNING)." >&2
        echo "    Wait for the job to start and try again." >&2
        return 1
    fi

    echo "$id $node"
}

open_vectoria_tunnel() {
    local arg="${1:-}"

    echo "[*] Looking for Vectoria job on Leonardo..."
    local result
    result=$(find_vectoria_job "$arg") || exit 1

    local job_id node
    job_id=$(echo "$result" | awk '{print $1}')
    node=$(echo "$result" | awk '{print $2}')

    echo "[+] Job $job_id RUNNING on node: $node"
    echo ""
    echo "[*] Opening tunnels:"
    echo "    localhost:${VECTORIA_LOCAL_PORT}  → ${node}:${VECTORIA_REMOTE_PORT}  (Vectoria API)"
    echo "    localhost:${OPENWEBUI_LOCAL_PORT}  → ${node}:${OPENWEBUI_REMOTE_PORT} (OpenWebUI)"
    echo ""
    echo "    Vectoria API: http://localhost:${VECTORIA_LOCAL_PORT}"
    echo "    OpenWebUI:    http://localhost:${OPENWEBUI_LOCAL_PORT}"
    echo ""
    echo "[*] Press Ctrl+C to close the tunnels."

    ssh -i "$CERT_KEY" \
        -L "0.0.0.0:${VECTORIA_LOCAL_PORT}:${node}:${VECTORIA_REMOTE_PORT}" \
        -L "0.0.0.0:${OPENWEBUI_LOCAL_PORT}:${node}:${OPENWEBUI_REMOTE_PORT}" \
        -o "ServerAliveInterval=30" \
        -o "ServerAliveCountMax=3" \
        -o "ExitOnForwardFailure=yes" \
        "$LEONARDO_USER@$LEONARDO_HOST" -N
}

open_vllm_tunnel() {
    local target=$1
    local local_port=8001
    local remote_port=8000

    if [[ "$target" =~ ^[0-9]+$ ]] && [[ ${#target} -gt 6 ]]; then
        local node_ip
        node_ip=$(get_node_ip_from_log "$target" "llm_launcher")
        if [ -z "$node_ip" ]; then
            echo "[!] IP not found for job $target."
            echo "    Check that the job has started: tail -f ~/llm_launcher-${target}.out"
            exit 1
        fi
        echo "[+] Node IP: $node_ip"
    else
        local node_ip=$target
    fi

    echo "[*] Opening tunnel: localhost:$local_port → $node_ip:$remote_port (via $LEONARDO_HOST)"
    echo "[*] Press Ctrl+C to close."
    ssh -i "$CERT_KEY" -L "0.0.0.0:${local_port}:${node_ip}:${remote_port}" \
        "$LEONARDO_USER@$LEONARDO_HOST" -N
}

# Resolves $1 to an absolute path and refuses to use it if it's $HOME, contains
# a .git directory, or has content that doesn't look like a wiki mirror ---
# these are the directories a wiki-pull or wiki-push rsync --delete could wipe
# if pointed at the wrong place. Echoes the resolved absolute path to stdout on
# success; prints an error to stderr and returns 1 on refusal. $2/$3 are the
# tenant and subcommand name, used only in the printed hints.
_wiki_dir_guard() {
    local dir="$1" tenant="$2" subcommand="$3"
    local resolved="$dir"
    [[ "$resolved" != /* ]] && resolved="$(pwd)/$resolved"

    if [[ "$resolved" == "$HOME" ]]; then
        echo "[!] Refusing to use your home directory ($resolved)." >&2
        echo "    Pass a dedicated local_dir, e.g.: $0 $subcommand $tenant data/wiki_sync/$tenant" >&2
        return 1
    fi

    if [[ -d "$resolved/.git" ]]; then
        echo "[!] Refusing to use a git repository root ($resolved)." >&2
        echo "    rsync --delete would erase everything in it that isn't part of the wiki mirror." >&2
        echo "    Pass a dedicated local_dir instead, e.g.: $0 $subcommand $tenant data/wiki_sync/$tenant" >&2
        return 1
    fi

    if [[ -d "$resolved" ]] && [[ -n "$(ls -A "$resolved" 2>/dev/null)" ]]; then
        if [[ ! -e "$resolved/page_sources.json" && ! -e "$resolved/pending_updates.json" && ! -d "$resolved/pages" ]]; then
            echo "[!] $resolved already has content that doesn't look like a wiki mirror" >&2
            echo "    (expected page_sources.json / pending_updates.json / pages/ from a previous sync)." >&2
            echo "    Refusing to use it -- rsync --delete would erase its current content." >&2
            echo "    Pass an empty or dedicated local_dir, e.g.: $0 $subcommand $tenant data/wiki_sync/$tenant" >&2
            return 1
        fi
    fi

    echo "$resolved"
}

wiki_pull() {
    local tenant="${1:-}"
    local local_dir="${2:-data/wiki_sync/${tenant}}"

    if [[ -z "$tenant" ]]; then
        echo "[!] Usage: $0 wiki-pull <tenant> [local_dir]"
        exit 1
    fi

    local_dir=$(_wiki_dir_guard "$local_dir" "$tenant" "wiki-pull") || exit 1

    echo "[*] Resolving \$SCRATCH on $LEONARDO_HOST..."
    local remote_scratch
    remote_scratch=$(ssh -i "$CERT_KEY" "$LEONARDO_USER@$LEONARDO_HOST" 'echo $SCRATCH')
    if [[ -z "$remote_scratch" ]]; then
        echo "[!] Could not resolve \$SCRATCH on $LEONARDO_HOST. Check your SSH connection/certificate."
        exit 1
    fi

    local remote_wiki_dir="${remote_scratch}/vectoria_data/index/${tenant}/wiki/"
    mkdir -p "$local_dir"

    echo "[+] Pulling wiki for tenant '$tenant':"
    echo "    $LEONARDO_USER@$LEONARDO_HOST:$remote_wiki_dir"
    echo "    → $local_dir"
    echo ""

    local rsync_output
    if rsync_output=$(rsync -az --delete -v \
        -e "ssh -i $CERT_KEY" \
        "$LEONARDO_USER@$LEONARDO_HOST:$remote_wiki_dir" \
        "$local_dir/" 2>&1); then
        local n_files
        n_files=$(echo "$rsync_output" | grep -c '\.md$')
        echo "[+] Synced ($n_files .md file(s) touched)"
    else
        echo "[!] rsync failed (wiki may not exist yet, or connection issue):"
        echo "$rsync_output" | sed 's/^/    /'
        exit 1
    fi
}

wiki_push() {
    local tenant="${1:-}"
    local local_dir="${2:-data/wiki_sync/${tenant}}"

    if [[ -z "$tenant" ]]; then
        echo "[!] Usage: $0 wiki-push <tenant> [local_dir]"
        exit 1
    fi

    local_dir=$(_wiki_dir_guard "$local_dir" "$tenant" "wiki-push") || exit 1

    if [[ ! -d "$local_dir/pages" ]]; then
        echo "[!] $local_dir does not look like a wiki mirror (no pages/ directory) -- nothing to push."
        exit 1
    fi

    echo "[*] Resolving \$SCRATCH on $LEONARDO_HOST..."
    local remote_scratch
    remote_scratch=$(ssh -i "$CERT_KEY" "$LEONARDO_USER@$LEONARDO_HOST" 'echo $SCRATCH')
    if [[ -z "$remote_scratch" ]]; then
        echo "[!] Could not resolve \$SCRATCH on $LEONARDO_HOST. Check your SSH connection/certificate."
        exit 1
    fi

    local remote_wiki_dir="${remote_scratch}/vectoria_data/index/${tenant}/wiki/"

    echo "[*] Checking for recent remote activity..."
    local recent
    recent=$(ssh -i "$CERT_KEY" "$LEONARDO_USER@$LEONARDO_HOST" \
        "find '$remote_wiki_dir' -newermt '-2 minutes' 2>/dev/null")
    if [[ -n "$recent" ]]; then
        echo "[!] Files modified on the remote in the last 2 minutes -- Vectoria may be actively"
        echo "    synthesizing the wiki right now, or pages exist there that you haven't pulled yet:"
        echo "$recent" | sed 's/^/    /'
    fi

    echo ""
    echo "[!] This will overwrite the remote wiki for tenant '$tenant':"
    echo "    $local_dir/"
    echo "    → $LEONARDO_USER@$LEONARDO_HOST:$remote_wiki_dir"
    echo "    Pages that exist remotely but not in $local_dir will be DELETED."
    echo ""
    read -r -p "Type 'yes' to continue: " confirm
    if [[ "$confirm" != "yes" ]]; then
        echo "[*] Aborted, nothing pushed."
        exit 1
    fi

    echo "[*] Pushing to $LEONARDO_HOST..."
    local rsync_output
    if rsync_output=$(rsync -az --delete -v \
        -e "ssh -i $CERT_KEY" \
        "$local_dir/" \
        "$LEONARDO_USER@$LEONARDO_HOST:$remote_wiki_dir" 2>&1); then
        echo "[+] Push complete."
    else
        echo "[!] rsync push failed:"
        echo "$rsync_output" | sed 's/^/    /'
        exit 1
    fi
}

# --- Main ---
case "${1:-}" in
    cert)
        renew_cert
        ;;
    ssh)
        open_ssh
        ;;
    tunnel)
        [ -z "${2:-}" ] && usage
        open_vllm_tunnel "$2"
        ;;
    vectoria)
        open_vectoria_tunnel "${2:-}"
        ;;
    wiki-pull)
        [ -z "${2:-}" ] && usage
        wiki_pull "${2}" "${3:-}"
        ;;
    wiki-push)
        [ -z "${2:-}" ] && usage
        wiki_push "${2}" "${3:-}"
        ;;
    *)
        usage
        ;;
esac
