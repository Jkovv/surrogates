#!/bin/bash
# Submit 500x500 jobs to Snellius via SSH.
# Credentials are read from .env — the password is NEVER printed to the terminal.
#
# Usage:
#   bash scripts/submit_snellius.sh [SCAN_ITER]
#   SCAN_ITER defaults to 0
#
# Requires: sshpass   (apt install sshpass)
# .env must define: SNELLIUS_USER, SNELLIUS_HOST, SNELLIUS_PASS
#   Optional:        SNELLIUS_PROJECT_DIR  (default: $HOME/codes/students/surrogates)

set -euo pipefail

SCAN_ITER=${1:-0}

# ── Load .env ──────────────────────────────────────────────────────────────
ENV_FILE="$(dirname "$0")/../.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env not found at $ENV_FILE"
    echo "       Copy .env.example to .env and fill in your credentials."
    exit 1
fi

# Source without printing — 'set -a' auto-exports all variables
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# ── Validate credentials ───────────────────────────────────────────────────
for VAR in SNELLIUS_USER SNELLIUS_HOST SNELLIUS_PASS; do
    if [ -z "${!VAR:-}" ]; then
        echo "ERROR: $VAR is not set in .env"
        exit 1
    fi
done

REMOTE_DIR="${SNELLIUS_PROJECT_DIR:-\$HOME/codes/students/surrogates}"

# ── Set SSHPASS from credential — sshpass reads this env var internally ───
# This avoids the password ever appearing in the process list or shell output.
export SSHPASS="$SNELLIUS_PASS"
unset SNELLIUS_PASS   # remove original copy

SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=no"
REMOTE="${SNELLIUS_USER}@${SNELLIUS_HOST}"

_ssh() { sshpass -e ssh $SSH_OPTS "$REMOTE" "$@"; }
_scp() { sshpass -e scp $SSH_OPTS "$@"; }
_rsync() { sshpass -e rsync -az --no-perms -e "ssh $SSH_OPTS" "$@"; }

echo "Connecting to $SNELLIUS_USER@$SNELLIUS_HOST ..."
echo "(password read from .env — not displayed)"

# ── Verify sshpass is available ────────────────────────────────────────────
if ! command -v sshpass &>/dev/null; then
    echo "ERROR: sshpass not installed. Run: sudo apt install sshpass"
    exit 1
fi

# ── Sync code to Snellius (exclude data, venvs, __pycache__, .git) ─────────
echo ""
echo "=== Syncing code to Snellius ==="
_rsync \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='data/' \
    --exclude='models/' \
    --exclude='*.npy' \
    --exclude='*.h5' \
    --exclude='jobs/logs/' \
    ./ "${REMOTE}:${REMOTE_DIR}/"
echo "Code synced."

# ── Optionally sync preprocessed data for the requested scan iteration ─────
PREPROC_LOCAL="data/ICCS2026_v2/scan_iteration_${SCAN_ITER}/preprocessed/500x500"
PREPROC_REMOTE="${REMOTE_DIR}/data/ICCS2026_v2/scan_iteration_${SCAN_ITER}/preprocessed/500x500"

if [ -d "$PREPROC_LOCAL" ]; then
    echo ""
    echo "=== Syncing preprocessed data for scan_iteration_${SCAN_ITER} ==="
    echo "    This may take a while (up to ~13 GB) ..."
    _ssh "mkdir -p ${PREPROC_REMOTE}"
    _rsync "$PREPROC_LOCAL/" "${REMOTE}:${PREPROC_REMOTE}/"
    echo "Data synced."
else
    echo "WARNING: $PREPROC_LOCAL not found locally — skipping data sync."
    echo "         Make sure the data exists on Snellius before jobs run."
fi

# ── Create required directories on Snellius ────────────────────────────────
echo ""
echo "=== Preparing remote directories ==="
_ssh "cd ${REMOTE_DIR} && mkdir -p jobs/logs models/deeponet_h models/pinn models/pi_deeponet models/sta_lstm models/unet"

# ── Submit jobs ────────────────────────────────────────────────────────────
echo ""
echo "=== Submitting jobs (scan_iteration_${SCAN_ITER}) ==="
_ssh "cd ${REMOTE_DIR} && bash jobs/submit_all_500.sh ${SCAN_ITER}"

echo ""
echo "=== Done. Monitor jobs with: ==="
echo "  sshpass -e ssh $SSH_OPTS ${REMOTE} 'squeue -u ${SNELLIUS_USER}'"
