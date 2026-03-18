#!/bin/bash
# sync_from_hpc.sh — Pull all results from Snellius to local machine.
#
# Usage:
#   ./hpc/sync_from_hpc.sh            # sync everything
#   ./hpc/sync_from_hpc.sh fits       # sync results/fits/ only
#   ./hpc/sync_from_hpc.sh figures    # sync results/figures/ only
#   ./hpc/sync_from_hpc.sh sensitivity # sync results/sensitivity/ only
#
# Run this BEFORE analysing any HPC output.  Scratch space on Snellius
# is ephemeral; pull results immediately after each job completes.

set -euo pipefail
source "$(dirname "$0")/../.env"

REMOTE="${SNELLIUS_USER}@snellius.surf.nl:${SNELLIUS_DIR}"
LOCAL="$(dirname "$0")/.."

# Directories to sync (add more as needed)
TARGETS=(fits figures sensitivity tables)

# If a specific target is requested, use only that one
if [[ $# -gt 0 ]]; then
    TARGETS=("$@")
fi

echo "=== Syncing results from Snellius → local ==="
echo "  Remote : ${REMOTE}/results/"
echo "  Local  : ${LOCAL}/results/"
echo "  Dirs   : ${TARGETS[*]}"
echo ""

RSYNC_OPTS="-avz --progress --human-readable"
if command -v sshpass &>/dev/null && [[ -n "${SNELLIUS_PASSWORD:-}" ]]; then
    RSYNC_CMD="sshpass -p ${SNELLIUS_PASSWORD} rsync ${RSYNC_OPTS}"
else
    # Fall back to key-based auth
    RSYNC_CMD="rsync ${RSYNC_OPTS}"
fi

for dir in "${TARGETS[@]}"; do
    echo "--- Syncing results/${dir}/ ---"
    mkdir -p "${LOCAL}/results/${dir}"
    ${RSYNC_CMD} \
        "${REMOTE}/results/${dir}/" \
        "${LOCAL}/results/${dir}/" \
        --exclude="*.tmp" \
        2>&1 || echo "  [WARN] ${dir} sync failed or directory missing on remote"
    echo ""
done

# Always sync top-level results files (e.g. identifiable_params.json)
echo "--- Syncing results/*.json ---"
${RSYNC_CMD} \
    --include="*.json" --include="*.csv" --exclude="*" \
    "${REMOTE}/results/" "${LOCAL}/results/" \
    2>&1 || echo "  [WARN] top-level results sync failed"
echo ""

echo "=== Sync complete: $(date) ==="
