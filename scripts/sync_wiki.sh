#!/usr/bin/env bash
# Syncs local docs/wiki/ documentation to the standalone GitHub Wiki git repository.
set -euo pipefail

REPO_OWNER="Kamonashis"
REPO_NAME="catboost-webgpu"
WIKI_REMOTE="https://github.com/${REPO_OWNER}/${REPO_NAME}.wiki.git"
TEMP_DIR=$(mktemp -d /tmp/catboost-wiki-sync.XXXXXX)

echo "==> Synchronizing docs/wiki/ to ${WIKI_REMOTE}..."

# Check if wiki remote exists / has been initialized on GitHub
if ! git ls-remote "${WIKI_REMOTE}" &>/dev/null; then
    echo "Notice: GitHub Wiki repository (${WIKI_REMOTE}) has not been initialized yet."
    echo "To initialize it: Visit https://github.com/${REPO_OWNER}/${REPO_NAME}/wiki and click 'Create the first page'."
    echo "All wiki files are safely committed in the main repository under docs/wiki/."
    exit 0
fi

# Clone wiki repo
git clone "${WIKI_REMOTE}" "${TEMP_DIR}"
cp docs/wiki/*.md "${TEMP_DIR}/"

cd "${TEMP_DIR}"
git add .
if git diff --staged --quiet; then
    echo "==> Wiki is already up-to-date with docs/wiki/."
else
    git commit -m "docs: sync wiki documentation from main repository"
    git push origin master || git push origin main
    echo "==> Successfully pushed wiki updates to GitHub Wiki!"
fi

rm -rf "${TEMP_DIR}"
