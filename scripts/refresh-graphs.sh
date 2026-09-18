#!/usr/bin/env bash
#
# Refresh the local code/knowledge graphs used for navigation.
#
# Both artifacts are gitignored (.gitnexus/, graphify-out/), so this keeps them
# accurate for the next session — it does not stage anything for a commit.
#
#   gitnexus  — code graph: impact analysis, blast radius, call tracing
#   graphify  — knowledge graph: code (free, AST) + docs (Gemini if configured)
#
# Usage:  ./scripts/refresh-graphs.sh
#
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Repo-level tooling keys (GEMINI_API_KEY) live in the root .env, separate from
# each demo's own runtime .env (GROQ_API_KEY).
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

echo "→ gitnexus: re-indexing repo ..."
# --skip-agents-md / --skip-skills protect the hand-written CLAUDE.md and AGENTS.md
gitnexus analyze --skip-agents-md --skip-skills | tail -4

echo
echo "→ graphify: re-extracting code (no LLM, no API cost) ..."
graphify update . | tail -4

echo
echo "Refresh complete."
echo "If docs/PDFs changed too, also run a full graphify rebuild from your AI assistant."
