#!/usr/bin/env bash
# Mount loop-spec's ADK agent directories for `adk run` / `adk web`.
#
# The Python app imports loop-spec's extension directly (src/devteam/loopspec.py);
# this mount is only for driving the loop-spec agents from the ADK CLI:
#
#   uv run adk run adk_agents/loop_spec
#
# The generated shims reference the submodule checkout by absolute path, so they
# are machine-local and gitignored; rerun this script after moving the repo.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
git -C "$REPO_ROOT" submodule update --init third_party/loop-spec
# `uv run` puts the project venv's python3 (which has google-adk) on PATH for
# the installer's version check.
exec uv run --project "$REPO_ROOT" bash "$REPO_ROOT/third_party/loop-spec/lib/adk-install.sh" install \
  --project "$REPO_ROOT" --model "${LOOP_SPEC_ADK_MODEL:-gemini-2.5-pro}" "$@"
