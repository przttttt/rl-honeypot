#!/usr/bin/env bash
set -euo pipefail

report=".markdownlint-report.md"

if npx --yes markdownlint-cli@0.45.0 --ignore "$report" "**/*.md" >"$report" 2>&1; then
  if [[ ! -s "$report" ]]; then
    printf 'Markdown lint passed.\n' >"$report"
  fi
else
  status=$?
  cat "$report" >&2
  exit "$status"
fi
