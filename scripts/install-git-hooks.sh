#!/usr/bin/env bash
# Point this repository at .githooks/ so hooks run for all contributors who run this once.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

git config core.hooksPath .githooks
printf 'Set core.hooksPath to .githooks (repo root: %s)\n' "$root"
