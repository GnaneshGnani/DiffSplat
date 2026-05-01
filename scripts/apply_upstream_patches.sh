#!/usr/bin/env bash
# Apply local project patches to the ignored official DiffSplat checkout.

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UPSTREAM="${ROOT}/DiffSplat"
PATCH="${ROOT}/patches/diffsplat_hpc_scratch.patch"

if [[ ! -d "${UPSTREAM}/.git" ]]; then
  echo "Official DiffSplat checkout not found at ${UPSTREAM}" >&2
  exit 1
fi

if [[ ! -f "${PATCH}" ]]; then
  echo "Patch file not found: ${PATCH}" >&2
  exit 1
fi

if git -C "${UPSTREAM}" apply --check "${PATCH}" >/dev/null 2>&1; then
  git -C "${UPSTREAM}" apply "${PATCH}"
  echo "Applied ${PATCH}"
elif git -C "${UPSTREAM}" apply --reverse --check "${PATCH}" >/dev/null 2>&1; then
  echo "Patch already applied: ${PATCH}"
else
  echo "Patch cannot be applied cleanly. Inspect ${UPSTREAM} manually." >&2
  exit 1
fi
