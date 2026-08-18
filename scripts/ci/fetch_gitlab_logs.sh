#!/usr/bin/env bash
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE.APACHE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fetch failed-job traces and artifacts from the mirrored GitLab pipeline and
# surface them in the GitHub Actions console. Intended to run in a GitHub
# workflow step gated on `if: failure()` after the pulp-actions gitlab-ci
# check. Always exits 0 — the original gitlab-ci step is the one that fails
# the workflow; this script is purely informational.
#
# Required environment:
#   GITLAB_TOKEN   Token with read_api scope (same token used by the mirror).
#   GITLAB_DOMAIN  e.g. iis-git.ee.ethz.ch
#   GITLAB_REPO    e.g. github-mirror/magia-sdk-mirror
#   COMMIT_SHA     GitHub commit SHA mirrored to GitLab.
#   OUT_DIR        Output directory (created if missing).

set -u
set -o pipefail

: "${GITLAB_TOKEN:?GITLAB_TOKEN is required}"
: "${GITLAB_DOMAIN:?GITLAB_DOMAIN is required}"
: "${GITLAB_REPO:?GITLAB_REPO is required}"
: "${COMMIT_SHA:?COMMIT_SHA is required}"
: "${OUT_DIR:=gitlab-logs}"

mkdir -p "$OUT_DIR"

API="https://${GITLAB_DOMAIN}/api/v4"
PROJECT_ID="${GITLAB_REPO//\//%2F}"
CURL=(curl -fsSL --retry 3 --retry-delay 5 -H "PRIVATE-TOKEN: ${GITLAB_TOKEN}")

sanitize() {
  # Replace characters that are awkward in filenames.
  echo "$1" | tr '/ :' '___'
}

# --- 1. Find the newest pipeline for this commit ---
PIPELINE_JSON=""
for attempt in 1 2 3 4 5; do
  if PIPELINE_JSON=$("${CURL[@]}" \
      "${API}/projects/${PROJECT_ID}/pipelines?sha=${COMMIT_SHA}&per_page=1" 2>/dev/null) \
     && [ "$(echo "$PIPELINE_JSON" | jq 'length')" -gt 0 ]; then
    break
  fi
  echo "::warning::No GitLab pipeline found for ${COMMIT_SHA} yet (attempt ${attempt}/5), retrying in 10s…" >&2
  PIPELINE_JSON=""
  sleep 10
done

if [ -z "$PIPELINE_JSON" ] || [ "$(echo "$PIPELINE_JSON" | jq 'length')" -eq 0 ]; then
  echo "::warning::Gave up looking for a GitLab pipeline matching ${COMMIT_SHA}. Not fetching logs."
  exit 0
fi

PIPELINE_ID=$(echo "$PIPELINE_JSON" | jq -r '.[0].id')
PIPELINE_URL=$(echo "$PIPELINE_JSON" | jq -r '.[0].web_url')
echo "Inspecting GitLab pipeline ${PIPELINE_ID}: ${PIPELINE_URL}"

# --- 2. Paginate through jobs ---
JOBS_JSON="[]"
page=1
while :; do
  PAGE_JSON=$("${CURL[@]}" \
    "${API}/projects/${PROJECT_ID}/pipelines/${PIPELINE_ID}/jobs?per_page=100&page=${page}") || {
    echo "::warning::Failed to fetch jobs page ${page}; stopping pagination." >&2
    break
  }
  count=$(echo "$PAGE_JSON" | jq 'length')
  [ "$count" -eq 0 ] && break
  JOBS_JSON=$(jq -s '.[0] + .[1]' <(echo "$JOBS_JSON") <(echo "$PAGE_JSON"))
  [ "$count" -lt 100 ] && break
  page=$((page + 1))
done

# --- 3. For each failed job: trace + artifacts ---
FAILED_COUNT=0
while IFS=$'\t' read -r job_id job_name job_stage job_url; do
  [ -z "$job_id" ] && continue
  FAILED_COUNT=$((FAILED_COUNT + 1))
  slug="$(sanitize "$job_stage")__$(sanitize "$job_name")"
  trace_path="${OUT_DIR}/${slug}.trace"
  art_dir="${OUT_DIR}/${slug}"

  # Trace
  if ! "${CURL[@]}" "${API}/projects/${PROJECT_ID}/jobs/${job_id}/trace" \
       -o "$trace_path" 2>/dev/null; then
    echo "::warning::Could not fetch trace for job ${job_id} (${job_name})" >&2
    echo "(trace unavailable)" > "$trace_path"
  fi

  # Artifacts (may not exist — 404 is fine)
  art_zip="${OUT_DIR}/${slug}.artifacts.zip"
  http_code=$(curl -sSL -o "$art_zip" -w '%{http_code}' \
    --retry 3 --retry-delay 5 \
    -H "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
    "${API}/projects/${PROJECT_ID}/jobs/${job_id}/artifacts" || echo "000")
  if [ "$http_code" = "200" ]; then
    mkdir -p "$art_dir"
    if ! unzip -o -q "$art_zip" -d "$art_dir"; then
      echo "::warning::Failed to unzip artifacts for job ${job_id}" >&2
    fi
  fi
  rm -f "$art_zip"

  echo "Trace: ${trace_path} (job=${job_name}, stage=${job_stage})"
done < <(echo "$JOBS_JSON" | jq -r '.[] | select(.status == "failed") | [.id, .name, .stage, .web_url] | @tsv')

echo "Fetched ${FAILED_COUNT} failed job(s) from pipeline ${PIPELINE_URL}"
exit 0
