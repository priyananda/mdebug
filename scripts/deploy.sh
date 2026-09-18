#!/usr/bin/env bash
#
# Build the mdebug server image with Cloud Build and deploy it to Cloud Run.
#
#   MDEBUG_PROJECT_ID=my-project scripts/deploy.sh setup    # once per project
#   MDEBUG_PROJECT_ID=my-project scripts/deploy.sh          # every deploy
#
# Docker is not required: the build happens in Cloud Build from an uploaded
# source context.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROJECT_ID="${MDEBUG_PROJECT_ID:?set MDEBUG_PROJECT_ID to your GCP project id}"
REGION="${MDEBUG_REGION:-us-central1}"
REPO="mdebug"
SERVICE="mdebug-server"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/server"

# The client is served from GitHub Pages, which is HTTPS-only. This REPLACES
# the default list in app/main.py rather than extending it, so it has to be
# complete -- localhost stays so `ng serve` can run against the deployed API.
CORS_ORIGINS="https://priyananda.github.io,http://localhost:4200"

WEIGHTS="${REPO_ROOT}/server/checkpoints/ckpt_5000.weights.pt"

# strip_checkpoint.py needs torch, which lives in the server venv rather than
# on the system interpreter.
PYTHON="${REPO_ROOT}/server/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"


setup() {
  gcloud auth list --filter=status:ACTIVE --format='value(account)' | grep -q . \
    || gcloud auth login

  # Check the project before touching anything. `gcloud config set project`
  # exits 0 even when you decline its "are you sure" prompt, so without this a
  # typo'd id gets past it and fails several commands later instead.
  gcloud projects describe "$PROJECT_ID" --format='value(projectId)' >/dev/null 2>&1 || {
    echo "error: cannot access project '$PROJECT_ID'. Visible projects:" >&2
    gcloud projects list --format='value(projectId)' >&2
    exit 1
  }

  # Cloud Build and Cloud Run both refuse without billing.
  gcloud billing projects describe "$PROJECT_ID" --format='value(billingEnabled)' 2>/dev/null \
    | grep -qi true \
    || echo "warning: could not confirm billing is enabled on $PROJECT_ID" >&2

  gcloud config set project "$PROJECT_ID"

  gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    --project="$PROJECT_ID"

  # Idempotent: re-running setup should be harmless.
  gcloud artifacts repositories describe "$REPO" \
      --location="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1 \
    || gcloud artifacts repositories create "$REPO" \
        --repository-format=docker \
        --location="$REGION" \
        --description="mdebug inference server images" \
        --project="$PROJECT_ID"

  # If `builds submit` fails with a logging error before running a single step,
  # it is the Compute Engine default service account missing log-writer on a
  # fresh project. Either grant it:
  #
  #   PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
  #   gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  #     --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  #     --role="roles/logging.logWriter"
  #
  # or add --default-buckets-behavior=regional-user-owned-bucket to the build.

  echo "Setup complete. Now run: MDEBUG_PROJECT_ID=$PROJECT_ID $0"
}


deploy() {
  "$PYTHON" "${REPO_ROOT}/scripts/strip_checkpoint.py"

  test -s "$WEIGHTS" \
    || { echo "error: $WEIGHTS is missing or empty" >&2; exit 1; }

  # Without this file gcloud uploads the whole 2.0 GB tree, and it only reports
  # the size after the upload has finished. Fail fast instead.
  test -f "${REPO_ROOT}/server/.gcloudignore" \
    || { echo "error: server/.gcloudignore is missing; refusing to upload 2 GB" >&2; exit 1; }

  # Never :latest -- a unique tag keeps revisions traceable to a commit and
  # guarantees Cloud Run rolls a genuinely new one.
  local tag
  tag="$(git -C "$REPO_ROOT" rev-parse --short HEAD)-$(date -u +%Y%m%d%H%M%S)"

  echo "Building ${IMAGE}:${tag} (expect a ~343 MB upload, not GB)"
  gcloud builds submit "${REPO_ROOT}/server" \
    --tag "${IMAGE}:${tag}" \
    --timeout=1800s \
    --project="$PROJECT_ID"

  # Sessions are in-process state (see app/registry.py), so min=max=1 is the
  # only correct topology: there is no second instance that could answer for a
  # session it never created. --timeout is Cloud Run's 60 min ceiling and
  # applies to the WebSocket as a hard wall-clock cap; the client's 20s ping
  # does not extend it.
  gcloud run deploy "$SERVICE" \
    --image="${IMAGE}:${tag}" \
    --region="$REGION" \
    --platform=managed \
    --execution-environment=gen2 \
    --allow-unauthenticated \
    --port=8080 \
    --cpu=2 \
    --memory=4Gi \
    --min-instances=1 \
    --max-instances=1 \
    --no-cpu-throttling \
    --cpu-boost \
    --concurrency=32 \
    --timeout=3600 \
    --set-env-vars="^@^MDEBUG_MAX_SESSIONS=4@MDEBUG_CORS_ORIGINS=${CORS_ORIGINS}" \
    --project="$PROJECT_ID"
  # ^@^ picks @ as the delimiter. Without it gcloud splits --set-env-vars on
  # commas and CORS_ORIGINS' own commas create a variable literally named
  # "http://localhost:4200".

  local url
  url="$(gcloud run services describe "$SERVICE" \
           --region="$REGION" --project="$PROJECT_ID" \
           --format='value(status.url)')"

  echo
  echo "Deployed: $url"
  echo "  $url/api/health   -> expect checkpointStep: 5000"
  echo
  echo "Paste into client/src/environments/environment.prod.ts:"
  echo "    useMock: false,"
  echo "    apiBase: '${url}',"
}


case "${1:-deploy}" in
  setup)  setup ;;
  deploy) deploy ;;
  *)      echo "usage: $0 [setup|deploy]" >&2; exit 2 ;;
esac
