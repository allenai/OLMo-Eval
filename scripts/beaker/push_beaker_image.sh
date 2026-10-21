#!/usr/bin/env bash
set -euo pipefail

# Push Docker image to Beaker with versioning
#
# This script uploads a Docker image to Beaker, implementing safe version
# management by archiving the previous image before replacing it.
#
# Usage:
#   ./scripts/beaker/push_beaker_image.sh                              # Use defaults
#   ./scripts/beaker/push_beaker_image.sh --source olmo-eval:cu1281-trc2100-amd64
#   ./scripts/beaker/push_beaker_image.sh --workspace ai2/oe-data      # Custom workspace
#   ./scripts/beaker/push_beaker_image.sh --dry-run                    # Preview only
#   ./scripts/beaker/push_beaker_image.sh --force                      # Force re-upload (delete existing tmp)
#
# The script will:
#   1. Upload the source image as a temporary image (-tmp)
#   2. Rename the current Beaker image with a timestamp suffix
#   3. Rename the temporary image to the final name
#
# This ensures safe rollback capability if issues are discovered.

# Defaults matching beaker.py
SOURCE_IMAGE=""
BEAKER_IMAGE=""
WORKSPACE="ai2/oe-data"
DRY_RUN=false
AUTO_NAME=true
FORCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --source)
            SOURCE_IMAGE="$2"
            shift 2
            ;;
        --beaker-image)
            BEAKER_IMAGE="$2"
            AUTO_NAME=false
            shift 2
            ;;
        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --source IMAGE        Local Docker image (default: olmo-eval:latest)"
            echo "  --beaker-image NAME   Beaker image name (default: auto-generated from source tag)"
            echo "  --workspace WS        Beaker workspace (default: ai2/oe-data)"
            echo "  --dry-run             Preview without pushing"
            echo "  --force               Force upload even if tmp image exists (deletes existing tmp)"
            echo "  --help                Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# Auto-detect source image if not specified (use most recent build)
if [[ -z "$SOURCE_IMAGE" ]]; then
    SOURCE_IMAGE=$(docker images olmo-eval --format "{{.Repository}}:{{.Tag}}" | head -n 1)
    if [[ -z "$SOURCE_IMAGE" ]]; then
        echo "Error: No olmo-eval images found. Build an image first."
        exit 1
    fi
    echo "Auto-detected source image: ${SOURCE_IMAGE}"
fi

# Auto-generate Beaker image name from source tag
if [[ "$AUTO_NAME" == "true" ]]; then
    # Extract tag from source image (e.g., olmo-eval:cu1281-trc2100-amd64 -> cu1281-trc2100-amd64)
    SOURCE_TAG=$(echo "$SOURCE_IMAGE" | cut -d':' -f2)
    BEAKER_IMAGE="olmo-eval-${SOURCE_TAG}"
    echo "Auto-generated Beaker image name: ${BEAKER_IMAGE}"
fi

# Check dependencies
if ! command -v beaker &> /dev/null; then
    echo "Error: 'beaker' CLI not found. Run via 'uv run beaker ...' or activate the project venv (.venv/bin/activate) after 'uv sync --frozen'."
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "Error: 'jq' not found. Install with: brew install jq (macOS) or apt install jq (Linux)"
    exit 1
fi

# Generate timestamp for versioning
TIMESTAMP=$(date -u +"%Y%m%d-%H%M%S")

# Get beaker account for image references
BEAKER_ACCOUNT=$(beaker account whoami --format json | jq -r '.[0].name')
if [[ -z "$BEAKER_ACCOUNT" ]]; then
    echo "Error: Could not determine Beaker account"
    exit 1
fi

# Image names (user-qualified for lookups)
TMP_IMAGE_NAME="${BEAKER_IMAGE}-tmp"
TMP_IMAGE_REF="${BEAKER_ACCOUNT}/${TMP_IMAGE_NAME}"
IMAGE_REF="${BEAKER_ACCOUNT}/${BEAKER_IMAGE}"
ARCHIVE_IMAGE="${BEAKER_IMAGE}-${TIMESTAMP}"

echo "Pushing image to Beaker..."
echo "  Source:      ${SOURCE_IMAGE}"
echo "  Beaker:      ${IMAGE_REF}"
echo "  Workspace:   ${WORKSPACE}"
echo "  Timestamp:   ${TIMESTAMP}"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would execute:"
    if [[ "$FORCE" == "true" ]]; then
        echo "  1. beaker image delete ${TMP_IMAGE_NAME} (if exists, --force)"
        echo "     beaker image create ${SOURCE_IMAGE} --name ${TMP_IMAGE_NAME} --workspace ${WORKSPACE}"
    else
        echo "  1. beaker image create ${SOURCE_IMAGE} --name ${TMP_IMAGE_NAME} --workspace ${WORKSPACE}"
        echo "     (or reuse existing temporary image from previous partial run)"
    fi
    echo "  2. beaker image rename ${IMAGE_REF} ${ARCHIVE_IMAGE} (if exists)"
    echo "  3. beaker image rename ${TMP_IMAGE_REF} ${BEAKER_IMAGE}"
    exit 0
fi

# Step 1: Upload as temporary image (or reuse existing from partial run)
echo "Step 1/3: Uploading image as temporary..."
TMP_IMAGE_JSON=$(beaker image get "${TMP_IMAGE_REF}" --format json 2>/dev/null || echo "")
TMP_IMAGE_ID=$(echo "$TMP_IMAGE_JSON" | jq -r '.[0].id // empty' 2>/dev/null || echo "")
TMP_IMAGE_COMMITTED=$(echo "$TMP_IMAGE_JSON" | jq -r '.[0].committed // empty' 2>/dev/null || echo "")
if [[ -n "$TMP_IMAGE_ID" && "$FORCE" == "true" ]]; then
    echo "  Deleting existing temporary image (--force): ${TMP_IMAGE_ID}"
    beaker image delete "${TMP_IMAGE_ID}"
    TMP_IMAGE_ID=""
elif [[ -n "$TMP_IMAGE_ID" && ( -z "$TMP_IMAGE_COMMITTED" || "$TMP_IMAGE_COMMITTED" == "0001-01-01T00:00:00Z" ) ]]; then
    # An uncommitted image means a previous upload was interrupted mid-push;
    # reusing it would promote a partial image.
    echo "  Existing temporary image is incomplete (upload never committed); deleting: ${TMP_IMAGE_ID}"
    beaker image delete "${TMP_IMAGE_ID}"
    TMP_IMAGE_ID=""
fi
if [[ -n "$TMP_IMAGE_ID" ]]; then
    echo "  Reusing existing temporary image from previous run: ${TMP_IMAGE_ID}"
else
    echo "  Uploading ${SOURCE_IMAGE} (this may take a while for large images)..."

    # Run beaker image create with output visible to user (no stdout redirection)
    if ! beaker image create \
        "${SOURCE_IMAGE}" \
        --name "${TMP_IMAGE_NAME}" \
        --workspace "${WORKSPACE}"; then
        echo "Error: Failed to upload image"
        exit 1
    fi

    # Query for the image ID after successful upload
    TMP_IMAGE_ID=$(beaker image get "${TMP_IMAGE_REF}" --format json | jq -r '.[0].id')
    if [[ -z "$TMP_IMAGE_ID" || "$TMP_IMAGE_ID" == "null" ]]; then
        echo "Error: Could not find uploaded image: ${TMP_IMAGE_REF}"
        exit 1
    fi
    echo "  Created temporary image: ${TMP_IMAGE_ID}"
fi

# Step 2: Archive existing image (if it exists)
echo "Step 2/3: Archiving existing image..."
EXISTING_IMAGE_JSON=$(beaker image get "${IMAGE_REF}" --format json 2>/dev/null || echo "")
EXISTING_IMAGE_ID=$(echo "$EXISTING_IMAGE_JSON" | jq -r '.[0].id // empty' 2>/dev/null || echo "")
EXISTING_IMAGE_COMMITTED=$(echo "$EXISTING_IMAGE_JSON" | jq -r '.[0].committed // empty' 2>/dev/null || echo "")
if [[ -n "$EXISTING_IMAGE_ID" && ( -z "$EXISTING_IMAGE_COMMITTED" || "$EXISTING_IMAGE_COMMITTED" == "0001-01-01T00:00:00Z" ) ]]; then
    echo "  Existing image is incomplete (upload never committed); deleting instead of archiving: ${EXISTING_IMAGE_ID}"
    beaker image delete "${EXISTING_IMAGE_ID}"
elif [[ -n "$EXISTING_IMAGE_ID" ]]; then
    beaker image rename "${EXISTING_IMAGE_ID}" "${ARCHIVE_IMAGE}"
    echo "  Archived to: ${ARCHIVE_IMAGE}"
else
    echo "  No existing image to archive"
fi

# Step 3: Rename temporary to final
echo "Step 3/3: Promoting temporary image..."
beaker image rename "${TMP_IMAGE_ID}" "${BEAKER_IMAGE}"

echo ""
echo "Success! Image available at: ${IMAGE_REF}"
echo ""
echo "To use in Beaker jobs:"
echo "  olmo-eval beaker launch --image ${IMAGE_REF} ..."
