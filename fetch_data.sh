#!/usr/bin/env bash
# Download the SMPL neutral body model needed by HybrIK.
#
# The file is licensed and cannot be redistributed, so this script fetches it
# from the official SMPLify download page. Set SMPLIFY_USERNAME and
# SMPLIFY_PASSWORD (register at https://smplify.is.tue.mpg.de/) and optionally
# HYBRIK_MODEL_DIR to choose where the file is written.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${HYBRIK_MODEL_DIR:-${ROOT}/model_files}"
ARCHIVE="${ROOT}/mpips_smplify_public_v2.zip"
TMP_DIR="$(mktemp -d)"

cleanup() {
    rm -rf "${TMP_DIR}" "${ARCHIVE}"
}
trap cleanup EXIT

if [[ -z "${SMPLIFY_USERNAME:-}" || -z "${SMPLIFY_PASSWORD:-}" ]]; then
    cat >&2 <<'EOF'
Set the SMPLify account credentials first, for example:

    export SMPLIFY_USERNAME='you@example.com'
    export SMPLIFY_PASSWORD='...'
    ./fetch_data.sh

Register at https://smplify.is.tue.mpg.de/ if you do not have an account.
EOF
    exit 2
fi

mkdir -p "${MODEL_DIR}"

curl --fail --location --retry 3 \
    --data-urlencode "username=${SMPLIFY_USERNAME}" \
    --data-urlencode "password=${SMPLIFY_PASSWORD}" \
    'https://download.is.tue.mpg.de/download.php?domain=smplify&sfile=mpips_smplify_public_v2.zip&resume=1' \
    --output "${ARCHIVE}"

unzip -q "${ARCHIVE}" -d "${TMP_DIR}"

MODEL_FILE="$(find "${TMP_DIR}" -name 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl' -print -quit)"
if [[ -z "${MODEL_FILE}" ]]; then
    echo "Could not find basicModel_neutral_lbs_10_207_0_v1.0.0.pkl in the archive." >&2
    exit 1
fi

install -m 0644 "${MODEL_FILE}" "${MODEL_DIR}/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
echo "Installed SMPL body model in ${MODEL_DIR}"
