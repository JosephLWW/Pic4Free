#!/bin/bash
# LoRA weight audit (CPU): learned or remain ~zero of initialization?
#SBATCH --job-name=Pic4Free_checklora
#SBATCH --output=slurm_logs/slurm_%j.out
#SBATCH --error=slurm_logs/slurm_%j.err
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=0:15:00

set -Eeuo pipefail

unset PYTHONPATH
unset VIRTUAL_ENV
hash -r

module purge
module load devel/python/3.12.3-gnu-14.2

WORKDIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not defined}"
WORKDIR="$(cd "${WORKDIR}" && pwd)"
cd "${WORKDIR}"

CACHE_ROOT="${PIC4FREE_CACHE_ROOT:-${SCRATCH:-${TMPDIR}}}"
CACHE_ROOT="${CACHE_ROOT}/pic4free/${USER}"
mkdir -p "${CACHE_ROOT}"

VENV="${CACHE_ROOT}/venv"
PYTHON_BIN="${VENV}/bin/python"
REQ_HASH_FILE="${VENV}/.requirements.sha256"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    python3 -m venv "${VENV}"
fi
REQ_HASH="$(sha256sum "${WORKDIR}/requirements.txt" | awk '{print $1}')"
if [[ ! -f "${REQ_HASH_FILE}" || "$(cat "${REQ_HASH_FILE}" 2>/dev/null || echo)" != "${REQ_HASH}" ]]; then
    "${PYTHON_BIN}" -m pip install --disable-pip-version-check --no-cache-dir --upgrade pip setuptools wheel
    "${PYTHON_BIN}" -m pip install --disable-pip-version-check --no-cache-dir -r "${WORKDIR}/requirements.txt"
fi

# Uso: sbatch slurm/check_lora.sh [out.json] [files...] (defecto: weights/)
OUT_JSON="${1:-data/output/lora_audit.json}"
shift || true
if [[ $# -eq 0 ]]; then
    set -- weights/lora_person_A.safetensors weights/lora_person_B.safetensors
fi

"${PYTHON_BIN}" src/check_lora_weights.py \
    --weights "$@" \
    --out "${OUT_JSON}"
