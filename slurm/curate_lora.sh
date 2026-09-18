#!/bin/bash
# Phase 0 LoRA curation (CPU): face-and-shoulder crops + captions + held-out.
#SBATCH --job-name=Pic4Free_curate
#SBATCH --output=slurm_logs/slurm_%j.out
#SBATCH --error=slurm_logs/slurm_%j.err
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=0:30:00

set -Eeuo pipefail

unset PYTHONPATH
unset VIRTUAL_ENV
hash -r

module purge
module load devel/python/3.12.3-gnu-14.2

WORKDIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not defined}"
WORKDIR="$(cd "${WORKDIR}" && pwd)"
cd "${WORKDIR}"
export PYTHONPATH="${WORKDIR}/src:${PYTHONPATH:-}"

if [[ -z "${PIC4FREE_CACHE_ROOT:-}" && -z "${SCRATCH:-}" && -z "${TMPDIR:-}" ]]; then
    echo "ERROR: SCRATCH/TMPDIR does not exist for caches." >&2
    exit 1
fi

CACHE_ROOT="${PIC4FREE_CACHE_ROOT:-${SCRATCH:-${TMPDIR}}}"
CACHE_ROOT="${CACHE_ROOT}/pic4free/${USER}"
mkdir -p "${CACHE_ROOT}/insightface"

export INSIGHTFACE_HOME="${CACHE_ROOT}/insightface"
export PIP_NO_CACHE_DIR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

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
else
    echo "[curate] Requirements already installed; reusing the venv."
fi

mkdir -p data/lora

"${PYTHON_BIN}" -m pic4free.cli.curate_lora \
    --a-dir "${WORKDIR}/data/identity_person_A" \
    --b-dir "${WORKDIR}/data/identity_person_B" \
    --out "${WORKDIR}/data/lora" \
    --trigger-a P4F_A --trigger-b P4F_B --held-n 2
