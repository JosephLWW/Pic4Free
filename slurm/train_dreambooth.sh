#!/bin/bash
# LoRA DreamBooth native diffusers (pipeline venv: torch+diffusers+peft).
# Uso: sbatch slurm/train_dreambooth.sh <A|B> <steps> <rank> [tag]
#SBATCH --job-name=Pic4Free_dblora
#SBATCH --output=slurm_logs/slurm_%j.out
#SBATCH --error=slurm_logs/slurm_%j.err
#SBATCH --partition=gpu_h100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --time=00:28:00

set -Eeuo pipefail

unset PYTHONPATH
unset VIRTUAL_ENV
hash -r

module purge
module load devel/python/3.12.3-gnu-14.2

PERSON="${1:?A o B}"
STEPS="${2:?steps}"
RANK="${3:?rank}"
TAG="${4:-${PERSON}}"

WORKDIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not defined}"
WORKDIR="$(cd "${WORKDIR}" && pwd)"
cd "${WORKDIR}"

CACHE_ROOT="${PIC4FREE_CACHE_ROOT:-${SCRATCH:-${TMPDIR}}}"
CACHE_ROOT="${CACHE_ROOT}/pic4free/${USER}"

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
    echo "${REQ_HASH}" > "${REQ_HASH_FILE}"
fi

if [[ "${PERSON}" == "A" ]]; then TRIGGER="P4F_A"; else TRIGGER="P4F_B"; fi

export HF_HOME="${CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_TOKEN="$(cat ${WORKDIR}/hf_token.txt)"
export PIP_NO_CACHE_DIR=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/cudnn/lib:${VENV}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

OUTDIR="${WORKDIR}/data/lora/dblora_${TAG}"
mkdir -p "${OUTDIR}"
# OUTDIR on shared storage (/pfs): checkpoints survive timeouts and a
# noof changes. Lesson learned: on local scratch, a TIMEOUT loses
# anything not persisted (the epilogue does not run).

# The diffusers DreamBooth dataset opens EVERYTHING in the directory
# (including ai-toolkit caption .txt files) -> image-only staging.
STAGE_DIR="${CACHE_ROOT}/dblora_stage/${TAG}"
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"
find "${WORKDIR}/data/lora/train_${PERSON}" -maxdepth 1 -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
    -exec cp -t "${STAGE_DIR}" {} +
echo "[dblora] staging: $(ls "${STAGE_DIR}" | wc -l) images in ${STAGE_DIR}"

# shellcheck disable=SC2086
"${PYTHON_BIN}" -m accelerate.commands.launch --mixed_precision=bf16 \
    scripts/train_dreambooth_lora_flux.py \
    --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
    --instance_data_dir="${STAGE_DIR}" \
    --instance_prompt="${TRIGGER} portrait photo, face and shoulders, natural light, photorealistic" \
    --rank=${RANK} \
    --resolution=1024 \
    --center_crop \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --max_train_steps=${STEPS} \
    --checkpointing_steps=100 \
    --checkpoints_total_limit=4 \
    --validation_prompt="${TRIGGER} portrait photo, neutral background, studio light" \
    --num_validation_images=2 \
    --validation_epochs=25 \
    --mixed_precision="bf16" \
    --seed=42 \
    --resume_from_checkpoint="latest" \
    --output_dir="${OUTDIR}"
TRAIN_EXIT=$?

# Persist to shared storage: latest checkpoint + samples + B finiteness.
PERSIST="${WORKDIR}/data/lora/dblora_${TAG}"
mkdir -p "${PERSIST}"
LAST_CKPT="$(ls -dt "${OUTDIR}"/checkpoint-* 2>/dev/null | head -1 || true)"
if [[ -n "${LAST_CKPT}" && -d "${LAST_CKPT}" ]]; then
    cp -r "${LAST_CKPT}" "${PERSIST}/" 2>/dev/null || true
fi
find "${OUTDIR}" -maxdepth 2 -name "*.png" -newermt "-1 day ago" 2>/dev/null | head -8 | while read -r _F; do
    cp -n "${_F}" "${PERSIST}/" 2>/dev/null || true
done
# Full checkpoint mirror to shared storage (resume across different nodes).
for _C in "${OUTDIR}"/checkpoint-*; do
    [[ -d "${_C}" ]] || continue
    _B="$(basename "${_C}")"
    [[ -d "${PERSIST}/${_B}" ]] || cp -r "${_C}" "${PERSIST}/" 2>/dev/null || true
done
"${PYTHON_BIN}" - <<PY 2>/dev/null || echo "[dblora] No final safetensors to audit (check log)."
import glob, struct, json
files = sorted(glob.glob("${PERSIST}/checkpoint-*/**/*.safetensors", recursive=True))
print("checkpoints persistidos:", files)
PY

exit ${TRAIN_EXIT}
