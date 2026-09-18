#!/bin/bash
#SBATCH --job-name=Pic4Free_array
#SBATCH --output=slurm_logs/slurm_%A_%a.out
#SBATCH --error=slurm_logs/slurm_%A_%a.err
#SBATCH --partition=gpu_h100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --array=0-96%16

set -Eeuo pipefail

echo "=============================================================================="
echo "Iniciando Tarea Slurm Array Pic4Free"
echo "Date and Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="

unset PYTHONPATH
unset VIRTUAL_ENV
hash -r

module purge
module load devel/python/3.12.3-gnu-14.2

# PyTorch >= 2.x installs sus propias librerias CUDA/cuDNN via pip (nvidia-cudnn-cu13).
# Do NOT load devel/cuda/12.x of the system: conflicts with with cuDNN 9.x of the venv
# and causes CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED in the first conv2d.
# Prepend the libs of the venv in LD_LIBRARY_PATH so the linker can find them.

# Resolve the repository from the Slurm submission directory
WORKDIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not defined}"
WORKDIR="$(cd "${WORKDIR}" && pwd)"

# Validate the minimum repository structure
for required in "requirements.txt" "src/main.py"; do
    if [[ ! -e "${WORKDIR}/${required}" ]]; then
        echo "ERROR: missing ${required} insiof ${WORKDIR}" >&2
        exit 1
    fi
done

cd "${WORKDIR}"

# Use temporary storage or Scratch, not the project quota
if [[ -z "${PIC4FREE_CACHE_ROOT:-}" && -z "${SCRATCH:-}" && -z "${TMPDIR:-}" ]]; then
    echo "ERROR: SCRATCH/TMPDIR does not exist for storing caches." >&2
    exit 1
fi

CACHE_ROOT="${PIC4FREE_CACHE_ROOT:-${SCRATCH:-${TMPDIR}}}"
CACHE_ROOT="${CACHE_ROOT}/pic4free/${USER}"

mkdir -p \
    "${CACHE_ROOT}/huggingface" \
    "${CACHE_ROOT}/torch" \
    "${CACHE_ROOT}/insightface"

export HF_HOME="${CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_TOKEN="$(cat ${WORKDIR}/hf_token.txt)"
export TORCH_HOME="${CACHE_ROOT}/torch"
export INSIGHTFACE_HOME="${CACHE_ROOT}/insightface"
export PIP_NO_CACHE_DIR=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PIC4FREE_IDENTITY_CACHE="${CACHE_ROOT}/identity"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export HOME="${CACHE_ROOT}/home"
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=false
export OMP_PLACES=threads
export ORT_DISABLE_CPU_AFFINITY=1
mkdir -p "${PIC4FREE_IDENTITY_CACHE}" "${XDG_CACHE_HOME}" "${HOME}"
mkdir -p "${WORKDIR}/weights"

# Final LoRAs: ai-toolkit stores them in compute scratch (not visible
# from login or other clusters); if missing from weights/, copy them here.
for _P in A B; do
    _DST="${WORKDIR}/weights/lora_person_${_P}.safetensors"
    _SRC="${CACHE_ROOT}/lora_out/lora_person_${_P}/lora_person_${_P}.safetensors"
    if [[ ! -f "${_DST}" && -f "${_SRC}" ]]; then
        echo "[Pic4Free] Copying LoRA ${_P} from scratch..."
        cp "${_SRC}" "${_DST}"
    fi
    if [[ -f "${_DST}" ]]; then
        python3 -c "
import struct, json
d = open('${_DST}','rb').read(8)
n = struct.unpack('<Q', d[:8])[0]
h = json.loads(open('${_DST}','rb').read(8+n)[8:])
ks = [k for k in h if k != '__metadata__']
print('LoRA ${_P} OK: ${_DST} with', len(ks), 'tensors')
assert len(ks) > 100, 'checkpoint suspiciously small'
" || echo "[Pic4Free] WARNING: LoRA ${_P} not verifiable; generic refinement."
    else
        echo "[Pic4Free] No LoRA ${_P}: refino facial generic."
    fi
done

VENV="${CACHE_ROOT}/venv"
PYTHON_BIN="${VENV}/bin/python"
PIP_BIN="${VENV}/bin/pip"
REQ_HASH_FILE="${VENV}/.requirements.sha256"

# Prepend the CUDA/cuDNN libraries bundled in the venv before the system libraries.
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/cudnn/lib:${VENV}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

# --- Set up the venv under an exclusive lock: array tasks share the venv
# and pip is not safe in parallthe (corrupts numpy/scipy/onnx).
VENV_LOCK="${CACHE_ROOT}/venv.lock"
exec 200>"${VENV_LOCK}"
echo "[Pic4Free] Acquiring venv lock..."
flock -x 200
echo "[Pic4Free] Lock adquirido."

# Crear venv si does not exist
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[Pic4Free] Virtual environment not found in ${VENV}. Creating environment..."
    python3 -m venv "${VENV}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: could not create the virtual environment in ${VENV}" >&2
    exit 1
fi

# Self-repair: if key imports fail (venv is corrupted by a previous race),
# rebuild from scratch. Safe for jobs already running: they only use in-memory modules.
if ! "${PYTHON_BIN}" -c "import numpy, scipy, torch, diffusers, omegaconf" 2>/dev/null; then
    echo "[Pic4Free] corrupted venv detected: rebuilding ${VENV}..."
    rm -rf "${VENV}"
    python3 -m venv "${VENV}"
    rm -f "${REQ_HASH_FILE}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: could not rebuild the virtual environment in ${VENV}" >&2
    exit 1
fi

# Explicit environment check
echo "Python real: $("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
echo "Python version: $("${PYTHON_BIN}" -c 'import sys; print(sys.version.split()[0])')"

# Clean rebuild on demand (corrupted venv after a partial failure):
#   FORCE_VENV=1 sbatch slurm/restore_array.sh
if [[ "${FORCE_VENV:-0}" == "1" ]]; then
    echo "[Pic4Free] FORCE_VENV=1: removing venv for clean rebuild..."
    rm -rf "${VENV}" "${REQ_HASH_FILE}"
    python3 -m venv "${VENV}"
fi

# Reuse dependencies if requirements.txt has not changed
REQ_HASH="$(sha256sum "${WORKDIR}/requirements.txt" | awk '{print $1}')"

if [[ ! -f "${REQ_HASH_FILE}" || "$(cat "${REQ_HASH_FILE}" 2>/dev/null || echo)" != "${REQ_HASH}" ]]; then
    echo "[Pic4Free] Installing or updating dependencies..."
    "${PYTHON_BIN}" -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --upgrade pip setuptools wheel

    "${PYTHON_BIN}" -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        -r "${WORKDIR}/requirements.txt"
    echo "${REQ_HASH}" > "${REQ_HASH_FILE}"
else
    echo "[Pic4Free] Requirements already installed; reusing the virtual environment."
fi

# ONNX Runtime GPU: purge/reinstall only if CUDA is unavailable.
HAS_CUDA_ORT="$("${PYTHON_BIN}" -c 'import onnxruntime as ort; print("yes" if "CUDAExecutionProvider" in ort.get_available_providers() else "no")' 2>/dev/null || echo "no")"
if [[ "${HAS_CUDA_ORT}" != "yes" ]]; then
    echo "[Pic4Free] ONNX without CUDA: purging CPU runtime and forcing onnxruntime-gpu..."
    "${PYTHON_BIN}" -m pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
    "${PYTHON_BIN}" -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --force-reinstall \
        "onnxruntime-gpu>=1.19.0"
else
    echo "[Pic4Free] ONNX Runtime GPU is already available; skipping the purge."
fi

#  VERIFY GPU PROVIDER
"${PYTHON_BIN}" - <<'PY'
import onnxruntime as ort

providers = ort.get_available_providers()
print("ONNX Runtime providers:", providers)

if "CUDAExecutionProvider" not in providers:
    raise RuntimeError("ONNX Runtime GPU is unavailable")
PY

# Release lock: the rest of the job (pipeline) runs in parallel
flock -u 200
exec 200>&-
echo "[Pic4Free] Venv lock released."

# Determine whether SUPIR upscaling is enabled in the config
ENABLE_UPSCALING="$("${PYTHON_BIN}" - <<'PY'
from src.config import load_config
cfg = load_config()
print("true" if cfg.superres.enable_upscaling else "false")
PY
)"
echo "[Pic4Free] superres.enable_upscaling=${ENABLE_UPSCALING}"

# Install and verify SUPIR only when upscaling is enabled
SUPIR_DIR="${CACHE_ROOT}/SUPIR_repo"
if [[ "${ENABLE_UPSCALING}" == "true" ]]; then
    # Ensure pkg_resources (required by SUPIR dependencies)
    "${PYTHON_BIN}" -m pip install --disable-pip-version-check --no-cache-dir setuptools -q

    if [[ ! -d "${SUPIR_DIR}" ]]; then
        echo "[Pic4Free] Cloning the official SUPIR repository..."
        git clone --depth 1 https://github.com/Fanghua-Yu/SUPIR.git "${SUPIR_DIR}"
    fi
    export PYTHONPATH="${SUPIR_DIR}:${PYTHONPATH:-}"
    echo "[Pic4Free] SUPIR configured in PYTHONPATH: ${SUPIR_DIR}"

    #  VERIFY SUPIR PACKAGE
    "${PYTHON_BIN}" - <<'PY'
import sys
try:
    from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL
    print("[OK] SUPIR package imported successfully.")
except ImportError as e:
    print(f"PYTHONPATH: {sys.path}")
    raise RuntimeError(f"SUPIR package unavailable: {e}")
PY
else
    # SUPIR is not required; add the directory to PYTHONPATH only if it already exists
    if [[ -d "${SUPIR_DIR}" ]]; then
        export PYTHONPATH="${SUPIR_DIR}:${PYTHONPATH:-}"
    fi
    echo "[Pic4Free] SUPIR upscaling disabled; skipping package verification."
fi

echo "[Pic4Free] Verificando PyTorch/CUDA..."
"${PYTHON_BIN}" - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Capacidad: {torch.cuda.get_device_capability(0)}")
    print(f"CUDA runtime: {torch.version.cuda}")
PY

JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
BACKEND="${BACKEND_OVERRIDE:-kontext}"
IDSCALE="${IDSCALE_OVERRIDE:-1.0}"
USE_INPAINT="${USE_INPAINT_OVERRIDE:-true}"
SEL="${SEL_OVERRIDE:-average}"
OUTPUT_BASE="${WORKDIR}/data/output/run_${JOB_ID}"

mkdir -p "${OUTPUT_BASE}/restored" \
         "${OUTPUT_BASE}/metrics" \
         "${OUTPUT_BASE}/logs"

echo "=============================================================================="
echo "EXECUTION ENVIRONMENT DETAILS"
echo "=============================================================================="
echo "Array Job ID:         ${JOB_ID}"
echo "Array Task ID:        ${TASK_ID}"
echo "Execution Node:    ${SLURMD_NODENAME}"
echo "Ruta Python:          $("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
echo "Python version:       $("${PYTHON_BIN}" -c 'import sys; print(sys.version.split()[0])')"
echo "CUDA device:     $("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "ERROR: CUDA unavailable")')"
echo "Output Directory:    ${OUTPUT_BASE}"
echo "=============================================================================="


GUIDANCE="3.5"
if [[ "${BACKEND}" == "kontext" ]]; then
    GUIDANCE="2.5"
fi
echo "Backend:           ${BACKEND} (guidance ${GUIDANCE}, id_scale ${IDSCALE}, use_inpaint ${USE_INPAINT}, selection ${SEL})"

HF_TOKEN_FILE="${WORKDIR}/hf_token.txt"
EXTRA_ARGS=()

if [[ -f "${HF_TOKEN_FILE}" ]]; then
    echo "[Pic4Free] Using HF token from: ${HF_TOKEN_FILE}"
    EXTRA_ARGS+=("flux.hf_token_file=${HF_TOKEN_FILE}")
fi

"${PYTHON_BIN}" src/main.py \
    task_id="${TASK_ID}" \
    job_id="${JOB_ID}" \
    paths.input_dir="${WORKDIR}/data/input" \
    paths.identity_a_dir="${WORKDIR}/data/identity_person_A" \
    paths.identity_b_dir="${WORKDIR}/data/identity_person_B" \
    paths.output_dir="${OUTPUT_BASE}" \
    flux.backend="${BACKEND}" \
    flux.model_id="black-forest-labs/FLUX.1-dev" \
    flux.num_inference_steps=35 \
    flux.guidance_scale=${GUIDANCE} \
    flux.id_scale=${IDSCALE} \
    flux.use_inpaint_pipeline=${USE_INPAINT} \
    identity.selection=${SEL} \
    masking.ssim_threshold=0.65 \
    superres.face_fidelity_weight=0.85 \
    superres.enable_face_refinement=true \
    "${EXTRA_ARGS[@]}"

EXIT_CODE=$?

# Copy logs if exist
if [[ -f "${WORKDIR}/slurm_logs/slurm_${JOB_ID}_${TASK_ID}.out" ]]; then
    cp "${WORKDIR}/slurm_logs/slurm_${JOB_ID}_${TASK_ID}.out" "${OUTPUT_BASE}/logs/" 2>/dev/null || true
fi
if [[ -f "${WORKDIR}/slurm_logs/slurm_${JOB_ID}_${TASK_ID}.err" ]]; then
    cp "${WORKDIR}/slurm_logs/slurm_${JOB_ID}_${TASK_ID}.err" "${OUTPUT_BASE}/logs/" 2>/dev/null || true
fi

echo "=============================================================================="
echo "Slurm Array task completed with exit code: ${EXIT_CODE}"
echo "Results available in: ${OUTPUT_BASE}"
echo "=============================================================================="

exit ${EXIT_CODE}