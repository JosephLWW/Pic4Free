#!/bin/bash
#SBATCH --job-name=Pic4Free_single
#SBATCH --output=slurm_logs/slurm_%j.out
#SBATCH --error=slurm_logs/slurm_%j.err
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00

set -Eeuo pipefail

echo "=============================================================================="
echo "Iniciando Job Individual Pic4Free"
echo "Fecha y Hora: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="

unset PYTHONPATH
unset VIRTUAL_ENV
hash -r

module purge
module load devel/python/3.12.3-gnu-14.2

# PyTorch >= 2.x instala sus propias librerias CUDA/cuDNN via pip (nvidia-cudnn-cu13).
# NO cargar devel/cuda/12.x del sistema: entra en conflicto con cuDNN 9.x del venv
# y provoca CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED en el primer conv2d.
# Anteponemos las libs del venv en LD_LIBRARY_PATH para que el linker las encuentre.

# Resolver el repositorio desde el directorio de envío de Slurm
WORKDIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR no está definido}"
WORKDIR="$(cd "${WORKDIR}" && pwd)"

# Validar estructura mínima del repositorio
for required in "requirements.txt" "src/main.py"; do
    if [[ ! -e "${WORKDIR}/${required}" ]]; then
        echo "ERROR: faltan ${required} dentro de ${WORKDIR}" >&2
        exit 1
    fi
done

cd "${WORKDIR}"

# Usar almacenamiento temporal o Scratch, no la cuota del proyecto
if [[ -z "${PIC4FREE_CACHE_ROOT:-}" && -z "${SCRATCH:-}" && -z "${TMPDIR:-}" ]]; then
    echo "ERROR: No existe SCRATCH/TMPDIR para almacenar cachés." >&2
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

# LoRAs finales: ai-toolkit los guarda en el scratch de cómputo (no visible
# desde login ni desde otros clústeres); si faltan en weights/, copiarlos aquí.
for _P in A B; do
    _DST="${WORKDIR}/weights/lora_person_${_P}.safetensors"
    _SRC="${CACHE_ROOT}/lora_out/lora_person_${_P}/lora_person_${_P}.safetensors"
    if [[ ! -f "${_DST}" && -f "${_SRC}" ]]; then
        echo "[Pic4Free] Copiando LoRA ${_P} desde scratch..."
        cp "${_SRC}" "${_DST}"
    fi
    if [[ -f "${_DST}" ]]; then
        python3 -c "
import struct, json
d = open('${_DST}','rb').read(8)
n = struct.unpack('<Q', d[:8])[0]
h = json.loads(open('${_DST}','rb').read(8+n)[8:])
ks = [k for k in h if k != '__metadata__']
print('LoRA ${_P} OK: ${_DST} con', len(ks), 'tensores')
assert len(ks) > 100, 'checkpoint sospechosamente pequeño'
" || echo "[Pic4Free] AVISO: LoRA ${_P} no verificable; refino genérico."
    else
        echo "[Pic4Free] Sin LoRA ${_P}: refino facial genérico."
    fi
done

VENV="${CACHE_ROOT}/venv"
PYTHON_BIN="${VENV}/bin/python"
PIP_BIN="${VENV}/bin/pip"
REQ_HASH_FILE="${VENV}/.requirements.sha256"

# Anteponer las libs CUDA/cuDNN empaquetadas en el venv antes que las del sistema.
# Esto permite que PyTorch 2.14+ use nvidia-cudnn-cu13 del venv en lugar del
# cuDNN del modulo devel/cuda/12.8, evitando CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED.
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/cudnn/lib:${VENV}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

# --- Setup del venv bajo lock exclusivo: varios jobs concurrentes comparten
# el venv y pip no es seguro en paralelo (corrompe numpy/scipy/onnx).
# Solo esta sección se serializa; el pipeline posterior corre en paralelo.
VENV_LOCK="${CACHE_ROOT}/venv.lock"
exec 200>"${VENV_LOCK}"
echo "[Pic4Free] Adquiriendo lock del venv..."
flock -x 200
echo "[Pic4Free] Lock adquirido."

# Crear venv si no existe
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[Pic4Free] Entorno virtual no encontrado en ${VENV}. Creando entorno..."
    python3 -m venv "${VENV}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: no se pudo crear el entorno virtual en ${VENV}" >&2
    exit 1
fi

# Autoreparación: si los imports clave fallan (venv corrupto por carrera previa),
# reconstruir desde cero. Seguro para jobs ya en marcha: solo usan módulos en memoria.
if ! "${PYTHON_BIN}" -c "import numpy, scipy, torch, diffusers, omegaconf" 2>/dev/null; then
    echo "[Pic4Free] venv corrupto detectado: reconstruyendo ${VENV}..."
    rm -rf "${VENV}"
    python3 -m venv "${VENV}"
    rm -f "${REQ_HASH_FILE}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: no se pudo reconstruir el entorno virtual en ${VENV}" >&2
    exit 1
fi

# Chequeo explícito del entorno
echo "Python real: $("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
echo "Python versión: $("${PYTHON_BIN}" -c 'import sys; print(sys.version.split()[0])')"

# Rebuild limpio bajo demanda (venv frankenstein tras fallo parcial):
#   FORCE_VENV=1 sbatch slurm/restore_single.sh ...
if [[ "${FORCE_VENV:-0}" == "1" ]]; then
    echo "[Pic4Free] FORCE_VENV=1: eliminando venv para reconstrucción limpia..."
    rm -rf "${VENV}" "${REQ_HASH_FILE}"
    python3 -m venv "${VENV}"
fi

# Reutilizar dependencias si el requirements.txt no ha cambiado
REQ_HASH="$(sha256sum "${WORKDIR}/requirements.txt" | awk '{print $1}')"

if [[ ! -f "${REQ_HASH_FILE}" || "$(cat "${REQ_HASH_FILE}" 2>/dev/null || echo)" != "${REQ_HASH}" ]]; then
    echo "[Pic4Free] Instalando o actualizando dependencias..."
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
    echo "[Pic4Free] Requisitos ya instalados; reutilizando el entorno virtual."
fi

# ONNX Runtime GPU: solo purgar/reinstalar si CUDA no está disponible.
# La purga incondicional en cada job es la principal fuente de carreras.
HAS_CUDA_ORT="$("${PYTHON_BIN}" -c 'import onnxruntime as ort; print("yes" if "CUDAExecutionProvider" in ort.get_available_providers() else "no")' 2>/dev/null || echo "no")"
if [[ "${HAS_CUDA_ORT}" != "yes" ]]; then
    echo "[Pic4Free] ONNX sin CUDA: purgando CPU y forzando onnxruntime-gpu..."
    "${PYTHON_BIN}" -m pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
    "${PYTHON_BIN}" -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --force-reinstall \
        "onnxruntime-gpu>=1.19.0"
else
    echo "[Pic4Free] ONNX Runtime GPU ya disponible; se omite la purga."
fi

# ✅ VERIFY GPU PROVIDER
"${PYTHON_BIN}" - <<'PY'
import onnxruntime as ort

providers = ort.get_available_providers()
print("ONNX Runtime providers:", providers)

if "CUDAExecutionProvider" not in providers:
    raise RuntimeError("ONNX Runtime GPU no está disponible")
PY

# Liberar lock: el resto del job (pipeline) corre en paralelo
flock -u 200
exec 200>&-
echo "[Pic4Free] Lock del venv liberado."

# Determinar si el upscaling SUPIR está activado en la config
ENABLE_UPSCALING="$("${PYTHON_BIN}" - <<'PY'
from src.config import load_config
cfg = load_config()
print("true" if cfg.superres.enable_upscaling else "false")
PY
)"
echo "[Pic4Free] superres.enable_upscaling=${ENABLE_UPSCALING}"

# Instalar y verificar SUPIR solo si el upscaling está activo
SUPIR_DIR="${CACHE_ROOT}/SUPIR_repo"
if [[ "${ENABLE_UPSCALING}" == "true" ]]; then
    # Asegurar pkg_resources (requerido por dependencias de SUPIR)
    "${PYTHON_BIN}" -m pip install --disable-pip-version-check --no-cache-dir setuptools -q

    if [[ ! -d "${SUPIR_DIR}" ]]; then
        echo "[Pic4Free] Clonando repositorio SUPIR oficial..."
        git clone --depth 1 https://github.com/Fanghua-Yu/SUPIR.git "${SUPIR_DIR}"
    fi
    export PYTHONPATH="${SUPIR_DIR}:${PYTHONPATH:-}"
    echo "[Pic4Free] SUPIR configurado en PYTHONPATH: ${SUPIR_DIR}"

    # ✅ VERIFY SUPIR PACKAGE
    "${PYTHON_BIN}" - <<'PY'
import sys
try:
    from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL
    print("[OK] SUPIR package importado correctamente.")
except ImportError as e:
    print(f"PYTHONPATH: {sys.path}")
    raise RuntimeError(f"SUPIR package no disponible: {e}")
PY
else
    # SUPIR no es necesario; añadir el directorio al PYTHONPATH solo si ya existe
    # para no romper ejecuciones previas que lo cachearon
    if [[ -d "${SUPIR_DIR}" ]]; then
        export PYTHONPATH="${SUPIR_DIR}:${PYTHONPATH:-}"
    fi
    echo "[Pic4Free] SUPIR upscaling desactivado; se omite la verificación del paquete."
fi

echo "[Pic4Free] Verificando PyTorch/CUDA..."
"${PYTHON_BIN}" - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Capacidad: {torch.cuda.get_device_capability(0)}")
    print(f"CUDA runtime: {torch.version.cuda}")
PY

JOB_ID="${SLURM_JOB_ID}"
TASK_ID="${1:-${SLURM_ARRAY_TASK_ID:-0}}"
BACKEND="${2:-flux}"
IDSCALE="${3:-1.0}"
USE_INPAINT="${4:-true}"
SEL="${5:-average}"
OUTPUT_BASE="${WORKDIR}/data/output/run_${JOB_ID}"

mkdir -p "${OUTPUT_BASE}/restored" \
         "${OUTPUT_BASE}/metrics" \
         "${OUTPUT_BASE}/logs"

echo "=============================================================================="
echo "Job ID:            ${SLURM_JOB_ID}"
echo "Task ID (Índice):  ${TASK_ID}"
echo "Nodo:              ${SLURMD_NODENAME}"
echo "Python:            $("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
echo "GPU:               $("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "ERROR: Sin GPU")')"
echo "Directorio Salida: ${OUTPUT_BASE}"
echo "=============================================================================="

GUIDANCE="3.5"
if [[ "${BACKEND}" == "kontext" ]]; then
    GUIDANCE="2.5"
fi
echo "Backend:           ${BACKEND} (guidance ${GUIDANCE}, id_scale ${IDSCALE}, use_inpaint ${USE_INPAINT}, selection ${SEL})"

HF_TOKEN_FILE="${WORKDIR}/hf_token.txt"
EXTRA_ARGS=()

if [[ -f "${HF_TOKEN_FILE}" ]]; then
    echo "[Pic4Free] Usando token HF desde: ${HF_TOKEN_FILE}"
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

if [[ -f "${WORKDIR}/slurm_logs/slurm_${SLURM_JOB_ID}.out" ]]; then
    cp "${WORKDIR}/slurm_logs/slurm_${SLURM_JOB_ID}.out" "${OUTPUT_BASE}/logs/" 2>/dev/null || true
fi
if [[ -f "${WORKDIR}/slurm_logs/slurm_${SLURM_JOB_ID}.err" ]]; then
    cp "${WORKDIR}/slurm_logs/slurm_${SLURM_JOB_ID}.err" "${OUTPUT_BASE}/logs/" 2>/dev/null || true
fi

echo "=============================================================================="
echo "Ejecución individual finalizada con código de salida: ${EXIT_CODE}"
echo "=============================================================================="

exit ${EXIT_CODE}