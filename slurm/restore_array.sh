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
#SBATCH --array=0-97%16

set -Eeuo pipefail

echo "=============================================================================="
echo "Iniciando Tarea Slurm Array Pic4Free"
echo "Fecha y Hora: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="

unset PYTHONPATH
unset VIRTUAL_ENV
hash -r

module purge
module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-14.2

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
export TORCH_HOME="${CACHE_ROOT}/torch"
export INSIGHTFACE_HOME="${CACHE_ROOT}/insightface"
export PIP_NO_CACHE_DIR=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PIC4FREE_IDENTITY_CACHE="${CACHE_ROOT}/identity"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export HOME="${CACHE_ROOT}/home"
mkdir -p "${PIC4FREE_IDENTITY_CACHE}" "${XDG_CACHE_HOME}" "${HOME}"

VENV="${CACHE_ROOT}/venv"
PYTHON_BIN="${VENV}/bin/python"
PIP_BIN="${VENV}/bin/pip"
REQ_HASH_FILE="${VENV}/.requirements.sha256"

# Crear venv si no existe o si está corrupto
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[Pic4Free] Entorno virtual no encontrado en ${VENV}. Creando entorno..."
    python3 -m venv "${VENV}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: no se pudo crear el entorno virtual en ${VENV}" >&2
    exit 1
fi

# Chequeo explícito del entorno
echo "Python real: $("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
echo "Python versión: $("${PYTHON_BIN}" -c 'import sys; print(sys.version.split()[0])')"

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

    # Mantener exclusivamente ONNX Runtime GPU.
    "${PYTHON_BIN}" -m pip uninstall -y onnxruntime >/dev/null 2>&1 || true
    "${PYTHON_BIN}" -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --upgrade \
        "onnxruntime-gpu>=1.19.0"

    printf '%s\n' "${REQ_HASH}" > "${REQ_HASH_FILE}"
else
    echo "[Pic4Free] Requisitos ya instalados; reutilizando el entorno virtual."
fi

# Asegurar exclusivamente ONNX Runtime GPU en cada ejecución
"${PYTHON_BIN}" -m pip uninstall -y onnxruntime >/dev/null 2>&1 || true
"${PYTHON_BIN}" -m pip install \
    --disable-pip-version-check \
    --no-cache-dir \
    --upgrade \
    "onnxruntime-gpu>=1.19.0"

# ✅ VERIFY GPU PROVIDER AFTER UNINSTALL
"${PYTHON_BIN}" - <<'PY'
import onnxruntime as ort

providers = ort.get_available_providers()
print("ONNX Runtime providers:", providers)

if "CUDAExecutionProvider" not in providers:
    raise RuntimeError("ONNX Runtime GPU no está disponible")
PY

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

JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
OUTPUT_BASE="${WORKDIR}/data/output/run_${JOB_ID}"

mkdir -p "${OUTPUT_BASE}/restored" \
         "${OUTPUT_BASE}/masks" \
         "${OUTPUT_BASE}/intermediates" \
         "${OUTPUT_BASE}/metrics" \
         "${OUTPUT_BASE}/logs"

echo "=============================================================================="
echo "DETALLES DEL ENTORNO DE EJECUCIÓN"
echo "=============================================================================="
echo "Array Job ID:         ${JOB_ID}"
echo "Array Task ID:        ${TASK_ID}"
echo "Nodo de Ejecución:    ${SLURMD_NODENAME}"
echo "Ruta Python:          $("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
echo "Versión Python:       $("${PYTHON_BIN}" -c 'import sys; print(sys.version.split()[0])')"
echo "Dispositivo CUDA:     $("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "ERROR: CUDA no disponible")')"
echo "Directorio Salida:    ${OUTPUT_BASE}"
echo "=============================================================================="


HF_TOKEN_FILE="${WORKDIR}/hf_token.txt"
EXTRA_ARGS=()

if [[ -f "${HF_TOKEN_FILE}" ]]; then
    echo "[Pic4Free] Usando token HF desde: ${HF_TOKEN_FILE}"
    EXTRA_ARGS+=(--hf_token_file "${HF_TOKEN_FILE}")
fi

"${PYTHON_BIN}" src/main.py \
    --task_id "${TASK_ID}" \
    --job_id "${JOB_ID}" \
    --input_dir "${WORKDIR}/data/input" \
    --identity_a_dir "${WORKDIR}/data/identity_person_A" \
    --identity_b_dir "${WORKDIR}/data/identity_person_B" \
    --output_dir "${OUTPUT_BASE}" \
    --num_inference_steps 28 \
    --guidance_scale 3.5 \
    --ssim_threshold 0.65 \
    --face_fidelity_weight 0.85 \
    --enable_face_refinement \
    "${EXTRA_ARGS[@]}"

EXIT_CODE=$?

# Copiar logs si existen
if [[ -f "${WORKDIR}/slurm_logs/slurm_${JOB_ID}_${TASK_ID}.out" ]]; then
    cp "${WORKDIR}/slurm_logs/slurm_${JOB_ID}_${TASK_ID}.out" "${OUTPUT_BASE}/logs/" 2>/dev/null || true
fi
if [[ -f "${WORKDIR}/slurm_logs/slurm_${JOB_ID}_${TASK_ID}.err" ]]; then
    cp "${WORKDIR}/slurm_logs/slurm_${JOB_ID}_${TASK_ID}.err" "${OUTPUT_BASE}/logs/" 2>/dev/null || true
fi

echo "=============================================================================="
echo "Tarea Slurm Array completada con código de salida: ${EXIT_CODE}"
echo "Resultados disponibles en: ${OUTPUT_BASE}"
echo "=============================================================================="

exit ${EXIT_CODE}