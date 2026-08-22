#!/bin/bash
#SBATCH --job-name=Pic4Free_single
#SBATCH --output=slurm_logs/slurm_%j.out
#SBATCH --error=slurm_logs/slurm_%j.err
#SBATCH --partition=gpu_h100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00

set -Eeuo pipefail

# Argumento opcional: índice de imagen para depuración
TASK_ID="${1:-0}"

echo "=============================================================================="
echo "Iniciando Job Individual Pic4Free: Job ${SLURM_JOB_ID} | Task ID: ${TASK_ID}"
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

# Directorios de trabajo dentro del repositorio
mkdir -p "${WORKDIR}/slurm_logs" \
         "${WORKDIR}/.cache/huggingface" \
         "${WORKDIR}/.cache/torch" \
         "${WORKDIR}/.cache/insightface"

export HF_HOME="${WORKDIR}/.cache/huggingface"
export TORCH_HOME="${WORKDIR}/.cache/torch"
export INSIGHTFACE_HOME="${WORKDIR}/.cache/insightface"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

VENV="${WORKDIR}/.venv"
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
    "${PYTHON_BIN}" -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
    "${PYTHON_BIN}" -m pip install --disable-pip-version-check -r "${WORKDIR}/requirements.txt"
    printf '%s\n' "${REQ_HASH}" > "${REQ_HASH_FILE}"
else
    echo "[Pic4Free] Requisitos ya instalados; reutilizando el entorno virtual."
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

OUTPUT_BASE="${WORKDIR}/data/output/run_${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_BASE}/restored" \
         "${OUTPUT_BASE}/masks" \
         "${OUTPUT_BASE}/intermediates" \
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

"${PYTHON_BIN}" src/main.py \
    --task_id "${TASK_ID}" \
    --job_id "${SLURM_JOB_ID}" \
    --input_dir "${WORKDIR}/data/input" \
    --identity_a_dir "${WORKDIR}/data/identity_person_A" \
    --identity_b_dir "${WORKDIR}/data/identity_person_B" \
    --output_dir "${OUTPUT_BASE}" \
    --num_inference_steps 28 \
    --guidance_scale 3.5 \
    --ssim_threshold 0.65 \
    --face_fidelity_weight 0.85 \
    --enable_face_refinement

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