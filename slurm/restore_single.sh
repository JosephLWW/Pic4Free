#!/bin/bash
#SBATCH --job-name=surfai_single
#SBATCH --output=slurm_logs/slurm_%j.out
#SBATCH --error=slurm_logs/slurm_%j.err
#SBATCH --partition=gpu_h100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8               # 8 cores para OpenCV, SSIM y dataloading
#SBATCH --mem=80GB                      # Memoria del nodo
#SBATCH --gres=gpu:1                    # 1 GPU H100
#SBATCH --time=0:30:00                  # Tiempo máximo asignado al job de depuración

# ==============================================================================
# SurfAI - HPC Slurm Single Image Execution Script (Interactive & Debugging)
# Permite pasar el índice de imagen como argumento: sbatch slurm/restore_single.sh 0
# ==============================================================================

# Argumento opcional: índice de tarea / imagen (por defecto 0)
TASK_ID="${1:-0}"

echo "=============================================================================="
echo "Iniciando Job Individual SurfAI: Job ${SLURM_JOB_ID} | Task ID: ${TASK_ID}"
echo "Fecha y Hora: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="

# 1. Limpieza de entorno y carga de módulos HPC
unset PYTHONPATH
module purge
module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-14.2

# 2. Resolución de directorios de trabajo
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${WORKDIR}/.venv"

cd "${WORKDIR}"

export HF_HOME="${WORKDIR}/.cache/huggingface"
export TORCH_HOME="${WORKDIR}/.cache/torch"
export INSIGHTFACE_HOME="${WORKDIR}/.cache/insightface"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${INSIGHTFACE_HOME}" slurm_logs

# 3. Activación del entorno virtual
if [ ! -d "${VENV}" ]; then
    echo "[SurfAI] Creando entorno virtual en ${VENV}..."
    python3 -m venv "${VENV}"
    source "${VENV}/bin/activate"
    pip install --quiet --upgrade pip setuptools wheel
    pip install --quiet -r "${WORKDIR}/requirements.txt"
else
    source "${VENV}/bin/activate"
fi

# 4. Creación del directorio de salida específico para este JOB ID
OUTPUT_BASE="${WORKDIR}/data/output/run_${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_BASE}/restored"
mkdir -p "${OUTPUT_BASE}/masks"
mkdir -p "${OUTPUT_BASE}/intermediates"
mkdir -p "${OUTPUT_BASE}/metrics"
mkdir -p "${OUTPUT_BASE}/logs"

# 5. Telemetría
echo "=============================================================================="
echo "Job ID:            ${SLURM_JOB_ID}"
echo "Task ID (Índice):  ${TASK_ID}"
echo "Nodo:              ${SLURMD_NODENAME}"
echo "Python:            $(which python)"
echo "GPU:               $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "ERROR: Sin GPU")')"
echo "Directorio Salida: ${OUTPUT_BASE}"
echo "=============================================================================="

# 6. Ejecución del pipeline para una sola imagen
python3 src/main.py \
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

# 7. Copia de logs
if [ -f "slurm_logs/slurm_${SLURM_JOB_ID}.out" ]; then
    cp "slurm_logs/slurm_${SLURM_JOB_ID}.out" "${OUTPUT_BASE}/logs/" 2>/dev/null || true
fi
if [ -f "slurm_logs/slurm_${SLURM_JOB_ID}.err" ]; then
    cp "slurm_logs/slurm_${SLURM_JOB_ID}.err" "${OUTPUT_BASE}/logs/" 2>/dev/null || true
fi

echo "=============================================================================="
echo "Ejecución individual finalizada con código de salida: ${EXIT_CODE}"
echo "=============================================================================="

exit ${EXIT_CODE}
