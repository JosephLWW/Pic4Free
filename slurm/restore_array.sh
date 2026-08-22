#!/bin/bash
#SBATCH --job-name=surfai_array
#SBATCH --output=slurm_logs/slurm_%A_%a.out
#SBATCH --error=slurm_logs/slurm_%A_%a.err
#SBATCH --partition=gpu_h100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8               # 8 cores para OpenCV, SSIM y dataloading
#SBATCH --mem=80GB                      # Memoria del nodo
#SBATCH --gres=gpu:1                    # 1 GPU H100 por tarea de array
#SBATCH --time=0:30:00                  # Tiempo máximo asignado por imagen
#SBATCH --array=0-97%16                 # 98 imágenes (0-97), hasta 16 tareas concurrentes

# ==============================================================================
# SurfAI - HPC Slurm Array Execution Script (State of the Art - August 2026)
# Pipeline: SSIM Masking -> InsightFace Multi-ID -> FLUX.2-Fill -> Post-Upscale
# ==============================================================================

echo "=============================================================================="
echo "Iniciando Tarea Slurm Array SurfAI: Job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} | Task ${SLURM_ARRAY_TASK_ID:-0}"
echo "Fecha y Hora: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="

# 1. Limpieza de entorno y carga de módulos HPC
unset PYTHONPATH
module purge
module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-14.2

# 2. Resolución de directorios de trabajo y almacenamiento
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${WORKDIR}/.venv"

cd "${WORKDIR}"

# Configuración de caches para evitar saturar el home o cuotas de red
export HF_HOME="${WORKDIR}/.cache/huggingface"
export TORCH_HOME="${WORKDIR}/.cache/torch"
export INSIGHTFACE_HOME="${WORKDIR}/.cache/insightface"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${INSIGHTFACE_HOME}" slurm_logs

# 3. Activación y verificación del entorno virtual
if [ ! -d "${VENV}" ]; then
    echo "[SurfAI] Entorno virtual no encontrado en ${VENV}. Creando entorno..."
    python3 -m venv "${VENV}"
    source "${VENV}/bin/activate"
    echo "[SurfAI] Instalando dependencias de producción..."
    pip install --quiet --upgrade pip setuptools wheel
    pip install --quiet -r "${WORKDIR}/requirements.txt"
else
    source "${VENV}/bin/activate"
fi

# 4. Creación del directorio de salida específico para este JOB ID
JOB_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
OUTPUT_BASE="${WORKDIR}/data/output/run_${JOB_ID}"

mkdir -p "${OUTPUT_BASE}/restored"
mkdir -p "${OUTPUT_BASE}/masks"
mkdir -p "${OUTPUT_BASE}/intermediates"
mkdir -p "${OUTPUT_BASE}/metrics"
mkdir -p "${OUTPUT_BASE}/logs"

# 5. Diagnóstico y telemetría de hardware
echo "=============================================================================="
echo "DETALLES DEL ENTORNO DE EJECUCIÓN"
echo "=============================================================================="
echo "Array Job ID:         ${JOB_ID}"
echo "Array Task ID:        ${TASK_ID}"
echo "Nodo de Ejecución:    ${SLURMD_NODENAME}"
echo "Ruta Python:          $(which python)"
echo "Versión Python:       $(python --version)"
echo "Dispositivo CUDA:     $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "ERROR: CUDA no disponible")')"
echo "Memoria GPU Total:    $(python -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB" if torch.cuda.is_available() else "N/A")')"
echo "Directorio Salida:    ${OUTPUT_BASE}"
echo "=============================================================================="

# 6. Invocación del pipeline de restauración de SurfAI pasando el TASK_ID
echo "[SurfAI] Despachando tarea ${TASK_ID} para procesamiento individual..."

python3 src/main.py \
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
    --enable_face_refinement

EXIT_CODE=$?

# 7. Copia sincronizada de logs a la carpeta de salida del Job
if [ -f "slurm_logs/slurm_${JOB_ID}_${TASK_ID}.out" ]; then
    cp "slurm_logs/slurm_${JOB_ID}_${TASK_ID}.out" "${OUTPUT_BASE}/logs/" 2>/dev/null || true
fi
if [ -f "slurm_logs/slurm_${JOB_ID}_${TASK_ID}.err" ]; then
    cp "slurm_logs/slurm_${JOB_ID}_${TASK_ID}.err" "${OUTPUT_BASE}/logs/" 2>/dev/null || true
fi

echo "=============================================================================="
echo "Tarea Slurm Array completada con código de salida: ${EXIT_CODE}"
echo "Resultados disponibles en: ${OUTPUT_BASE}"
echo "=============================================================================="

exit ${EXIT_CODE}
