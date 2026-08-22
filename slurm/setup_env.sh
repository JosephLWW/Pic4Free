#!/bin/bash
#SBATCH --job-name=surfai_setup_env
#SBATCH --output=slurm_logs/setup_%j.out
#SBATCH --error=slurm_logs/setup_%j.err
#SBATCH --partition=gpu_h100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00

# ==============================================================================
# SurfAI - HPC Environment Provisioning & Pre-Caching Script
# Descarga y precalienta pesos de FLUX 2, InsightFace y paquetes en .venv
# ==============================================================================

echo "=============================================================================="
echo "Iniciando Aprovisionamiento de Entorno SurfAI en Nodo: ${SLURMD_NODENAME}"
echo "=============================================================================="

unset PYTHONPATH
module purge
module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-14.2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${WORKDIR}/.venv"

cd "${WORKDIR}"

export HF_HOME="${WORKDIR}/.cache/huggingface"
export TORCH_HOME="${WORKDIR}/.cache/torch"
export INSIGHTFACE_HOME="${WORKDIR}/.cache/insightface"

mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${INSIGHTFACE_HOME}" slurm_logs

if [ ! -d "${VENV}" ]; then
    echo "[Setup] Creando nuevo entorno virtual en ${VENV}..."
    python3 -m venv "${VENV}"
fi

source "${VENV}/bin/activate"

echo "[Setup] Actualizando pip e instalando dependencias..."
pip install --upgrade pip setuptools wheel
pip install -r "${WORKDIR}/requirements.txt"

echo "[Setup] Verificando compatibilidad con PyTorch CUDA 12.8..."
python3 -c '
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name:     {torch.cuda.get_device_name(0)}")
    print(f"Capability:      {torch.cuda.get_device_capability(0)}")
'

echo "=============================================================================="
echo "Aprovisionamiento de entorno completado exitosamente."
echo "=============================================================================="
