#!/bin/bash
#SBATCH --job-name=Pic4Free_setup_env
#SBATCH --output=slurm_logs/setup_%j.out
#SBATCH --error=slurm_logs/setup_%j.err
#SBATCH --partition=gpu_h100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00

set -euo pipefail

unset PYTHONPATH
unset VIRTUAL_ENV
hash -r

module purge
module load devel/cuda/12.8
module load devel/python/3.12.3-gnu-14.2

# Slurm debe recibir este directorio como directorio de envío.
WORKDIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR no está definido}"
WORKDIR="$(cd "${WORKDIR}" && pwd)"

if [[ ! -f "${WORKDIR}/requirements.txt" ]]; then
    echo "ERROR: requirements.txt no existe en ${WORKDIR}" >&2
    echo "Envíe el job desde la raíz del repositorio." >&2
    exit 1
fi

cd "${WORKDIR}"

VENV="${WORKDIR}/.venv"

if [[ ! -x "${VENV}/bin/python" ]]; then
    python3 -m venv "${VENV}"
fi

PYTHON="${VENV}/bin/python"
PIP="${VENV}/bin/pip"

echo "Python: $("${PYTHON}" -c 'import sys; print(sys.executable)')"
echo "pip:    $("${PIP}" --version)"

"${PYTHON}" -m pip install --upgrade pip setuptools wheel
"${PYTHON}" -m pip install -r "${WORKDIR}/requirements.txt"

"${PYTHON}" -c '
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
