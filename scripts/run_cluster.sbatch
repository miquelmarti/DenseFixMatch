#!/usr/bin/env bash
#SBATCH -A <account> -p <partition>
#SBATCH --output=<path-to-log-dir>/jobs/%J.out
#SBATCH --error=<path-to-log-dir>/jobs/%J.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<user-email>
#SBATCH --gpus-per-node=<gpu-cfg>
#SBATCH -t 1-00:00:00

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
echo ${CUDA_VISIBLE_DEVICES}
source $HOME/.bashrc
conda activate $1
echo "Using conda env: $CONDA_PREFIX"
echo "Current dir: ${PWD}"
export PYTHONPATH=.
export HYDRA_SYSTEM_CONFIG="cluster"
ln -s <path-to-datasets-dir> $TMPDIR/datasets
ln -s <path-to-output-dir> $TMPDIR/output
eval $2
