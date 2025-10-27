#!/bin/bash
#SBATCH --job-name=CNNDetEval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=A40,L40S,A100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# -------- shell hygiene --------
set -euo pipefail
umask 077
mkdir -p logs

# -------- print job header --------
echo "================= SLURM JOB START ================="
echo "Job:    $SLURM_JOB_NAME  (ID: $SLURM_JOB_ID)"
echo "Node:   ${SLURMD_NODENAME:-$(hostname)}"
echo "GPUs:   ${SLURM_GPUS_ON_NODE:-unknown}  (${SLURM_JOB_GPUS:-not-set})"
echo "Start:  $(date)"
echo "==================================================="

datetime="$(date '+%Y%m%d_%H%M%S')"
result_dir="results/${datetime}_CNNDet"
mkdir -p "${result_dir}"

DATA_ROOT="/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only"
OUTPUT_CSV="${result_dir}/predictions.csv"
MODEL_PATH="./weights/blur_jpg_prob0.5.pth"
MODEL_NAME="CNNDet_blur_jpg_prob0.5"
DONE_CSVS_DIR="results"

# -------- conda activate --------
source /home/infres/ziyliu-24/miniconda3/etc/profile.d/conda.sh
conda activate hifi37

srun python3 CNNDetEval.py \
    --data_root "$DATA_ROOT" \
    --output_csv "$OUTPUT_CSV" \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --done_csv_list "$DONE_CSVS_DIR"

EXIT_CODE=$?

echo "================== SLURM JOB END =================="
echo "End:   $(date)"
echo "Exit:  ${EXIT_CODE}"
echo "==================================================="
exit "${EXIT_CODE}"
