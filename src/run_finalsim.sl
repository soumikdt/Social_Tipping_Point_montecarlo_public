#!/bin/bash -l
#SBATCH -J montecarlo
#SBATCH -A g98-2086
#SBATCH -p topola
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G            # 5 GB per core × 24
#SBATCH --time=2-00:00        # ← 48  h   (correct format)
#SBATCH --output=montecarlo.%j.log
#SBATCH --hint=nomultithread  # optional but nice

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

ulimit -t unlimited

cd ~/montecarlosim
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Monte Carlo run…"

python -u montecarlo.py \
  || { echo "[$(date '+%Y-%m-%d %H:%M:%S')] montecarlo.py failed"; exit 1; }

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Monte Carlo run complete."
