#!/bin/bash
#SBATCH -A plgnlp-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem 32GB
#SBATCH -t 12:00:00

module load python/3.8.6-gcccore-10.2.0
module load cuda/11.6.0
source $SCRATCH/venv/bin/activate
cd $SCRATCH/t5-gpt2-scripts
export TRANSFORMERS_CACHE="$SCRATCH/transformers-cache"

python main.py -n plt5 -r baseline -b ./data-iterations/0-baseline -t azwierzc/plt5-large-poquad -m azwierzc/plt5-large-poquad --results-dir ./0-plt5-poquad-results --save-pretrained -q "45" -a "16" --test-max-length "15" --test-batch-size "256" -M ./plt5-poquad/baseline/trained-model