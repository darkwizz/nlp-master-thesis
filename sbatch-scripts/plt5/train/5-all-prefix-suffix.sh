#!/bin/bash
#SBATCH -A plgnlp-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem 32GB
#SBATCH -t 24:00:00

module load python/3.9.6-gcccore-11.2.0
module load cuda/11.6.0
module load cudnn/8.4.1.50-cuda-11.6.0
source $SCRATCH/venv/bin/activate
cd $SCRATCH/t5-gpt2-scripts
export TRANSFORMERS_CACHE="$SCRATCH/transformers-cache"

python main.py -n plt5 -r baseline -b ./data-iterations/5-all-prefix-suffix -t allegro/plt5-large -m allegro/plt5-large --results-dir ./5-plt5-large-results --save-pretrained -q "115" -a "160" --test-max-length "30" --test-batch-size "256" -M ./plt5-large-checkpoints/baseline/5-trained-model -o ./training-log/5-plt-large