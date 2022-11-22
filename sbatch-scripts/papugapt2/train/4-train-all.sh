#!/bin/bash
#SBATCH -A plgnlp-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem 32GB
#SBATCH -t 16:00:00

module load python/3.9.6-gcccore-11.2.0
module load cuda/11.6.0
module load cudnn/8.4.1.50-cuda-11.6.0
source $SCRATCH/venv/bin/activate
cd $SCRATCH/t5-gpt2-scripts
export TRANSFORMERS_CACHE="$SCRATCH/transformers-cache"

python main.py -n papugapt2 -r baseline -b ./data-iterations/4-all -t flax-community/papuGaPT2-large -m flax-community/papuGaPT2-large --results-dir ./4-papugapt2-large-results --save-pretrained -q "46" -a "30" --test-max-length "16" -M ./papugapt2-large-checkpoints/baseline/4-trained-model --test-batch-size 256 -o ./training-log/4-papugapt2