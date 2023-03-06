#!/bin/bash
#SBATCH --gres=gpu:1
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

srun python /vol/bitbucket/es1519/NLPClassification_01/roberta_model/convert_weights.py --folder $1 --device "cuda:0"

# Param 1: folder location (e.g. /vol/bitbucket/es1519/NLPClassification_01/roberta_model/saved/ROBERTA/lightning_logs/version_69896/checkpoints)