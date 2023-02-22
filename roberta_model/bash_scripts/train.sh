#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=es1519 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

echo "Running $2 epochs."
srun python /vol/bitbucket/es1519/NLPClassification_01/roberta_model/train.py --config $1 -e $2

# Param 1: config (e.g. /vol/bitbucket/es1519/NLPClassification_01/roberta_model/configs/RoBERTa_config.json)
# Param 2: number of epochs