#!/bin/bash
#SBATCH --gres=gpu:1
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

echo "Folder of Checkpoints: $1"
echo "Config: $2"

python /vol/bitbucket/es1519/NLPClassification_01/roberta_model/model_eval/evaluate.py --test_csv /vol/bitbucket/es1519/NLPClassification_01/roberta_model/DontPatronizeMe/csv_files/dev.csv --folder $1 --config $2

# Param 1: folder (e.g. /vol/bitbucket/es1519/NLPClassification_01/roberta_model/saved/ROBERTA/lightning_logs/version_69835/checkpoints/converted)
# Param 2: config (e.g. /vol/bitbucket/es1519/NLPClassification_01/roberta_model/configs/RoBERTa_config.json)