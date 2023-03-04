#!/bin/bash
#SBATCH --gres=gpu:1
export PATH=/vol/bitbucket/es1519/myvenv/bin/:$PATH
source activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

echo "Checkpoint: $1"
echo "Test File: $2"
echo "Config: $3"

python /vol/bitbucket/es1519/NLPClassification_01/roberta_model/model_eval/evaluate.py --checkpoint $1 --test_csv $2 --config $3

# Param 1: checkpoint (e.g. /vol/bitbucket/es1519/NLPClassification_01/roberta_model/saved/ROBERTA/lightning_logs/version_69677/checkpoints/epoch=0-step=122_converted.ckpt)
# Param 2: test_csv (e.g. /vol/bitbucket/es1519/NLPClassification_01/roberta_model/DontPatronizeMe/csv_files/dontpatronizeme_pcl_test.csv)
# Param 3: config (e.g. /vol/bitbucket/es1519/NLPClassification_01/roberta_model/configs/RoBERTa_config.json)