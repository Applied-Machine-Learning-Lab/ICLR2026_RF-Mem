#!/bin/bash
#SBATCH --job-name=my_job           
# SBATCH --output=./log/output_%j.txt      
# SBATCH --error=./log/error_%j.txt         
#SBATCH --cpus-per-task=8            
#SBATCH --mem=100G                    
#SBATCH --time=12:00:00               
LOG_DIR="test3-multi-qa-MiniLM-L6-cos-v1"
# LOG_DIR="test1-all-mpnet-base-v2"
# LOG_DIR="test1-bge-base-en-v1.5"
DATA_DIR="./eval_data/eval_data_v1"
SAVE_DIR="test2" 
RETRI_MODEL="multi-qa-MiniLM-L6-cos-v1"

export CUDA_VISIBLE_DEVICES=3


for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  for tau in 0.25 0.30; do
    echo "Running with alpha=$alpha, tau=$tau"
    python main_batch.py \
    --step evaluate \
    --retri_name $RETRI_MODEL \
    --log_dir $LOG_DIR \
    --data_dir $DATA_DIR \
    --save_dir $SAVE_DIR \
    --B 4 \
    --F 1 \
    --alpha $alpha \
    --tau $tau \
    --num_chunks 10 \
    --slm_class no \
    --rag_adapt RF-Mem \
    --api yes
  done
done
