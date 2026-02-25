#!/bin/bash
#SBATCH --job-name=my_job         
#SBATCH --output=./output_%j.txt        
#SBATCH --error=./error_%j.txt          
#SBATCH --cpus-per-task=4             
#SBATCH --mem=100G                    
#SBATCH --time=12:00:00               
export PYTHONIOENCODING=utf-8

export CUDA_VISIBLE_DEVICES=1
# retrieval_mode can in (fami, reco, RF-Mem)
python -u ./src/retrieval/RF_emem.py \
    --retrieval_model_name="multi-qa-MiniLM-L6-cos-v1"  \
    --data_type='s' \
    --retrieval_mode='RF-Mem' \
    --tau="0.10" \
    --alpha=0.7 \
    --B='4' \
    --F='1' 