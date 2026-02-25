#!/bin/bash
#SBATCH --job-name=job            
#SBATCH --output=./log/output_%j.txt        
#SBATCH --error=./log/error_%j.txt          
#SBATCH --cpus-per-task=8            
#SBATCH --mem=100G                     
#SBATCH --time=12:00:00              

RETRI_MODEL="multi-qa-MiniLM-L6-cos-v1"
# RETRI_MODEL="all-mpnet-base-v2"
# RETRI_MODEL="BAAI/bge-base-en-v1.5"
export CUDA_VISIBLE_DEVICES=1

for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
  for tau in -0.1 0.05 0.10 0.15 0.20 0.25; do
    python -u main_batch.py \
    --step evaluate \
    --topk 10 \
    --B 3 \
    --F 2 \
    --alpha $alpha \
    --tau $tau \
    --slm_class no \
    --mode RF-Mem \
    --api yes\
    --model_path ./LLM_src/Qwen3-8B-base \
    --retri_name $RETRI_MODEL \
    --question_path ./data/questions_32k.csv \
    --context_path ./data/shared_contexts_32k.jsonl \
    --result_path ./results/eval_results_32k_gpt-4.1-mini_rag_adapt.csv >> new_expr_alpha_tau.log
  done
done
