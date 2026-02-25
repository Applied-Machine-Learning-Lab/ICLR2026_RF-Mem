
import torch
import os
import json
import random
import argparse
import yaml
import re
import torch
from tqdm import tqdm
import csv
import json
import pandas as pd
import asyncio, aiohttp
from collections import Counter
import numpy as np
import time

from openai import OpenAI

from utils import count_csv_rows,load_rows_with_context,class_prompt,decide_strategy_with_probe
from llm_model.qwen3_8b import VLLMQwen
from llm_model.api_model import APILLMClient
from retri_mdoel.EmbdRetri import EmbeddingRetrievaler

from eval import eval


INSTRUCTION = (
    "You are a multiple-choice answer generator. "
    "You MUST respond with exactly one option in the form of (a), (b), (c), or (d). "
    "Do not include any explanation, reasoning, or extra text. "
    "Do not output anything else besides the chosen option. "
    "If the correct answer is unknown, make the best guess and still only respond in that format. "
    "If your output does not exactly match one of (a), (b), (c), or (d), your answer will be considered incorrect."
)
BATCH_SIZE = 8

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    with open(file_path, "w") as f:
        if isinstance(data, dict) or isinstance(data, list):
            json.dump(data, f, indent=4)
        elif isinstance(data, str):
            try:
                json.loads(data)
                f.write(data)
            except json.JSONDecodeError:
                raise ValueError("Provided string is not valid JSON")
        else:
            raise TypeError("Data must be a list/dictionary or a JSON string")


def build_rag_corpus(community_path,noise):
    conversation_data_all = load_json(os.path.join(community_path, 'private_data', f"noise_{noise}", "conversation_data_all.json"))
    user_ai_interaction_data_all = load_json(os.path.join(community_path, 'private_data', f"noise_{noise}", "user_ai_interaction_data_all.json"))
    purchase_history_data_all = load_json(os.path.join(community_path, 'private_data', f"noise_{noise}", "purchase_history_data_all.json"))
    conversation_data_all_lookup = {entry["Name"]: entry["Data"] for entry in conversation_data_all}
    user_ai_interaction_data_all_lookup = {entry["Name"]: entry["Data"] for entry in user_ai_interaction_data_all}
    purchase_history_data_all_lookup = {entry["Name"]: entry["Data"] for entry in purchase_history_data_all}
    return conversation_data_all_lookup, user_ai_interaction_data_all_lookup,purchase_history_data_all_lookup

async def main(cmd_args, verbose=False):
   
    if cmd_args.api == 'yes':
        llm = APILLMClient(
                api_key='',
                model="gpt-4.1-mini",
                base_url="https://api.v3.cm/v1/",  
            )
    else:
        llm = VLLMQwen(
            model=cmd_args.model_path,
            dtype="auto",                 
            tensor_parallel_size=1,       
            gpu_memory_utilization=0.8,
        )
    test_result_dir = os.path.join(cmd_args.log_dir, cmd_args.save_dir)
    save_path = os.path.join(test_result_dir, f"results_all_community_chunk_{cmd_args.num_chunks}")
    os.makedirs(save_path, exist_ok=True)
    results_all_community = []
    results_all_community.append({"Model": "gpt-4.1-mini+"+cmd_args.retri_name, "Results": []})
    retriever = EmbeddingRetrievaler(
                    model_name=cmd_args.retri_name,
                    top_k=cmd_args.num_chunks,
                )


    hist_num_ls = []
    fast_or_slow_ls = []
    retri_ls = []
    ent_ls = []
    time_ls = []
    for community_id in [0,1]:
        community_path = f"./eval_data/eval_data_v1/synthetic_data/community_{community_id}"
        noise = 0.0
        qa_gt_context_all = load_json(community_path +"/eval_info/qa_gt_context_all_noise_0.0.json")
        eval_info_all = load_json(community_path +"/eval_info/eval_info_all.json")
        qa_gt_context_all_lookup = {entry["q_id"]: entry for entry in qa_gt_context_all}
        conver_all, user_ai_all,purchase_his_all = build_rag_corpus(community_path,noise)
        for eval_info_dict in eval_info_all[:3]:
            name = eval_info_dict["Name"]
            eval_info = eval_info_dict["Eval_Info"]
            conversation_data = conver_all[name]
            user_ai_interaction_data = user_ai_all[name]
            purchase_history_data = purchase_his_all[name]
            documents_dict = {"conversation_data": conversation_data, "user_ai_interaction_data": user_ai_interaction_data, "purchase_history_data": purchase_history_data}
            retriever.build_from_history(documents_dict)
            QA = eval_info["qa"]
            for i,entry in enumerate(QA):
                question = entry['question']
            
                if cmd_args.rag_adapt == 'RF-Mem':
                    mode, diag = decide_strategy_with_probe(retriever, question, probe_k=cmd_args.num_chunks, ent_th=cmd_args.tau)
                    ent = diag['entropy']
                    ent_ls.append(ent)
                    fast_or_slow = {"text": mode}
                elif cmd_args.rag_adapt == 'reco':
                    fast_or_slow = {"text":"slow"}
                else:
                    fast_or_slow = {"text":"fast"}
                fast_or_slow_ls.append(fast_or_slow["text"])
                if cmd_args.rag == 'yes':
                    if 'fast' in fast_or_slow["text"]:
                        t4 = time.perf_counter()
                        hits = retriever.retrieve(question, tau=0.0, top_k=cmd_args.num_chunks)
                        ids, context = retriever.to_messages(hits)
                        t5 = time.perf_counter()
                        time_spend = t5-t4
                    elif 'slow' in fast_or_slow["text"]:
                        t4 = time.perf_counter()
                        hits = retriever.retrieve_pyramid_v2(question,out_k=cmd_args.num_chunks, tau=0.0, depth=10, beam_width=cmd_args.B, fanout=cmd_args.F, expansion="mean_group", alpha=cmd_args.alpha)
                        ids, context = retriever.to_messages(hits)
                        t5 = time.perf_counter()
                        time_spend = t5-t4
                    else:
                        hits = retriever.retrieve_pyramid_v2(question,out_k=cmd_args.num_chunks, tau=0.0, beam_width=cmd_args.B, fanout=cmd_args.F, expansion="mean_group",alpha=cmd_args.alpha)
                        ids, context = retriever.to_messages(hits)
                    results_all_community[-1]["Results"].append(
                        {"q_id": entry["q_id"],
                        "question": entry["question"],
                        "answer": entry["answer"],
                        "prediction":entry,
                        "type": entry["type"],
                        "difficulty": entry["difficulty"],
                        "retrieved_segment_ids": list(ids),
                        "ground_truth_segment_ids": qa_gt_context_all_lookup[entry["q_id"]]["segment_id"],
                        }
                    )
                time_ls.append(time_spend)
    if cmd_args.rag == 'yes':
        print(Counter(fast_or_slow_ls))
        time_arr = np.array(time_ls)
        mean_val = np.mean(time_arr)
        var_val = np.var(time_arr, ddof=0)
        print("time mean:", mean_val)
        print("time std:", var_val)
    save_json(results_all_community, os.path.join(save_path, f"result_noise_{noise}.json"))
    eval(cmd_args)
if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')

    parser.add_argument('--api', type=str, default="no", help='if using api')
    parser.add_argument('--model_path', type=str, default="./LLM_src/Qwen3-8B-base", help='Set LLM model.')
    parser.add_argument("--data_dir", type=str, default="eval_data", help="Path to personabench datasets")
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to save eval results.")
    parser.add_argument("--save_dir", type=str, default="eval_data_v1")
    parser.add_argument("--num_chunks", type=int, default=5, help="Number of chunks for retrieval tasks")

    parser.add_argument("--retri_name", type=str, default="multi-qa-MiniLM-L6-cos-v1", help='Set LLM model.')


    parser.add_argument('--step', type=str, default='prepare', help='Step to run: prepare or evaluate')
    parser.add_argument('--rag', type=str, default='yes', help='if using rag strategy')

    parser.add_argument('--B', type=int, default=3, help='B')
    parser.add_argument('--F', type=int, default=3, help='F')
    parser.add_argument('--alpha', type=float, default=0.8, help='alpha')
    parser.add_argument('--tau', type=float, default=0.20, help='tau')
    parser.add_argument('--score_th', type=float, default=0.60, help='tau')
    parser.add_argument('--score_bt', type=float, default=0.20, help='tau')

    parser.add_argument('--slm_class', type=str, default='no', help='if using rag strategy')
    parser.add_argument('--rag_adapt', type=str, default='RF-Mem', help='if using rag strategy')

    parser.add_argument('--token_path', type=str, default='./apitokens', help='Path to the API tokens')
    parser.add_argument('--clean', dest='clean', action='store_true', help='Remove existing csv and json files and start clean')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')

    cmd_args = parser.parse_args()
    if cmd_args.api == 'yes':
        asyncio.run(main(cmd_args, verbose=cmd_args.verbose))
    else:
        main(cmd_args, verbose=cmd_args.verbose)
