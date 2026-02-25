
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
from tabulate import tabulate
from collections import Counter
import numpy as np
import time

from openai import OpenAI

from utils import count_csv_rows,load_rows_with_context,class_prompt, compute_eval_tables,decide_strategy_with_probe
from llm_model.qwen3_8b import VLLMQwen
from llm_model.api_model import APILLMClient
from retri_mdoel.EmbdRetri import EmbeddingRetrievaler


INSTRUCTION = (
    "You are a multiple-choice answer generator. "
    "You MUST respond with exactly one option in the form of (a), (b), (c), or (d). "
    "Do not include any explanation, reasoning, or extra text. "
    "Do not output anything else besides the chosen option. "
    "If the correct answer is unknown, make the best guess and still only respond in that format. "
    "If your output does not exactly match one of (a), (b), (c), or (d), your answer will be considered incorrect."
)
BATCH_SIZE = 8

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
    question_path = cmd_args.question_path
    context_path = cmd_args.context_path
    result_path = cmd_args.result_path

    if os.path.exists(result_path):
        os.remove(result_path)

    all_errors = []
    total_rows = count_csv_rows(question_path)
    all_prompt = []
    all_history = []
    all_correct_answer = []
    output_json = []
    cache_path = cmd_args.question_path.split('/')[-1].split('.csv')[0] + '_' + cmd_args.retri_name
    retriever = EmbeddingRetrievaler(
                    model_name=cmd_args.retri_name,
                    top_k=5,
                    cache_path = cache_path
                )
    
    hist_num_ls = []
    fast_or_slow_ls = []
    ent_ls = []
    time_ls = []
    mean_score = []
    print("B:", cmd_args.B, "F:", cmd_args.F,"alpha:", cmd_args.alpha,"tau:", cmd_args.tau,"score_th:", cmd_args.score_th,"score_bt:", cmd_args.score_bt)

    for row_data, context in tqdm(load_rows_with_context(question_path, context_path), total=total_rows):
        try:
            persona_id = row_data["persona_id"]
            question_id = row_data["question_id"]
            question_type = row_data["question_type"]
            topic = row_data["topic"]
            context_length_in_tokens = row_data["context_length_in_tokens"]
            context_length_in_letters = row_data["context_length_in_letters"]
            distance_to_ref_in_blocks = row_data["distance_to_ref_in_blocks"]
            distance_to_ref_in_tokens = row_data["distance_to_ref_in_tokens"]
            num_irrelevant_tokens = row_data["num_irrelevant_tokens"]
            distance_to_ref_proportion_in_context = row_data["distance_to_ref_proportion_in_context"]
            question = row_data["user_question_or_message"]
            correct_answer = row_data["correct_answer"]
            all_options = row_data["all_options"]
            shared_context_id = row_data["shared_context_id"]
            end_index_in_shared_context = row_data["end_index_in_shared_context"]
            retriever.build_from_history(context)
            context = retriever.docs

            if cmd_args.mode == 'RF-Mem':
                mode, diag = decide_strategy_with_probe(retriever, question, tau=0.0, probe_k=10, ent_th=float(cmd_args.tau), score_th=0.60, score_bt=0.30)
                ent = diag['entropy']
                ent_ls.append(ent)
                mean_score.append(diag['mean_score'])
                fast_or_slow = {"text": mode}
            elif cmd_args.mode == 'reco':
                fast_or_slow = {"text":"slow"}
            elif cmd_args.mode == 'fami':
                fast_or_slow = {"text":"fast"}
            fast_or_slow_ls.append(fast_or_slow["text"])
            if cmd_args.rag == 'yes':
                if 'fast' in fast_or_slow["text"]:
                    t4 = time.perf_counter()
                    hits = retriever.retrieve(question, top_k=cmd_args.topk, tau=0.3)
                    t5 = time.perf_counter()
                    time_spend = t5-t4
                    context = retriever.to_messages(hits)
                elif 'slow' in fast_or_slow["text"]:
                    t4 = time.perf_counter()
                    hits = retriever.retrieve_pyramid_v2(question,out_k=cmd_args.topk, tau=0.3, depth=cmd_args.topk, beam_width=cmd_args.B, fanout=cmd_args.F, expansion="mean_group", alpha=cmd_args.alpha, mmr_lambda=0.95)
                    t5 = time.perf_counter()
                    time_spend = t5-t4
                    context = retriever.to_messages(hits)
                else:
                    hits = retriever.retrieve_pyramid_v2(question,out_k=cmd_args.topk, tau=0.0,depth=5, beam_width=4, fanout=3, expansion="mean_group", alpha=0.8, weight_temp=20.0, mmr_lambda=0.8)
                    context = retriever.to_messages(hits)
                time_ls.append(time_spend)
                hist_num_ls.append(len(hits))
            prompt = question + '\n\n' + INSTRUCTION + '\n\n' + all_options
            all_prompt.append(prompt)
            all_history.append(context)
            all_correct_answer.append(correct_answer)
            output_json.append({
                "score":'',
                "persona_id": persona_id,
                "question_id": question_id,
                "question": question,
                "question_type": question_type,
                "topic": topic,
                "context_length_in_tokens": context_length_in_tokens,
                "context_length_in_letters": context_length_in_letters,
                "distance_to_ref_in_blocks": distance_to_ref_in_blocks,
                "distance_to_ref_in_tokens": distance_to_ref_in_tokens,
                "num_irrelevant_tokens": num_irrelevant_tokens,
                "distance_to_ref_proportion_in_context": distance_to_ref_proportion_in_context,
                "model_response": '',
                "len_of_model_response": 0,
                "predicted_answer": '',
                "correct_answer": correct_answer,
            })
        except Exception as e:
            print(f"Error: {e}")
            all_errors.append({
                "persona_id": row_data["persona_id"],
                "question_id": row_data["question_id"],
                "error": str(e)
            })
            continue
    print('Process prompt finished, total prompts:', len(all_prompt))
    
    if cmd_args.rag == 'yes':
        print(hist_num_ls)
        print(fast_or_slow_ls)
        print(Counter(hist_num_ls))
        print(Counter(fast_or_slow_ls))
        time_arr = np.array(time_ls)
        mean_val = np.mean(time_arr)
        var_val = np.var(time_arr, ddof=0)
        print("time mean:", mean_val)
        print("time std:", var_val)
        print(sum(hist_num_ls)/len(hist_num_ls))
    if len(time_ls)>0:
        q25, q50, q75 = np.percentile(time_arr, [25, 50, 75])
        print("25% :", q25)
        print("50% :", q50)
        print("75% :", q75)

    if cmd_args.api == 'yes':
        token_ls = []
        model_response = await llm.aquery(
            prompt=all_prompt,
            system=None,
            history = all_history,
            max_tokens=32,
            temperature=0.0,
            top_p=0.9,
            seed=42,
            max_concurrency = 200
        )
        raws = model_response["raw"]
        model_response = model_response["text"]
        for item in raws:
            try:
                tokens = item["usage"]["prompt_tokens"]
                token_ls.append(tokens)
            except:
                continue
        print(sum(token_ls)/len(token_ls))
        for i, response in enumerate(model_response):
            score, predicted_answer = llm.extract_answer(response, all_correct_answer[i])
            if verbose:
                print(f"Question: {all_prompt[i]}")
                print(f"Predicted Answer: {predicted_answer}")
                print(f"Correct Answer: {all_options[i]}")
                print(f"Score: {score}")
            output_json[i]['score'] = score
            output_json[i]['model_response'] = response
            output_json[i]['len_of_model_response'] = len(response)
            output_json[i]['predicted_answer'] = predicted_answer
    else:
        for i in tqdm(range(0, len(all_prompt), BATCH_SIZE)):
            batch_prompts = all_prompt[i:i+BATCH_SIZE]
            batch_histories = all_history[i:i+BATCH_SIZE] 
            # Send the query to the LLM
            model_response = llm.query(
                prompt=batch_prompts,
                system=None,
                history = batch_histories,
                max_tokens=32,
                temperature=0.0,
                top_p=0.9,
                seed=42,
            )
            model_response = model_response["text"]
            for j, response in enumerate(model_response):
                score, predicted_answer = llm.extract_answer(response, all_correct_answer[i+j])
                if verbose:
                    print(f"Question: {all_prompt[i+j]}")
                    print(f"Predicted Answer: {predicted_answer}")
                    print(f"Correct Answer: {all_options[j]}")
                    print(f"Score: {score}")
                output_json[i+j]['score'] = score
                output_json[i+j]['model_response'] = response
                output_json[i+j]['len_of_model_response'] = len(response)
                output_json[i+j]['predicted_answer'] = predicted_answer
        

    if all_errors:
        for error in all_errors:
            print(f"Error for persona_id {error['persona_id']} and question_id {error['question_id']}: {error['error']}")
    df = pd.DataFrame(output_json)

    # 保存为 CSV
    df.to_csv(result_path, index=False, encoding="utf-8-sig")
    
    ooverall_df, topics_df, qtypes_df = compute_eval_tables(df)
    pd.options.display.float_format = "{:.4f}".format

    print("=== Overall Results ===")
    print(tabulate(ooverall_df, headers='keys', tablefmt='grid', floatfmt=".4f"))
    print("\n")

    print("=== Topics Results ===")
    topics_df_sorted = topics_df.sort_values(["Accuracy","Samples"], ascending=[False, False])
    print(tabulate(topics_df_sorted, headers='keys', tablefmt='grid', floatfmt=".4f"))
    print("\n")

    print("=== Question Types Results ===")
    qtypes_df_sorted = qtypes_df.sort_values(["Accuracy","Samples"], ascending=[False, False])
    print(tabulate(qtypes_df_sorted, headers='keys', tablefmt='grid', floatfmt=".4f"))


    print(f"Save as: {result_path}")
    json_file = result_path.replace('.csv','.json')
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)

    print(f"Save finished: {json_file}")

if __name__ == "__main__":

    print("CUDA available:", torch.cuda.is_available())

    parser = argparse.ArgumentParser(description='Command line arguments')

    parser.add_argument('--api', type=str, default="no", help='if using api')
    parser.add_argument('--model_path', type=str, default="./LLM_src/Qwen3-8B-base", help='Set LLM model.')
    parser.add_argument('--retri_name', type=str, default="multi-qa-MiniLM-L6-cos-v1", help='Set LLM model.')

    parser.add_argument('--step', type=str, default='prepare', help='Step to run: prepare or evaluate')
    parser.add_argument('--mode', type=str, default='RF-Mem', help='if using rag strategy')
    parser.add_argument('--B', type=int, default=3, help='B')
    parser.add_argument('--F', type=int, default=3, help='F')
    parser.add_argument('--alpha', type=float, default=0.8, help='alpha')
    parser.add_argument('--tau', type=float, default=0.20, help='tau')
    parser.add_argument('--score_th', type=float, default=0.6, help='tau')
    parser.add_argument('--score_bt', type=float, default=0.2, help='tau')

    parser.add_argument('--topk', type=int, default=10, help='if using rag strategy')

    parser.add_argument('--slm_class', type=str, default='yes', help='if using rag strategy')

    parser.add_argument('--token_path', type=str, default='./apitokens', help='Path to the API tokens')
    parser.add_argument('--clean', dest='clean', action='store_true', help='Remove existing csv and json files and start clean')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')

    parser.add_argument('--question_path', type=str, default='./data/questions_128k.csv', help='Path to the questions CSV file')
    parser.add_argument('--context_path', type=str, default='./data/shared_contexts_128k.jsonl', help='Path to the contexts JSONL file')
    parser.add_argument('--result_path', type=str, default='./data/eval_results.csv', help='Path to save the results CSV file')

    cmd_args = parser.parse_args()
    if cmd_args.api == 'yes':
        asyncio.run(main(cmd_args, verbose=cmd_args.verbose))
    else:
        main(cmd_args, verbose=cmd_args.verbose)
