import os
import json
import random
import argparse
import yaml
import re
import torch
from tqdm import tqdm
import csv

import pandas as pd
import numpy as np


class_prompt = """You are an expert in information retrieval.  
I will give you a user question.  
Your task is to decide whether answering this question requires **fast retrieval** (quick, shallow search for direct facts) or **slow retrieval** (thorough, multi-step search and reasoning).  

Guidelines:  
- Choose **fast** if the question is short, fact-based, or requires a direct answer from a known source.  
- Choose **slow** if the question is complex, ambiguous, multi-hop, or requires integrating multiple sources and reasoning.  

Output only one word:  
- "fast" for fast retrieval  
- "slow" for slow retrieval  

Question: {user_question}"""
def build_jsonl_index(jsonl_path):
    """
    Scan the JSONL file once to build a mapping: {key: file_offset}.
    Assumes each line is a JSON object with a single key-value pair.
    """
    index = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            key = next(iter(json.loads(line).keys()))
            index[key] = offset
    return index

def count_csv_rows(csv_path):
    with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
        return sum(1 for _ in f) - 1  # Subtract 1 for header row
    
def load_context_by_id(jsonl_path, offset):
    """
    Seek to a known offset in the JSONL and load exactly that line.
    Returns the value associated with the single key in the JSON object.
    """
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        f.seek(offset)
        item = json.loads(f.readline())
        return next(iter(item.values()))
    
def load_rows_with_context(csv_path, jsonl_path):
    jsonl_index = build_jsonl_index(jsonl_path)

    with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        prev_sid = None
        prev_context = None

        for row_number, row in enumerate(reader, start=1):
            row_data = {}
            for column_name, value in row.items():
                row_data[column_name] = value

            sid = row_data["shared_context_id"]
            if sid != prev_sid:
                current_context = load_context_by_id(jsonl_path, jsonl_index[sid])

                prev_sid = sid
                prev_context = current_context
            else:
                current_context = prev_context

            yield row_data, current_context

def _norm_series(s: pd.Series) -> pd.Series:
    return pd.Series(s, dtype="string").fillna("").str.strip().str.casefold()

def _extract_option(s: pd.Series) -> pd.Series:
    s = _norm_series(s)
    m = s.str.extract(r"\(([a-d])\)")
    m[0] = m[0].fillna(s.str.extract(r"\b([a-d])\b")[0])
    return m[0].fillna("")

def compute_eval_tables(df: pd.DataFrame):

    if "score" in df.columns:
        is_corr = df["score"].astype(float).round().astype(int).astype(bool)
    elif {"predicted_answer", "correct_answer"}.issubset(df.columns):
        y_pred = _extract_option(df["predicted_answer"])
        y_true = _extract_option(df["correct_answer"])
        is_corr = (y_pred == y_true)
    else:
        raise ValueError("Need for ('predicted_answer','correct_answer') ")

    df = df.copy()
    df["is_correct"] = is_corr

    total = len(df)
    correct = int(df["is_correct"].sum())
    wrong = total - correct
    acc = correct / total if total else float("nan")
    overall_df = pd.DataFrame([{
        "Accuracy": acc, "Total samples": total, "Correct": correct, "Wrong": wrong
    }])

    if "topic" in df.columns:
        topics_df = (
            df.groupby("topic", dropna=False)
              .agg(Accuracy=("is_correct", "mean"),
                   Samples=("is_correct", "size"),
                   Correct=("is_correct", "sum"))
              .reset_index()
              .sort_values(["Accuracy","Samples"], ascending=[False, False])
        )
    else:
        topics_df = pd.DataFrame(columns=["topic","Accuracy","Samples","Correct"])

    if "question_type" in df.columns:
        qtypes_df = (
            df.groupby("question_type", dropna=False)
              .agg(Accuracy=("is_correct", "mean"),
                   Samples=("is_correct", "size"),
                   Correct=("is_correct", "sum"))
              .reset_index()
              .sort_values(["Accuracy","Samples"], ascending=[False, False])
        )
    else:
        qtypes_df = pd.DataFrame(columns=["question_type","Accuracy","Samples","Correct"])

    return overall_df, topics_df, qtypes_df

def print_eval_tables(overall_df, topics_df, qtypes_df):
    print(f"Accuracy: {overall_df['Accuracy'].iat[0]:.4f}")
    print(f"Total samples: {overall_df['Total samples'].iat[0]}")
    print(f"Correct: {overall_df['Correct'].iat[0]}, Wrong: {overall_df['Wrong'].iat[0]}")

    if not topics_df.empty:
        for _, r in topics_df.iterrows():
            print(f"Topic: {r['topic']}  Accuracy: {r['Accuracy']:.4f}  Samples: {int(r['Samples'])}")
    if not qtypes_df.empty:
        for _, r in qtypes_df.iterrows():
            print(f"Question type: {r['question_type']}  Accuracy: {r['Accuracy']:.4f}  Samples: {int(r['Samples'])}")



def _entropy_from_scores(scores: np.ndarray) -> float:
    scores = 20.0 * scores
    z = scores - scores.max()
    p = np.exp(z).astype(np.float32)
    p = p / (p.sum() + 1e-12)
    return float(-(p * (np.log(p + 1e-12))).mean())

def decide_strategy_with_probe(retriever, question: str, tau: float=0.3, probe_k: int = 12, ent_th: float = 0.2,score_th: float=0.5,score_bt: float=0.2):
    probe = retriever.retrieve_raw(question, top_k=probe_k, return_vecs=False, tau = tau)
    scores = np.asarray(probe["scores"], dtype=np.float32)
    mean_score = np.mean(scores)
    if len(scores) == 0:
        return "slow", {"entropy": 0, "rule": "no_hits"}

    s_sorted = np.sort(scores)[::-1]
    ent = _entropy_from_scores(s_sorted)
    if mean_score>=score_th:
        return "fast", {"entropy": ent, "rule": "mean is high","mean_score": mean_score}
    elif mean_score<=score_bt:
        return "slow", {"entropy": ent, "rule": "slow_high_entropy","mean_score": mean_score}
    # print(ent)
    if   ent<ent_th:
        return "fast", {"entropy": ent, "rule": "fast_low_entropy","mean_score": mean_score}
    else:
        return "slow", {"entropy": ent, "rule": "slow_high_entropy","mean_score": mean_score}