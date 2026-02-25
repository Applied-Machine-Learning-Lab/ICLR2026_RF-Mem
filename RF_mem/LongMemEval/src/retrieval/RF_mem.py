import json
import numpy as np

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from scipy.special import softmax
from typing import List, Tuple, Set, Optional
from collections import Counter
import hashlib,json,os

personabench_path = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, personabench_path) 

import time
import argparse
import faiss
from sentence_transformers import SentenceTransformer
from src.retrieval.eval_utils import evaluate_retrieval, evaluate_retrieval_turn2session
from tqdm import tqdm
# from openai_yy import OpenAIGPT
import asyncio
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import cosine_similarity
import scipy
import cvxpy as cp
import ot
import pandas as pd
from scipy.special import softmax

def safe_generate(model, prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            return model.generate(prompt)
        except:
            print(f"Attempt {attempt+1} failed  Retrying...")
            time.sleep(2 * (attempt + 1)) 
    raise RuntimeError("Failed to generate after retries.")


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
def load_json_res(res):
    import re
    match = re.search(r"(?:json)?\s*(\{.*?\})\s*", res, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            data = {}
            print("error in transform json", e)
    else:
        print("Outpu t is not in JSON format")
        data = {}
    return data

def save_json(data, file_path):
    with open(file_path, "w") as f:
        if isinstance(data, dict) or isinstance(data, list):
            json.dump(data, f, indent=4)
        elif isinstance(data, str):
            try:
                # Verify if it's a valid JSON string
                json.loads(data)
                f.write(data)
            except json.JSONDecodeError:
                raise ValueError("Provided string is not valid JSON")
        else:
            raise TypeError("Data must be a list/dictionary or a JSON string")
        
def gen_retrieval_prompt_HyDE(query,model):
    prompt_template = """Please write a paragraph that answers the question.\nQuestion:  {query} \nOutput:"""
    instruction = prompt_template.format(query=query)
    # print(instruction)
    prompt = {"instruction": instruction}
    response = safe_generate(model, prompt, max_retries=3)
    return response

def remove_key(json_data, key_to_remove):
    if isinstance(json_data, dict):
        return {k: remove_key(v, key_to_remove) for k, v in json_data.items() if k != key_to_remove}
    elif isinstance(json_data, list):
        return [remove_key(item, key_to_remove) for item in json_data]
    else:
        return json_data
    
def convert_json_to_plain_text(json_data, exclude=None):
    """
    conver json data to plain text for RAG
    exclude: the keys that should not be included
    """
    
    for key_to_remove in exclude:
        json_data = remove_key(json_data, key_to_remove)
    plian_text = json.dumps(json_data, separators=(',', ':'))
    return plian_text


class RAGRetriever:
    def __init__(self, retriever_model, retriever_model_name,data_type,cache_path="LME"):
        self.retriever_model = retriever_model
        self.model = retriever_model
        self.index = None
        self.chunks = []
        self.segment_ids = []
        self.user_embd_mean = None
        cache_dir = './'+cache_path + '_embd/'
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)


    def _history_uuid(self, history):
        text = json.dumps(history, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()
    
    def build_index(self, test_item):
        corpus, corpus_ids, corpus_timestamps = [], [], []
        for cur_sess_id, sess_entry, ts in zip(test_item['haystack_session_ids'], test_item['haystack_sessions'], test_item['haystack_dates']):
            user_data = []
            for item in sess_entry:
                user_data.append(item['content'])
            tmp_data = {
                'date': ts,
                'conversation':user_data
            }
            segment_id = cur_sess_id
            plian_text = json.dumps(tmp_data, separators=(',', ':'))
            corpus.append(plian_text)
            corpus_ids.append(segment_id)

        key = self._history_uuid(plian_text)
        emb_path = os.path.join(self.cache_dir, f"{key}.npz")
        faiss_path = os.path.join(self.cache_dir, f"{key}.faiss")
        if os.path.exists(emb_path) and os.path.exists(faiss_path):
            arr = np.load(emb_path)["embeddings"]
            index = faiss.read_index(faiss_path)
            self.chunks = corpus
            self.embeddings = arr
            self.index = index
            self.segment_ids = np.array(corpus_ids)
        else:
            self.chunks = corpus
            embeddings = self.retriever_model.encode(corpus, convert_to_numpy=True)
            dim = embeddings.shape[1]
            self.embeddings = embeddings
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings.astype(np.float32))
            
            np.savez_compressed(emb_path, embeddings=embeddings)
            faiss.write_index(index, faiss_path)
            self.index = index
            self.embeddings = embeddings
            self.segment_ids = np.array(corpus_ids)


    def _l2norm(self, v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / (n + eps)

    def query(self, questions, top_k=5, tau=0.0):
        if self.index is None:
            raise ValueError("Index has not been built. Please call build_index() first.")

        q_embeddings = self.retriever_model.encode(questions, convert_to_numpy=True)
        D, I = self.index.search(q_embeddings, top_k)
        cand_sc = D[0].tolist()
        cand_ids  = I[0].tolist()
        cands = []
        out_D = []
        out_I = []
        for cid, sc in zip(cand_ids, cand_sc):
            if sc>=tau:
                out_I.append(cid)
                out_D.append(sc)
        rankings_id =self.segment_ids[np.array([out_I])].tolist()[0]
        retrieved_chunks = np.array(self.chunks)[np.array([out_I])].tolist()[0]
        return out_D, out_I, retrieved_chunks, rankings_id
    def retrieve_raw(self, query: str, top_k: int = 12, tau: float = 0.0, return_vecs: bool = True):
        if self.index is None:
            raise ValueError("Index not built. Call build_from_history first.")
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, idxs = self.index.search(q_emb, top_k)
        ids = []
        scs = []
        for idx, score in zip(idxs[0], scores[0]):
            if score >= tau:  
                ids.append(int(idx))
                scs.append(float(score))
        if return_vecs:
            vecs = self.embeddings[ids]
        else:
            vecs = None
        return {"ids": ids, "scores": scs, "vecs": vecs}
    def retrieve_pyramid_v2(
        self,
        query: str,
        out_k: int = 5,                 
        tau: float = 0.5,                
        depth: int = 5,                 
        beam_width: int = 3,             
        fanout: int = 3,                 
        expansion: str = "mean_group",   
        alpha: float = 0.8,              
        weight_temp: float = 20.0,       
        mmr_lambda: float = 0.95,       
        exclude_self: bool = False,     
        exclude_idx: Optional[int] = None, 
        case: bool = False,
    ) -> List[Tuple[int, float]]:

        q0 = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]


        class Branch:
            __slots__ = ("q", "hits", "score_sum", "parent")

            def __init__(self, q, parent=None):
                self.q: np.ndarray = q
                self.hits: List[Tuple[int, float]] = []   
                self.score_sum: float = 0.0              
                self.parent = parent                      


        global_seen: Set[int] = set()
        global_results: List[Tuple[int, float]] = []

        if exclude_self and exclude_idx is not None:
            global_seen.add(int(exclude_idx))
        beam: List[Branch] = [Branch(q0)]

        for lvl in range(depth):
            next_candidates: List[Branch] = []

            level_selected_vecs: List[np.ndarray] = []

            for br in beam:
                K = (beam_width+lvl) * fanout
                scores, idxs = self.index.search(br.q[None, :], K)
                cand_ids = idxs[0].tolist()
                cand_sc  = scores[0].tolist()
                cands: List[Tuple[int, float]] = []
                for cid, sc in zip(cand_ids, cand_sc):
                    if sc >= tau and cid not in global_seen:
                        if exclude_self and exclude_idx is not None and int(cid) == int(exclude_idx):
                            continue
                        cands.append((int(cid), float(sc)))

                if not cands:
                    continue
                selected = []
                seen = []
                vecs = [self.embeddings[cid] for cid, _ in cands]

                for i, (cid, sc) in enumerate(cands):
                    if cid not in seen:
                        v = vecs[i]
                        sims = [float(np.dot(v, vecs[j])) for j in range(len(vecs)) if j != i]
                        max_sim = max(sims) if sims else 0.0
                        mmr = mmr_lambda * sc - (1.0 - mmr_lambda) * max_sim
                        selected.append((cid, mmr))
                        seen.append(cid)
                    else:
                        continue
                if expansion == "mean":
                    centroid = self._l2norm(np.mean(np.stack([self.embeddings[c] for c, _ in selected], axis=0), axis=0))
                    q_new = self._l2norm(alpha * br.q + (1.0 - alpha) * centroid)
                    child = Branch(q_new)
                    child.hits = selected[:]  
                    child.score_sum = br.score_sum + float(sum(sc for _, sc in selected))
                    next_candidates.append(child)
                elif expansion == "weighted":
                    ws = np.array([sc for _, sc in selected], dtype=np.float32)
                    ws = np.exp(ws * float(weight_temp))
                    ws = ws / (ws.sum() + 1e-12)
                    centroid = self._l2norm((ws[:, None] * np.stack([self.embeddings[c] for c, _ in selected], axis=0)).sum(axis=0))
                    q_new = self._l2norm(alpha * br.q + (1.0 - alpha) * centroid)
                    child = Branch(q_new)
                    child.hits = selected[:]  
                    child.score_sum = br.score_sum + float(sum(sc for _, sc in selected))
                    next_candidates.append(child)
                elif expansion == "single":
                    for cid, sc in selected:
                        v = self.embeddings[cid]
                        q_new = self._l2norm(alpha * br.q + (1.0 - alpha) * v)
                        child = Branch(q_new)
                        child.hits = [(cid, sc)]
                        child.score_sum = br.score_sum + float(sc)
                        next_candidates.append(child)
                    continue
                elif expansion == "mean_group":
                    vecs = np.stack([self.embeddings[c] for c, _ in selected], axis=0).astype(np.float32)
                    n_clusters = min(beam_width, len(selected))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    labels_km = kmeans.fit_predict(vecs)
                    for k in range(n_clusters):
                        group_idx = [i for i, lab in enumerate(labels_km) if lab == k]
                        group_cids = [selected[i][0] for i in group_idx]
                        group_scs  = [selected[i][1] for i in group_idx]
                        centroid = self._l2norm(vecs[group_idx].mean(axis=0))
                        q_new = self._l2norm(alpha * br.q + (1.0 - alpha) * centroid + q0)
                        child = Branch(q_new, br)
                        child.hits = [(cid, sc) for cid, sc in zip(group_cids, group_scs)]
                        child.score_sum = br.score_sum + float(sum(group_scs))
                        next_candidates.append(child)
                else:
                    raise ValueError("expansion must be 'mean' | 'weighted' | 'single'| 'mean_group'.")

            if not next_candidates:
                continue
            
            next_candidates.sort(key=lambda b: b.score_sum, reverse=True)
            beam = next_candidates[:beam_width]
            if case:
                print(lvl)
                print(len(beam))
                
                for be in beam:
                    print(be.hits)
                print(len(next_candidates))
                for be in next_candidates:
                    print("parent")
                    print(be.parent.hits)
                    print("new retrieved")
                    print(be.hits)
            for br in beam:
                for cid, sc in br.hits:
                    if cid not in global_seen:
                        global_seen.add(cid)
                        global_results.append((cid, sc))
            if len(global_results) >= out_k:
                break

        best_score: Dict[int, float] = {}
        for idx, sc in global_results:
            if idx not in best_score or sc > best_score[idx]:
                best_score[idx] = sc
        I = [item[0] for item in global_results]
        rankings_id =self.segment_ids[np.array([I])].tolist()[0]
        retrieved_chunks = np.array(self.chunks)[np.array([I])].tolist()[0]
        return [], I, retrieved_chunks, rankings_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="s", help="Which model to use.")
    parser.add_argument("--retrieval_model_name", type=str, default="BAAI/bge-base-en-v1.5", help="Which model to use.")
    parser.add_argument("--retrieval_mode", type=str, default="base", help="Which model to use.")
    parser.add_argument("--tau", type=float, default=0.0, help="Which model to use.")
    parser.add_argument("--alpha", type=float, default=0.0, help="Which model to use.")
    parser.add_argument("--B", type=int, default=3, help="Which model to use.")
    parser.add_argument("--F", type=int, default=1, help="Which model to use.")

    
    
    args = parser.parse_args()
    retriever_model_name = args.retrieval_model_name
    print(retriever_model_name)
    method = args.model_type
    print(method)
    data_type = args.data_type
    in_file = f"./data/longmemeval_data/longmemeval_{data_type}.json"
    print(in_file)
    if "longmemeval_s" in in_file:
        save_path = in_file.replace('.json','_mem.json')
    else:
        save_path = in_file.replace('.json','_mem.json')
   
    print(save_path)
    in_data = json.load(open(in_file))
    retriever_model = SentenceTransformer(retriever_model_name, trust_remote_code=True)
    
    results= []
    out_json = []
    hist_num_ls = []
    fast_or_slow_ls = []
    ent_ls = []
    time_ls = []
    mean_score = []
    topk_uni=50
    for test_item in in_data:
        question = test_item['question']
        retriever = RAGRetriever(retriever_model,retriever_model_name, data_type,cache_path=args.retrieval_model_name)
        retriever.build_index(test_item)
        questions = [question]
        if args.retrieval_mode =='fami':
            t4 = time.perf_counter()
            D, I, retrieved_chunks,rankings_id = retriever.query(questions, top_k=topk_uni,tau=0.0)
            t5 = time.perf_counter()
            time_spend = t5-t4
        elif args.retrieval_mode =='reco':
            t4 = time.perf_counter()
            D, I, retrieved_chunks,rankings_id = retriever.retrieve_pyramid_v2(questions[0],out_k=topk_uni, tau=0.0, depth=50, beam_width=args.B, fanout=args.F, expansion="mean_group", alpha=0.8, mmr_lambda=0.99)
            t5 = time.perf_counter()
            time_spend = t5-t4
        elif args.retrieval_mode =='RF-Mem':
            mode, diag = decide_strategy_with_probe(retriever, question, tau=0.0, probe_k=10, ent_th=args.tau, score_th=0.60, score_bt=0.00)
            ent = diag['entropy']
            ent_ls.append(ent)
            mean_score.append(diag['mean_score'])
            fast_or_slow = {"text": mode}
            fast_or_slow_ls.append(fast_or_slow["text"])
            t4 = time.perf_counter()
            if mode=='fast':
                D, I, retrieved_chunks,rankings_id = retriever.query(questions, top_k=topk_uni, tau=0.0)
            else:
                D, I, retrieved_chunks,rankings_id = retriever.retrieve_pyramid_v2(questions[0],out_k=topk_uni, tau=0.0, depth=50,beam_width=args.B, fanout=args.F, expansion="mean_group", alpha=args.alpha, mmr_lambda=1.0)
            t5 = time.perf_counter()
            time_spend = t5-t4
        else:
            D, I, retrieved_chunks,rankings_id = retriever.query(questions, top_k=topk_uni)
        # print(len(I))
        time_ls.append(time_spend)
        corpus_ids = [item for item in test_item['haystack_session_ids']]
        ret_res = []
        rankings = []
        for res,ids in zip(retrieved_chunks,I):
            tmp_rank = {
                    'corpus_id': ids,
                    'text': res,
                }
            rankings.append(ids)
            ret_res.append(tmp_rank)
        cur_results = {
            'question_id': test_item['question_id'],
            'question_type': test_item['question_type'],
            'question': test_item['question'],
            'answer': test_item['answer'],
            'question_date': test_item['question_date'],
            'haystack_dates': test_item['haystack_dates'],
            'haystack_sessions': test_item['haystack_sessions'],
            'haystack_session_ids': test_item['haystack_session_ids'],
            'answer_session_ids': test_item['answer_session_ids'],
            'retrieval_results': {
                'query': question,
                'ranked_items': ret_res,
                'metrics': {
                    'session': {},
                    'turn': {}
                    }
                }
            }

        correct_docs = list(set([doc_id for doc_id in corpus_ids if "answer" in doc_id]))
        for k in [5,10,20,50]:
            recall_any, recall_all, ndcg_any = evaluate_retrieval(rankings, correct_docs, corpus_ids, k=k)
            cur_results['retrieval_results']['metrics']['session'].update({
                'recall_all@{}'.format(k): recall_all,
                # 'ndcg_any@{}'.format(k): ndcg_any
            })

        out_json.append(cur_results)
        results.append(cur_results)
    time_arr = np.array(time_ls)
    mean_val = np.mean(time_arr)
    var_val = np.var(time_arr, ddof=0)
    print("time mean:", mean_val)
    print("time std:", var_val)
    q25, q50, q75 = np.percentile(time_arr, [25, 50, 75])
    print("25% :", q25)
    print("50% :", q50)
    print("75% :", q75)
    averaged_results = {
        'session': {},
        'turn': {}
    }
    ignored_qs_abstention, ignored_qs_no_target = set(), set()
    for k in results[0]['retrieval_results']['metrics']['session']:
        try:
            results_list = []
            for eval_entry in results:
                # will skip abstention instances for reporting the metric
                if '_abs' in eval_entry['question_id']:
                    ignored_qs_abstention.add(eval_entry['question_id'])
                    continue
                # will also skip instances with no target labels
                if not any(('has_answer' in turn) and (turn['has_answer']) for turn in [x for y in eval_entry['haystack_sessions'] for x in y if x['role'] == 'user']):
                    ignored_qs_no_target.add(eval_entry['question_id'])
                    continue
                results_list.append(eval_entry['retrieval_results']['metrics']['session'][k])
                
            averaged_results['session'][k] = np.mean(results_list)
        except:
            continue
    session_data = averaged_results.get("session", {})
    df = pd.DataFrame(list(session_data.items()), columns=["Metric", "Value"])
    print(df.head(30))
    print(json.dumps(averaged_results))
    save_json(out_json,save_path)
