from typing import List, Dict, Any, Tuple, Union
from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from scipy.special import softmax
import json
from typing import List, Tuple, Set, Optional

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def build_rag_corpus(community_path,noise):
    conversation_data_all = load_json(os.path.join(community_path, 'private_data', f"noise_{noise}", "conversation_data_all.json"))
    user_ai_interaction_data_all = load_json(os.path.join(community_path, 'private_data', f"noise_{noise}", "user_ai_interaction_data_all.json"))
    purchase_history_data_all = load_json(os.path.join(community_path, 'private_data', f"noise_{noise}", "purchase_history_data_all.json"))
    conversation_data_all_lookup = {entry["Name"]: entry["Data"] for entry in conversation_data_all}
    user_ai_interaction_data_all_lookup = {entry["Name"]: entry["Data"] for entry in user_ai_interaction_data_all}
    purchase_history_data_all_lookup = {entry["Name"]: entry["Data"] for entry in purchase_history_data_all}
    return conversation_data_all_lookup, user_ai_interaction_data_all_lookup,purchase_history_data_all_lookup

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


class EmbeddingRetrievaler:
    def __init__(self, model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1", top_k: int = 5):
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k
        self.index = None
        self.docs: List[str] = []
        self.doc_meta: List[Dict[str, Any]] = []
        self.embeddings = None
        self._svd = None
        self._Z_docs = None  

    

    def build_from_history(self, documents_dict):
        chunks = []
        segment_ids = []
        for document_type, document_data in documents_dict.items():
            if document_type != "conversation_data":
                for data_entry in document_data:
                    segment_id = data_entry["segment_id"]
                    chunk = convert_json_to_plain_text(data_entry, exclude=["segment_id","session"])
                    chunks.append(chunk)
                    segment_ids.append(segment_id)
            else:
                # deal conversation data
                for sub_conversation_data in document_data:
                    for data_entry in sub_conversation_data["Conversations"]:
                        segment_id = data_entry["segment_id"]
                        chunk = convert_json_to_plain_text(data_entry, exclude=["segment_id","session"])
                        chunks.append(chunk)
                        segment_ids.append(segment_id)

        self.chunks = chunks
        self.segment_ids = segment_ids
        self.embeddings = self.model.encode(self.chunks, convert_to_numpy=True, normalize_embeddings=True)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  
        self.index.add(self.embeddings)

    def retrieve(self, query: str, top_k: int = None,tau: float = 0.0) -> List[Tuple[int, float]]:
        if self.index is None:
            raise ValueError("Index not built. Call build_from_history first.")
        k = top_k or self.top_k
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, idxs = self.index.search(q_emb, k)
        res = []
        for idx, score in zip(idxs[0], scores[0]):
            if score >= tau:  
                res.append((int(idx), float(score)))
        return res

    def retrieve_raw(self, query: str, top_k: int = 12, tau: float = 0.0, return_vecs: bool = True):
        if self.index is None:
            raise ValueError("Index not built. Call build_from_history first.")
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, idxs = self.index.search(q_emb, top_k)

        ids = idxs[0].tolist()
        scs = scores[0].astype(float).tolist()
        if return_vecs:
            vecs = self.embeddings[ids]
        else:
            vecs = None
        return {"ids": ids, "scores": scs, "vecs": vecs}
    
    
    def format_context(self, hits: List[Tuple[int, float]], with_scores: bool = False) -> str:
        lines = ["Retrieved context:"]
        for rank, (idx, score) in enumerate(hits, 1):
            meta = self.doc_meta[idx]
            prefix = f"[{rank}] (role={meta['role']}, src_msg={meta['src_index']}"
            if with_scores:
                prefix += f", score={score:.4f}"
            prefix += ")"
            lines.append(prefix)
            lines.append(self.docs[idx])
        return "\n".join(lines)
    
    def to_messages(self, hits, with_scores: bool = False, prefix: str = "", role: str = "system"):
        msgs = []
        ids = []
        for rank, (idx, score) in enumerate(hits, 1):
            rankings_id =self.segment_ids[idx]
            retrieved_chunks = self.chunks[idx]
            ids.append(rankings_id)
            msgs.append(retrieved_chunks)
        return ids, msgs
    
    def _l2norm(self, v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / (n + eps)
    
    
    def retrieve_pyramid_v2(
        self,
        query: str,
        out_k: int = 5,                 
        tau: float = 0.0,               
        depth: int = 3,                 
        beam_width: int = 3,            
        fanout: int = 3,                
        expansion: str = "mean_group",  
        alpha: float = 0.8,             
        weight_temp: float = 20.0,      
        mmr_lambda: float = 0.9,        
        exclude_self: bool = False,     
        exclude_idx: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        q0 = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
        class Branch:
            __slots__ = ("q", "hits", "score_sum")
            def __init__(self, q):
                self.q: np.ndarray = q
                self.hits: List[Tuple[int, float]] = []  
                self.score_sum: float = 0.0            
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
                vecs = [self.embeddings[cid] for cid, _ in cands]

                for i, (cid, sc) in enumerate(cands):
                    v = vecs[i]
                    sims = [float(np.dot(v, vecs[j])) for j in range(len(vecs)) if j != i]
                    max_sim = max(sims) if sims else 0.0
                    mmr = mmr_lambda * sc - (1.0 - mmr_lambda) * max_sim
                    selected.append((cid, mmr))
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
                        child = Branch(q_new)
                        child.hits = [(cid, sc) for cid, sc in zip(group_cids, group_scs)]
                        child.score_sum = br.score_sum + float(sum(group_scs))
                        next_candidates.append(child)
                else:
                    raise ValueError("expansion must be 'mean' | 'weighted' | 'single'| 'mean_group'.")

            if not next_candidates:
                break

            next_candidates.sort(key=lambda b: b.score_sum, reverse=True)
            beam = next_candidates[:beam_width]
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
        merged = global_results[:out_k]
        return merged

if __name__ == "__main__":
    retriever = EmbeddingRetrievaler(
                    model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                    top_k=5,
                )
    for community_id in [0,1]:
        community_path = f"../eval_data/eval_data_v1/synthetic_data/community_{community_id}"
        noise = 0.0
        qa_gt_context_all = load_json(community_path +"/eval_info/qa_gt_context_all_noise_0.0.json")
        eval_info_all = load_json(community_path +"/eval_info/eval_info_all.json")
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
                hits = retriever.retrieve(question, top_k=5)
                ids, context = retriever.to_messages(hits)
                break
            break
        break