from typing import List, Dict, Any, Tuple, Union
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from scipy.special import softmax
from typing import List, Tuple, Set, Optional

import hashlib,json,os

class EmbeddingRetrievaler:
    def __init__(self, model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1", top_k: int = 5, cache_path="./32k_embd/"):
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k
        self.index = None
        self.docs: List[str] = []
        self.doc_meta: List[Dict[str, Any]] = []
        self.embeddings = None
        self._svd = None
        self._Z_docs = None 
        cache_dir = './'+cache_path + '_embd/'
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _history_uuid(self, history):
        text = json.dumps(history, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()
    
    def build_from_history(self, history: List[Union[str, Dict[str, str]]]):
        norm = []
        if type(history[0])==str:
            docs = history
            metas = []
        else:
            for i, msg in enumerate(history):
                if isinstance(msg, dict):
                    role = (msg.get("role") or "unknown").strip().lower()
                    text = (msg.get("content") or "").strip()
                else:
                    role, text = "unknown", str(msg).strip()
                if not text:
                    continue
                if role == "system":
                    continue
                norm.append({"role": role, "content": text, "src_index": i})

            merged_same = []
            for item in norm:
                if not merged_same:
                    merged_same.append({
                        "role": item["role"],
                        "content": item["content"],
                        "src_indices": [item["src_index"]],
                    })
                    continue
                last = merged_same[-1]
                if item["role"] == last["role"]:
                    last["content"] += "\n\n" + item["content"]
                    last["src_indices"].append(item["src_index"])
                else:
                    merged_same.append({
                        "role": item["role"],
                        "content": item["content"],
                        "src_indices": [item["src_index"]],
                    })

            docs, metas = [], []
            i = 0
            while i < len(merged_same):
                cur = merged_same[i]
                if cur["role"] == "user":
                    if i + 1 < len(merged_same) and merged_same[i + 1]["role"] == "assistant":
                        nxt = merged_same[i + 1]
                        text = (
                            "User:\n" + cur["content"] +
                            "\n\nAssistant:\n" + nxt["content"]
                        )
                        docs.append(text)
                        metas.append({
                            "roles": ["user", "assistant"],
                            "src_indices": cur["src_indices"] + nxt["src_indices"],
                        })
                        i += 2
                        continue
                    else:
                        i += 1
                        continue
                elif cur["role"] == "assistant":
                    text = "Assistant:\n" + cur["content"]
                    docs.append(text)
                    metas.append({
                        "roles": ["assistant"],
                        "src_indices": cur["src_indices"],
                    })
                    i += 1
                else:
                    text = f"{cur['role'].capitalize()}:\n" + cur["content"]
                    docs.append(text)
                    metas.append({
                        "roles": [cur["role"]],
                        "src_indices": cur["src_indices"],
                    })
                    i += 1
        key = self._history_uuid(docs)
        emb_path = os.path.join(self.cache_dir, f"{key}.npz")
        faiss_path = os.path.join(self.cache_dir, f"{key}.faiss")

        if os.path.exists(emb_path) and os.path.exists(faiss_path):
            arr = np.load(emb_path)["embeddings"]
            index = faiss.read_index(faiss_path)
            self.docs = docs
            self.embeddings = arr
            self.index = index
            # print(f"[CACHE HIT] loaded embeddings/index for {key}")
        else:
            self.docs = docs
            self.doc_meta = metas
            arr = self.model.encode(
                self.docs,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            dim = arr.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(arr.astype(np.float32))
            
            np.savez_compressed(emb_path, embeddings=arr)
            faiss.write_index(index, faiss_path)
            self.index = index
            self.embeddings = arr

        if not self.docs:
            self.embeddings = None
            self.index = None
            return

    def retrieve(self, query: str, top_k: int = None,tau: float = 0.5) -> List[Tuple[int, float]]:
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
        for rank, (idx, score) in enumerate(hits, 1):
            meta = self.docs[idx]
            msgs.append(meta)
        return msgs
    
    def _l2norm(self, v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / (n + eps)
    
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
        merged = global_results[:out_k]
        if case:
            print(merged)
        return merged

if __name__ == "__main__":
    context = [
        {"role": "user", "content": "I love iphone"},
        {"role": "assistant", "content": "Iphone is from Apple Company."},
        {"role": "user", "content": "Especially, please introduce the iphone 17 to me."}
    ]

    retriever = EmbeddingRetrievaler(model_name="multi-qa-MiniLM-L6-cos-v1", top_k=2)
    retriever.build_from_history(context)
    retriever.stabilized_embeddings()
    print(retriever.P)

    question = "How to choose the version of iphone?"
    hits = retriever.retrieve(question)
    print(hits)

    rag_context = retriever.format_context(hits, with_scores=True)
    print(rag_context)

    rag_context = retriever.to_messages(hits, with_scores=True)
    print(rag_context)