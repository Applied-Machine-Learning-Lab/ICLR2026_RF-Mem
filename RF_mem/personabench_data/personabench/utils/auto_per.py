import os
import sys
from pathlib import Path

# 自动获取 personabench 的绝对路径（跨平台兼容）
personabench_path = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, personabench_path) 

import argparse
import faiss
import tqdm
import logging
import numpy as np
import json
from personabench.utils.shared_args import add_shared_arguments
from personabench.utils.utils import get_api_key, load_json, convert_json_to_plain_text, save_json, set_random_seed, setup_logger
from sentence_transformers import SentenceTransformer
from code.personabench_data.personabench.models.openai_my import OpenAIGPT

from personabench.utils.eval import calculate_retrieval_scores,calculate_qa_scores



def create_vector_base(documents_dict, retriever_model=None):
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
    if retriever_model:
        embeddings = retriever_model.encode(chunks, convert_to_numpy=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
    else:
        index = None
    return chunks, index, np.array(segment_ids)


def load_json_res(res):
    import re
    match = re.search(r"(?:json)?\s*(\{.*?\})\s*", res, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            data = json.loads(json_str)
            # print("Locutionary:", data["locutionary"])
            # print("Illocutionary:", data["illocutionary"])
            # print("Perlocutionary:", data["perlocutionary"])
        except json.JSONDecodeError as e:
            data = {}
            print("error in transform json", e)
    else:
        print("Outpu t is not in JSON format")
        data = {}
    return data

def extract_gt_segment_ids(QA, qa_gt_context_all_lookup, num_chunks):
    gt_segment_ids_retrieval = []
    for qa in QA:
        q_id = qa["q_id"]
        question_gt_ids = []
        for individual_ids in qa_gt_context_all_lookup[q_id]["segment_id"].values():
            question_gt_ids.append(individual_ids[0])
        if len(question_gt_ids) < num_chunks:
            additional_count = num_chunks - len(question_gt_ids)
            # Repeat the list to fill up the required length
            repeat_part = question_gt_ids * (additional_count // len(question_gt_ids))
            remaining_part = question_gt_ids[:additional_count % len(question_gt_ids)]
            question_gt_ids += repeat_part + remaining_part
        elif len(question_gt_ids) > num_chunks:
            raise ValueError(f"find {len(question_gt_ids)} ground truth, which is larger than {num_chunks}")
        gt_segment_ids_retrieval.append(question_gt_ids)
    return np.array(gt_segment_ids_retrieval)

def gen_retrieval_prompt(query,doc,model,use_key = ["locutionary","illocutionary","perlocutionary"]):
    prompt_template = """Your task is to generate **three personalized rewrites** of the user's current question,
each reflecting one layer of Speech-Act Theory:  
  1. Locutionary Act → literal, surface wording  
  2. Illocutionary Act → highlights the speaker's communicative intent or social function  
  3. Perlocutionary Act → formulated to achieve a desired effect on the listener  

Context  
------------
User dialogue history (for style imitation):
{history}

Current question to rewrite:
{query}
------------

Guidelines
1. **Imitate** the user’s habitual vocabulary, tone, and formality observed in the history.  
2. Each rewrite ≤ 25 words.  
3. Preserve the original factual meaning.  
4. Output **valid JSON** with exactly these keys:  
   ```json
   {{
     "locutionary": "...",
     "illocutionary": "...",
     "perlocutionary": "..."
   }}
   ```
"""
    instruction = prompt_template.format(query=query, history=doc)
    # print(instruction)
    prompt = {"instruction": instruction}
    response = model.generate(prompt)
    query_dict = load_json_res(response)
    query_ls = []
    for key in use_key:
        query_ls.append(query_dict[key])
    # print(query_ls)
    return '\n'.join(query_ls)

def extract_gt_segment_ids(QA, qa_gt_context_all_lookup, num_chunks):
    gt_segment_ids_retrieval = []
    for qa in QA:
        q_id = qa["q_id"]
        question_gt_ids = []
        for individual_ids in qa_gt_context_all_lookup[q_id]["segment_id"].values():
            question_gt_ids.append(individual_ids[0])
        if len(question_gt_ids) < num_chunks:
            additional_count = num_chunks - len(question_gt_ids)
            # Repeat the list to fill up the required length
            repeat_part = question_gt_ids * (additional_count // len(question_gt_ids))
            remaining_part = question_gt_ids[:additional_count % len(question_gt_ids)]
            question_gt_ids += repeat_part + remaining_part
        elif len(question_gt_ids) > num_chunks:
            raise ValueError(f"find {len(question_gt_ids)} ground truth, which is larger than {num_chunks}")
        gt_segment_ids_retrieval.append(question_gt_ids)
    return np.array(gt_segment_ids_retrieval)

def gen_retrieval_prompt(query,doc,model,use_key = ["locutionary","illocutionary","perlocutionary"]):
    prompt_template = """Your task is to generate **three personalized rewrites** of the user's current question,
each reflecting one layer of Speech-Act Theory:  
  1. Locutionary Act → literal, surface wording  
  2. Illocutionary Act → highlights the speaker's communicative intent or social function  
  3. Perlocutionary Act → formulated to achieve a desired effect on the listener  

Context  
------------
User dialogue history (for style imitation):
{history}

Current question to rewrite:
{query}
------------

Guidelines
1. **Imitate** the user’s habitual vocabulary, tone, and formality observed in the history.  
2. Each rewrite ≤ 25 words.  
3. Preserve the original factual meaning.  
4. Output **valid JSON** with exactly these keys:  
   ```json
   {{
     "locutionary": "...",
     "illocutionary": "...",
     "perlocutionary": "..."
   }}
   ```
"""
    instruction = prompt_template.format(query=query, history=doc)
    # print(instruction)
    prompt = {"instruction": instruction}
    response = model.generate(prompt)
    query_dict = load_json_res(response)
    query_ls = []
    for key in use_key:
        query_ls.append(query_dict[key])
    # print(query_ls)
    return '\n'.join(query_ls)


def gen_optimizl_user_history(model,history):
    prompt_template = """You are an expert personalization engine.
        Your task is to craft **one concise “User personality and speech style”** (≤ 200 words)
        that best captures the user’s stable preferences and context.
        A good description should maximize downstream retrieval quality in our RAG pipeline.
        Scoring rule: **RetrievalReward = nDCG@10** of the answer returned
        when this description is used.

        Below are previous descriptions we tried and the resulting rewards
        (higher = better):

        {history_lines}

        ### PART 3 – YOUR OUTPUT
        Please output only a descirption.
        Do not add any other explanation or text.
        Output:\n"""
    instruction = prompt_template.format(history_lines=history)
    # print(instruction)
    prompt = {"instruction": instruction}
    response = model.generate(prompt)
    return response
def gen_response(query_ls, model, retrival_res):
    prompt_template = (
                        "You are provided with the following relevant information: {doc}.\n"
                        "Answer the question below as directly and concisely as possible, using only the name(s) "
                        "of the relevant entity or entities. Avoid adding any extra words or explanations.\n\n"
                        "Question: {query}")
    final_response = []
    for i, query in enumerate(query_ls):
        # generate prediction for individual question
        instruction = prompt_template.format(query=query, doc=retrival_res[i])
        prompt = {"instruction": instruction}
        prediction = model.generate(prompt)
        final_response.append(prediction)
    return final_response


def user_specific_RAG(corpus,retriever_model,chunks,index, segment_ids, num_chunks,retrieval_type,gt_segment_ids_retrieval):
    '''
        retriever_model: sentenceformer model
        chunks,index, segment_ids: faiss database
        corpus: rag corpus
        num_chunks: the number of RAG result
        retrieval_type: gt or other model
        gt_segment_ids_retrieval: gt ids.

    '''
    if retrieval_type == "gt-context":  # use ground-truth context
        id_to_index = {id_: idx for idx, id_ in enumerate(segment_ids)}
        I = np.vectorize(id_to_index.get)(gt_segment_ids_retrieval)
    else:
        q_embeddings = retriever_model.encode(corpus, convert_to_numpy=True)
        D, I = index.search(q_embeddings, num_chunks)
    retrival_res = np.array(chunks)[I]  # (# questions, # retrieve)
    return retrival_res

def auto_per_optimize(QA,model,epoch, retriever_model,chunks,index, segment_ids,num_chunks,retrieval_type,gt_segment_ids_retrieval,qa_gt_context_all_lookup, eval_type = "final"):
    QA_10= QA[:10]
    query_ls = [item['question'] for item in QA_10]
    best_user_desc = ""
    best_score = 0
    history = []
    for i in range(epoch):
        user_desc = gen_optimizl_user_history(model,history)
        doc = user_specific_RAG([user_desc], retriever_model,chunks,index, segment_ids,  num_chunks,retrieval_type,gt_segment_ids_retrieval)
        rewrite_quert_ls = []
        for q in query_ls:
            rewrite_query = gen_retrieval_prompt(q,doc,model,use_key = ["locutionary","illocutionary","perlocutionary"])
            rewrite_quert_ls.append(rewrite_query)
        retrieval_res = user_specific_RAG(rewrite_quert_ls,retriever_model,chunks,index, segment_ids, num_chunks,retrieval_type,gt_segment_ids_retrieval)
        results =[]
        if eval_type=="final":
            final_res = gen_response(query_ls, model, retrieval_res)
            for q_dict,prediction in zip(QA_10,final_res):
                results.append({"q_id": q_dict["q_id"],
                                "question": q_dict["question"],
                                "answer": q_dict["answer"],
                                "prediction":prediction,
                                "type": q_dict["type"],
                                "difficulty": q_dict["difficulty"],
                                "retrieved_segment_ids": list(segment_ids[I[i]]),
                                "ground_truth_segment_ids": qa_gt_context_all_lookup[q_dict["q_id"]]["segment_id"]}
                                )
            reward = calculate_qa_scores(results)
            reward = reward["Overall"]["total_precision"]
            if reward>best_score:
                best_user_desc = user_desc
                score = best_score
            history_tem = f"""You generate personality is {user_desc}, get total_precision score {reward}."""
        else:
            final_res = ['']*10
            for q_dict,prediction in zip(QA_10,final_res):
                results.append({"q_id": q_dict["q_id"],
                                "question": q_dict["question"],
                                "answer": q_dict["answer"],
                                "prediction":prediction,
                                "type": q_dict["type"],
                                "difficulty": q_dict["difficulty"],
                                "retrieved_segment_ids": list(segment_ids[I[i]]),
                                "ground_truth_segment_ids": qa_gt_context_all_lookup[q_dict["q_id"]]["segment_id"]}
                                )
            reward = calculate_retrieval_scores(results)
            reward = reward["Overall"]["total_recall"]
            if reward>best_score:
                best_user_desc = user_desc
                score = best_score
            history_tem = f"""You generate personality is {user_desc}, get total_recall score {reward}."""
        history.append(history_tem)
    return best_user_desc




def gen_user_specific_prompt(args):
    base_models = [r.strip() for r in args.base_models.split(',')]
    retrievers = [r.strip() for r in args.retrievers.split(',')]
    community_ids = [r.strip() for r in args.test_community_ids.split(',')]
    noises = [r.strip() for r in args.test_noises.split(',')]
    test_result_dir = os.path.join(args.log_dir, args.save_dir)
    

    # loop over noise level, test all Q&A
    for noise in tqdm.tqdm(noises):
        logging.info(f"Evaluating with supported documents at noise {noise}.")
        results_all_community = []
        for retriever in retrievers:
            for model_name in base_models:
                logging.info(f"Model is {model_name}, retriever is {retriever}")
                if "gpt" in model_name.lower():
                    # api_key = get_api_key(args)
                    api_key = ''
                    model = OpenAIGPT(api_key, model=model_name)
                else:
                    raise ValueError(f"Model type {model_name} not implemented")
                # Initialize embedding model
                if retriever == "gt-context":  # use ground-truth retriving
                    retriever_model = None
                else:
                    retriever_model = SentenceTransformer(retriever, trust_remote_code=True)  # Example embedding model
                results_all_community.append({"Model": f"{model_name}+{retriever}", "Results": []})
                logging.info(f"evaluating {model_name}+{retriever}")
                save_path = os.path.join(test_result_dir, f"results_all_community_chunk_{args.num_chunks}")
                os.makedirs(save_path, exist_ok=True)
                # evaluate all communities
                for community_id in community_ids:
                    community_path = os.path.join(args.data_dir, "synthetic_data", community_id)
                    eval_info_all = load_json(os.path.join(community_path, 'eval_info', 'eval_info_all.json'))
                    qa_gt_context_all = load_json(os.path.join(community_path, 'eval_info', f'qa_gt_context_all_noise_{noise}.json'))
                    qa_gt_context_all_lookup = {entry["q_id"]: entry for entry in qa_gt_context_all}  # for retrieve by q_id
                    # load all documents
                    conversation_data_all = load_json(os.path.join(community_path, 'private_data', f"noise_{noise}", "conversation_data_all.json"))
                    user_ai_interaction_data_all = load_json(os.path.join(community_path, 'private_data', f"noise_{noise}", "user_ai_interaction_data_all.json"))
                    purchase_history_data_all = load_json(os.path.join(community_path, 'private_data', f"noise_{noise}", "purchase_history_data_all.json"))
                    conversation_data_all_lookup = {entry["Name"]: entry["Data"] for entry in conversation_data_all}
                    user_ai_interaction_data_all_lookup = {entry["Name"]: entry["Data"] for entry in user_ai_interaction_data_all}
                    purchase_history_data_all_lookup = {entry["Name"]: entry["Data"] for entry in purchase_history_data_all}
                    # evaluate for each person
                    for eval_info_dict in eval_info_all[:args.num_people]:
                        name = eval_info_dict["Name"]
                        eval_info = eval_info_dict["Eval_Info"]
                        conversation_data = conversation_data_all_lookup[name]
                        user_ai_interaction_data = user_ai_interaction_data_all_lookup[name]
                        purchase_history_data = purchase_history_data_all_lookup[name]
                        documents_dict = {"conversation_data": conversation_data, "user_ai_interaction_data": user_ai_interaction_data, "purchase_history_data": purchase_history_data}
                        chunks, index, segment_ids = create_vector_base(documents_dict, retriever_model)
                        QA = eval_info["qa"]
                        questions = [entry['question'] for entry in QA]
                        gt_segment_ids_retrieval = extract_gt_segment_ids(QA, qa_gt_context_all_lookup, args.num_chunks)
                        if retriever == "gt-context":  # use ground-truth context
                            id_to_index = {id_: idx for idx, id_ in enumerate(segment_ids)}
                            I = np.vectorize(id_to_index.get)(gt_segment_ids_retrieval)
                        else:
                            q_embeddings = retriever_model.encode(questions, convert_to_numpy=True)
                            D, I = index.search(q_embeddings, args.num_chunks)
                        retrival_res = np.array(chunks)[I]  # (# questions, # retrieve)
                        ### v1.0 改动
                        # rewrite query
                        # for i, q_dict in enumerate(QA):
                        #     rewrite_query=gen_retrieval_prompt(q_dict["question"],retrival_res[i],model)
                        #     q_dict['rewrite_query']  = rewrite_query
                        # # retrieval query 相当于个性化2查
                        # rewrite_queries = [entry['rewrite_query'] for entry in QA]
                        # if retriever == "gt-context":  # use ground-truth context
                        #     pass
                        # else:
                        #     q_embeddings = retriever_model.encode(rewrite_queries, convert_to_numpy=True)
                        #     D, I = index.search(q_embeddings, args.num_chunks)
                        # retrival_res = np.array(chunks)[I]  # (# questions, # retrieve)
                        ###
                        ### v2.0 改动
                        # rewrite query
                        user_desc = auto_per_optimize(QA,model,10, retriever_model,chunks,index, segment_ids,args.num_chunks,retriever,gt_segment_ids_retrieval,qa_gt_context_all_lookup, eval_type = "final")
                        doc = user_specific_RAG([user_desc], retriever_model,chunks,index, segment_ids, args.num_chunks,retriever,gt_segment_ids_retrieval)
                        for i, q_dict in enumerate(QA):
                            rewrite_query=gen_retrieval_prompt(q_dict["question"],doc,model)
                            q_dict['rewrite_query']  = rewrite_query
                        # retrieval query 相当于个性化2查
                        rewrite_queries = [entry['rewrite_query'] for entry in QA]
                        if retriever == "gt-context":  # use ground-truth context
                            pass
                        else:
                            q_embeddings = retriever_model.encode(rewrite_queries, convert_to_numpy=True)
                            D, I = index.search(q_embeddings, args.num_chunks)
                        retrival_res = np.array(chunks)[I]  # (# questions, # retrieve)
                        ###
                        logging.info("Retrieve finished")
                        prompt_template = (
                            "You are provided with the following relevant information: {doc}.\n"
                            "Answer the question below as directly and concisely as possible, using only the name(s) "
                            "of the relevant entity or entities. Avoid adding any extra words or explanations.\n\n"
                            "Question: {query}")
                        for i, q_dict in enumerate(QA):
                            # generate prediction for individual question
                            instruction = prompt_template.format(query=q_dict["question"], doc=retrival_res[i])
                            prompt = {"instruction": instruction}
                            prediction = model.generate(prompt)
                            results_all_community[-1]["Results"].append({"q_id": q_dict["q_id"],
                                                                         "question": q_dict["question"],
                                                                         "answer": q_dict["answer"],
                                                                         "prediction":prediction,
                                                                         "type": q_dict["type"],
                                                                         "difficulty": q_dict["difficulty"],
                                                                         "retrieved_segment_ids": list(segment_ids[I[i]]),
                                                                         "ground_truth_segment_ids": qa_gt_context_all_lookup[q_dict["q_id"]]["segment_id"]}
                                                                         )
        save_json(results_all_community, os.path.join(save_path, f"result_noise_{noise}.json"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_shared_arguments(parser)
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to save eval results.")
    parser.add_argument("--save_dir", type=str, default="eval_data_v1", help="Path to save eval results.")
    parser.add_argument("--test_community_ids", type=str, default="community_0,community_1", 
                        help="Comma-separated list of community ids to test")
    parser.add_argument('--num_people', type=int, default=3, 
                        help="Number of individuals to test in each community")
    parser.add_argument("--base_models", type=str, default="gpt-3.5-turbo,gpt-4o", 
                        help="Comma-separated list of base models to evaluate")
    parser.add_argument("--retrievers", type=str, default="gt-context,all-MiniLM-L6-v2,BAAI/bge-m3,all-mpnet-base-v2", 
                        help="Comma-separated list of retriever models to evaluate, gt-context use ground truth source as context during evaluation")
    parser.add_argument("--num_chunks", type=int, default=5, help="Number of chunks for retrieval tasks")
    parser.add_argument("--test_noises", type=str, default="0.0,0.3,0.5,0.7", 
                        help="Comma-separated list of noise levels for testing")
    args = parser.parse_args()
    setup_logger(args.verbose)
    set_random_seed(args.seed)
    gen_user_specific_prompt(args)