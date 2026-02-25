from typing import List, Optional, Dict, Any, Tuple
from vllm import LLM, SamplingParams
import torch
import re

class VLLMQwen:

    def __init__(
        self,
        model: str = "Qwen/Qwen3-7B-Instruct",
        dtype: str = "auto",                # "auto" | "bfloat16" | "float16" | "float32"
        tensor_parallel_size: int = 1,      
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        **llm_kwargs: Any,
    ):
        self.llm = LLM(
            model=model,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            **llm_kwargs,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def _build_chat_text(
        self,
        prompt: str,
        system: Optional[str],
        history: Optional[List[Dict[str, str]]],
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            for turn in history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})


        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        return chat_text

    def query(
        self,
        prompt: str or List[str],
        system: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        stop: Optional[List[str]] = None,     
        stop_token_ids: Optional[List[int]] = None,
        seed: Optional[int] = None,
        n: int = 1,
        **generate_kwargs: Any,
    ) -> Dict[str, Any]:
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop=stop,
            stop_token_ids=stop_token_ids,
            seed=seed,
            **generate_kwargs,
        )
        if type(prompt) is str:
            chat_text = self._build_chat_text(prompt, system, history)
            outputs = self.llm.generate(prompts=[chat_text], sampling_params=sampling_params)
            out = outputs[0]
            text = out.outputs[0].text if out.outputs else ""
            finish_reason = out.outputs[0].finish_reason if out.outputs else "length"
            return {
                "text": text.strip(),
                "finish_reason": finish_reason,
                "raw": out,
            }
        else:
            prompts = []
            for prm, his in zip(prompt,history):
                chat_text = self._build_chat_text(prm, system, his)
                prompts.append(chat_text)
            outputs = self.llm.generate(prompts=prompts, sampling_params=sampling_params)
            out_text = []
            fin_rea = []
            for out in outputs:
                out_text.append(out.outputs[0].text)
                finish_reason = out.outputs[0].finish_reason if out.outputs else "length"
                fin_rea.append(finish_reason)
            return {
                "text": out_text,
                "finish_reason": fin_rea,
                "raw": outputs,
            }
        
    
    def extract_answer(self, predicted_answer: str, correct_answer: str) -> Tuple[bool, str]:

        def _extract_only_options(text: str) -> set[str]:
            text = text.lower()
            in_parens = re.findall(r'\(([a-d])\)', text)
            if in_parens:
                return set(in_parens)
            return set(re.findall(r'\b([a-d])\b', text))

        correct = correct_answer.lower().strip("() ")

        full_response = predicted_answer

        predicted_answer = predicted_answer.strip()
        if "<final_answer>" in predicted_answer:
            predicted_answer = predicted_answer.split("<final_answer>")[-1].strip()
        if predicted_answer.endswith("</final_answer>"):
            predicted_answer = predicted_answer[:-len("</final_answer>")].strip()
        pred_options = _extract_only_options(predicted_answer)
        if pred_options == {correct}:
            return True, predicted_answer

        response_options = _extract_only_options(full_response)
        if response_options == {correct}:
            return True, predicted_answer

        return False, predicted_answer

if __name__ == "__main__":

    print("CUDA available:", torch.cuda.is_available())

    llm = VLLMQwen(
        model="./LLM_src/Qwen3-0.6B",
        dtype="auto",                
        tensor_parallel_size=1,       
        gpu_memory_utilization=0.2,
    )

    ans = llm.query(
        prompt="Describe the memory augumented LLM",
        system="You are a helpful assistant.",
        max_tokens=256,
        temperature=0.6,
        top_p=0.9,
        stop=["\nuser:"],         
        seed=42,
    )
    print(ans["text"])

    # huggingface-cli download --resume-download Qwen/Qwen3-8B --local-dir Qwen3-8B