from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple, Union
import requests, time, re, os
import asyncio, aiohttp, random, json
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

class APILLMClient:

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        base_url: str = "https://api.v3.cm/v1/",
        timeout: int = 60,
        organization: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        backoff_base: float = 0.8,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.organization = organization
        self.extra_headers = extra_headers or {}
        self.max_retries = max_retries
        self.backoff_base = backoff_base


        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def _build_messages(
        self,
        prompt: str,
        system: Optional[str],
        history: Optional[List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            for turn in history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _post_chat_completions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = resp.text
                    time.sleep(self.backoff_base * (2 ** (attempt - 1)))
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                last_err = str(e)
                time.sleep(self.backoff_base * (2 ** (attempt - 1)))
        raise RuntimeError(f"Error and retry {self.max_retries} times: {last_err}")

    def query(
        self,
        prompt: Union[str, List[str]],
        system: Optional[str] = None,
        history: Optional[Union[List[Dict[str, str]], List[List[Dict[str, str]]]]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        n: int = 1,
        **generate_kwargs: Any,
    ) -> Dict[str, Any]:
        if isinstance(prompt, str):
            messages = self._build_messages(prompt, system, history if isinstance(history, list) else None)
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "n": n,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                **({"stop": stop} if stop else {}),
                **({"seed": seed} if seed is not None else {}),
                **generate_kwargs,
            }
            data = self._post_chat_completions(payload)
            choice0 = (data.get("choices") or [{}])[0]
            msg = (choice0.get("message") or {}).get("content", "") or ""
            finish_reason = choice0.get("finish_reason", "length")
            return {"text": msg.strip(), "finish_reason": finish_reason, "raw": data}
        
    async def aquery(
        self,
        prompt: Union[str, List[str]],
        system: Optional[str] = None,
        history: Optional[Union[List[Dict[str, str]], List[List[Dict[str, str]]]]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_concurrency: int = 20,
        n: int = 1,
        **generate_kwargs: Any,
    ) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        texts, reasons, raws = [], [], []

        sem = asyncio.Semaphore(max_concurrency)

        async with aiohttp.ClientSession(headers=headers) as session:
            N = len(prompt)
            texts: List[str] = ["" for _ in range(N)]
            reasons: List[str] = ["" for _ in range(N)]
            raws: List[Dict[str, Any]] = [{} for _ in range(N)]

            async def _worker(idx: int, prm: str, his: List[Dict[str, str]]):
                messages = self._build_messages(prm, system, his)
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "n": n,
                    **generate_kwargs,
                }
                async with sem:
                    async with session.post(url, json=payload) as resp:
                        data = await resp.json()
                choice0 = (data.get("choices") or [{}])[0]
                msg = (choice0.get("message") or {}).get("content", "") or ""
                finish_reason = choice0.get("finish_reason", "length")
                texts[idx] = msg.strip()
                reasons[idx] = finish_reason
                raws[idx] = data

            tasks = [asyncio.create_task(_worker(i, prm, his)) for i, (prm, his) in enumerate(zip(prompt, history))]

            await tqdm_asyncio.gather(*tasks, total=len(tasks))

        return {"text": texts, "finish_reason": reasons, "raw": raws}
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
    client = APILLMClient(
        api_key='',
        model="gpt-4.1-mini",
        base_url="https://api.v3.cm/v1/",  
    )

    ans = client.query(
        prompt="Describe the memory augumented LLM",
        system="You are a helpful assistant.",
        max_tokens=256,
        temperature=0.6,
        top_p=0.9,
        stop=["\nuser:"],
        seed=42,
    )
    print(ans["text"])

    async def test():
        ans = await client.aquery(
            prompt=["Describe the memory augumented LLM","Describe the retrieval augumented LLM"],
            system="You are a helpful assistant.",
            history = ['',''],
            max_tokens=256,
            temperature=0.6,
            top_p=0.9,
            stop=["\nuser:"],
            seed=42,
        )
        print(ans["text"])

    asyncio.run(test())