import http.client
import json
import aiohttp
import asyncio

class OpenAIGPT:
    def __init__(self, api_key, model) -> None:
        api_key = ""
        self.api_key = api_key
        
        self.model = model
        self.client = http.client.HTTPSConnection("us.vveai.com")
        self.headers = {
                'Authorization': 'Bearer '+ api_key,
                'Content-Type': 'application/json'
                }
    def generate(self, prompt):
        if "system" in prompt:
            payload = json.dumps({
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": prompt['system']},
                            {"role": "user", "content": prompt['instruction']}
                        ],
                        "temperature": 0,
                        "stream": False
                        })
            
            self.client.request("POST", "/v1/chat/completions", payload, self.headers)
            res = self.client.getresponse()
            response = res.read()
            res_json = json.loads(response.decode("utf-8"))
        else:
            payload = json.dumps({
                        "model": self.model,
                        "messages": [
                            {"role": "user", "content": prompt['instruction']}
                        ],
                        "temperature": 0,
                        "stream": False
                        })
            
            self.client.request("POST", "/v1/chat/completions", payload, self.headers)
            res = self.client.getresponse()
            response = res.read()
            res_json = json.loads(response.decode("utf-8"))
        return res_json['choices'][0]["message"]["content"].strip()
    async def async_generate(self, prompt):
        url = "https://api.gpt.ge/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt['system']} if "system" in prompt else None,
                {"role": "user", "content": prompt['instruction']}
            ],
            "temperature": 0,
            "stream": False
        }
        payload["messages"] = [msg for msg in payload["messages"] if msg]  # Remove None values

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=self.headers) as response:
                if response.status != 200:
                    raise Exception(f"Request failed with status {response.status}")
                res_json = await response.json()
                return res_json['choices'][0]["message"]["content"].strip()
