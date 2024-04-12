import json
import re
import requests

from tiktoken import get_encoding as tiktoken_get_encoding
from transformers import AutoTokenizer

from constants.models import (
    MODEL_MAP,
    STOP_SEQUENCES_MAP,
    TOKEN_LIMIT_MAP,
    TOKEN_RESERVED,
)
from messagers.message_outputer import OpenaiStreamOutputer
from utils.enver import enver


class MessageStreamer:

    def __init__(self, model: str):
        if model in MODEL_MAP.keys():
            self.model = model
        else:
            self.model = "default"
        self.model_fullname = MODEL_MAP[self.model]
        self.message_outputer = OpenaiStreamOutputer()

        if self.model == "gemma-7b":
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP["mistral-7b"])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_fullname)

    def parse_line(self, line):
        line = line.decode("utf-8")
        line = re.sub(r"data:\s*", "", line)
        data = json.loads(line)
        try:
            content = data["token"]["text"]
        except:
            pass
        return content

    def count_tokens(self, text):
        tokens = self.tokenizer.encode(text)
        token_count = len(tokens)
        return token_count

    def chat_response(
        self,
        prompt: str = None,
        temperature: float = 0.5,
        top_p: float = 0.95,
        max_new_tokens: int = None,
        api_key: str = None,
        use_cache: bool = False,
    ):
        self.request_url = (
            f"https://api-inference.huggingface.co/models/{self.model_fullname}"
        )
        self.request_headers = {
            "Content-Type": "application/json",
        }

        if api_key:
            self.request_headers["Authorization"] = f"Bearer {api_key}"

        if temperature is None or temperature < 0:
            temperature = 0.0
        temperature = max(temperature, 0.01)
        temperature = min(temperature, 0.99)
        top_p = max(top_p, 0.01)
        top_p = min(top_p, 0.99)

        token_limit = int(
            TOKEN_LIMIT_MAP[self.model] - TOKEN_RESERVED - self.count_tokens(prompt)
        )
        if token_limit <= 0:
            raise ValueError("Prompt exceeded token limit!")

        if max_new_tokens is None or max_new_tokens <= 0:
            max_new_tokens = token_limit
        else:
            max_new_tokens = min(max_new_tokens, token_limit)

        self.request_body = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "return_full_text": False,
            },
            "options": {
                "use_cache": use_cache,
            },
            "stream": True,
        }

        if self.model in STOP_SEQUENCES_MAP.keys():
            self.stop_sequences = STOP_SEQUENCES_MAP[self.model]

        stream_response = requests.post(
            self.request_url,
            headers=self.request_headers,
            json=self.request_body,
            proxies=enver.requests_proxies,
            stream=True,
        )
        status_code = stream_response.status_code
        if status_code == 200:
            pass
        else:
            pass

        return stream_response

    def chat_return_dict(self, stream_response):
        final_output = self.message_outputer.default_data.copy()
        final_output["choices"] = [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "",
                },
            }
        ]

        final_content = ""
        token_count = 0
        for line in stream_response.iter_lines():
            if not line:
                continue
            content = self.parse_line(line)

            if content.strip() == self.stop_sequences:
                break
            else:
                final_content += content
                token_count += self.count_tokens(content)

        if self.model in STOP_SEQUENCES_MAP.keys():
            final_content = final_content.replace(self.stop_sequences, "")

        final_content = final_content.strip()
        final_output["choices"][0]["message"]["content"] = final_content

        final_output["tokens_used"] = token_count

        return final_output
        
    def chat_return_generator(self, stream_response):
        is_finished = False
        line_count = 0
        for line in stream_response.iter_lines():
            if line:
                line_count += 1
            else:
                continue

            content = self.parse_line(line)

            if content.strip() == self.stop_sequences:
                content_type = "Finished"
                is_finished = True
            else:
                content_type = "Completions"
                if line_count == 1:
                    content = content.lstrip()

            output = self.message_outputer.output(
                content=content, content_type=content_type
            )
            yield output

        if not is_finished:
            yield self.message_outputer.output(content="", content_type="Finished")
