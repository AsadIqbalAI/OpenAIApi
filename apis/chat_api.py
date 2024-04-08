import argparse
import markdown2
import os
import sys
import uvicorn
import requests
from fastapi import UploadFile, File
os.system('pip install python-multipart')

from pathlib import Path
from typing import Union

from fastapi import FastAPI, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from messagers.message_composer import MessageComposer
from mocks.stream_chat_mocker import stream_chat_mock
from networks.message_streamer import MessageStreamer
from utils.logger import logger
from constants.models import AVAILABLE_MODELS_DICTS
import json


class ChatAPIApp:
    def __init__(self):
        self.app = FastAPI(
            docs_url="/",
            title="HuggingFace LLM API",
            swagger_ui_parameters={"defaultModelsExpandDepth": -1},
            version="1.0",
        )
        self.setup_routes()

    def get_available_models(self):
        return {"object": "list", "data": AVAILABLE_MODELS_DICTS}

    def extract_api_key(
        credentials: HTTPAuthorizationCredentials = Depends(
            HTTPBearer(auto_error=False)
        ),
    ):
        api_key = None
        if credentials:
            api_key = credentials.credentials
        else:
            api_key = os.getenv("HF_TOKEN")

        if api_key:
            if api_key.startswith("hf_"):
                return api_key
            else:
                logger.warn(f"Invalid HF Token!")
        else:
            logger.warn("Not provide HF Token!")
        return None

    class ChatCompletionsPostItem(BaseModel):
        model: str = Field(
            default="mixtral-8x7b",
            description="(str) `mixtral-8x7b`",
        )
        messages: list = Field(
            default=[{"role": "user", "content": "Hello, who are you?"}],
            description="(list) Messages",
        )
        temperature: Union[float, None] = Field(
            default=0.5,
            description="(float) Temperature",
        )
        top_p: Union[float, None] = Field(
            default=0.95,
            description="(float) top p",
        )
        max_tokens: Union[int, None] = Field(
            default=-1,
            description="(int) Max tokens",
        )
        use_cache: bool = Field(
            default=True,
            description="(bool) Use cache",
        )
        stream: bool = Field(
            default=False,
            description="(bool) Stream",
        )

    def chat_completions(
        self, item: ChatCompletionsPostItem, api_key: str = Depends(extract_api_key)
    ):
        streamer = MessageStreamer(model=item.model)
        composer = MessageComposer(model=item.model)
        composer.merge(messages=item.messages)
        # streamer.chat = stream_chat_mock

        stream_response = streamer.chat_response(
            prompt=composer.merged_str,
            temperature=item.temperature,
            top_p=item.top_p,
            max_new_tokens=item.max_tokens,
            api_key=api_key,
            use_cache=item.use_cache,
        )
        if item.stream:
            event_source_response = EventSourceResponse(
                streamer.chat_return_generator(stream_response),
                media_type="text/event-stream",
                ping=2000,
                ping_message_factory=lambda: ServerSentEvent(**{"comment": ""}),
            )
            return event_source_response
        else:
            data_response = streamer.chat_return_dict(stream_response)
            return data_response

    def get_readme(self):
        readme_path = Path(__file__).parents[1] / "README.md"
        with open(readme_path, "r", encoding="utf-8") as rf:
            readme_str = rf.read()
        readme_html = markdown2.markdown(
            readme_str, extras=["table", "fenced-code-blocks", "highlightjs-lang"]
        )
        return readme_html

    def summarize(self, data: dict):  # Updated parameter type to 'dict'
        text = data.get("text", "")  # Using get() to handle missing key gracefully
        word_count = len(text.split())

        if word_count > 500:
            return JSONResponse(content={"error": "Text exceeds 500 words"}, status_code=400)

        api_url = "https://openaiapi-ytg7.onrender.com/api/v1/chat/completions"
        payload = {
            "model": "mixtral-8x7b",
            "messages": [
                {"role": "assistant", "content": "You are good at summarizing long and lengthy text into some piece of lines."},
                {"role": "user", "content": f"Summarize this text now: {text}"},
            ],
            "temperature": 0.5,
            "top_p": 0.95,
            "max_tokens": -1,
            "use_cache": True,
            "stream": False,
        }

        try:
            response = requests.post(api_url, json=payload)
            response.raise_for_status()
            result = response.json()

            if "choices" in result and result["choices"]:
                return {"summary": result["choices"][0]["message"]["content"]}
            else:
                return JSONResponse(content={"error": "Invalid API response"}, status_code=500)

        except requests.RequestException as e:
            return JSONResponse(content={"error": f"Request error: {str(e)}"}, status_code=500)

    from fastapi import FastAPI, File, UploadFile, HTTPException
        
    async def process_file(self, file: UploadFile = File(...)):
            try:
                # Read the file content as bytes
                file_content = file.file.read()
    
                # Send the file content to the API
                API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
                headers = {"Authorization": "Bearer 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"}
                response = requests.post(API_URL, headers=headers, data=file_content)
    
                # Check if the request was successful
                response.raise_for_status()
    
                # Return the API response
                return response.json()
            except Exception as e:
                # If there's an error, return an HTTPException with a status code of 500
                raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


    def explain_complex(self, data: dict): 
        topic = data.get("topic", "")  

        api_url = "https://openaiapi-ytg7.onrender.com/api/v1/chat/completions"
        payload = {
            "model": "mixtral-8x7b",
            "messages": [
                {"role": "assistant", "content": "You are good at explaining difficult topics like cloud computing etc. Explain the topic for 5 years old. make sure to also provided the examples where required."},
                {"role": "user", "content": f"Explain this topic for 5 years od: {topic}"},
            ],
            "temperature": 0.5,
            "top_p": 0.95,
            "max_tokens": -1,
            "use_cache": True,
            "stream": False,
        }

        try:
            response = requests.post(api_url, json=payload)
            response.raise_for_status()
            result = response.json()

            if "choices" in result and result["choices"]:
                return {"explanation": result["choices"][0]["message"]["content"]}
            else:
                return JSONResponse(content={"error": "Invalid API response"}, status_code=500)

        except requests.RequestException as e:
            return JSONResponse(content={"error": f"Request error: {str(e)}"}, status_code=500)



    def setup_routes(self):
        for prefix in ["", "/v1", "/api", "/api/v1"]:
            if prefix in ["/api/v1"]:
                include_in_schema = True
            else:
                include_in_schema = False

            self.app.get(
                prefix + "/models",
                summary="Get available models",
                include_in_schema=include_in_schema,
            )(self.get_available_models)

            self.app.post(
                prefix + "/chat/completions",
                summary="Chat completions in conversation session",
                include_in_schema=include_in_schema,
            )(self.chat_completions)
            
            self.app.post(
                prefix + "/explain_complex",
                summary="Explain complex text",
                response_model=dict,
                include_in_schema=include_in_schema,
            )(self.explain_complex)

            self.app.post(
                prefix + "/summarize",
                summary="Summarize text",
                response_model=dict,
                include_in_schema=include_in_schema,
            )(self.summarize)
            self.app.post(
                prefix + "/whisper_v3",
                summary="Send a whisper",
                response_model=dict,
                include_in_schema=include_in_schema,
            )(self.process_file)


        self.app.get(
            "/readme",
            summary="README of HF LLM API",
            response_class=HTMLResponse,
            include_in_schema=False,
        )(self.get_readme)


class ArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(ArgParser, self).__init__(*args, **kwargs)

        self.add_argument(
            "-s",
            "--server",
            type=str,
            default="0.0.0.0",
            help="Server IP for HF LLM Chat API",
        )
        self.add_argument(
            "-p",
            "--port",
            type=int,
            default=23333,
            help="Server Port for HF LLM Chat API",
        )

        self.add_argument(
            "-d",
            "--dev",
            default=False,
            action="store_true",
            help="Run in dev mode",
        )

        self.args = self.parse_args(sys.argv[1:])


app = ChatAPIApp().app

if __name__ == "__main__":
    args = ArgParser().args
    if args.dev:
        uvicorn.run("__main__:app", host=args.server, port=args.port, reload=True)
    else:
        uvicorn.run("__main__:app", host=args.server, port=args.port, reload=False)

    # python -m apis.chat_api      # [Docker] on product mode
    # python -m apis.chat_api -d   # [Dev]    on develop mode
