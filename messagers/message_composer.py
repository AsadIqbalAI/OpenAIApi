import re
from pprint import pprint
import requests
from transformers import AutoTokenizer

from constants.models import AVAILABLE_MODELS, MODEL_MAP
from utils.logger import logger

from datetime import datetime
import pytz

def get_current_date_time_with_gmt_and_day():
    # Get the current date and time in UTC
    current_date_time_utc = datetime.utcnow()

    # Convert UTC time to a specific timezone (e.g., GMT)
    gmt_timezone = pytz.timezone("GMT")
    current_date_time_gmt = current_date_time_utc.replace(tzinfo=pytz.utc).astimezone(gmt_timezone)

    # Get the day of the week
    day_of_week = current_date_time_gmt.strftime("%A")

    # Format the date and time string
    formatted_date_time = current_date_time_gmt.strftime("%Y-%m-%d %H:%M:%S %Z")

    return f"{formatted_date_time} (GMT) - {day_of_week}"


class MessageComposer:
    def __init__(self, model: str = None):
        if model in AVAILABLE_MODELS:
            self.model = model
        else:
            self.model = "mixtral-8x7b"
        self.model_fullname = MODEL_MAP[self.model]
        self.system_roles = ["system"]
        self.inst_roles = ["user", "system", "inst"]
        self.answer_roles = ["assistant", "bot", "answer", "model"]
        self.default_role = "user"

    def concat_messages_by_role(self, messages):
        def is_same_role(role1, role2):
            if (
                (role1 == role2)
                or (role1 in self.inst_roles and role2 in self.inst_roles)
                or (role1 in self.answer_roles and role2 in self.answer_roles)
            ):
                return True
            else:
                return False

        concat_messages = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if concat_messages and is_same_role(role, concat_messages[-1]["role"]):
                concat_messages[-1]["content"] += "\n" + content
            else:
                if role in self.inst_roles:
                    message["role"] = "inst"
                elif role in self.answer_roles:
                    message["role"] = "answer"
                else:
                    message["role"] = "inst"
                concat_messages.append(message)
        return concat_messages

    def merge(self, messages) -> str:
        # Templates for Chat Models
        # - https://huggingface.co/docs/transformers/main/en/chat_templating
        #   - https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1#instruction-format
        #   - https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO#prompt-format
        #   - https://huggingface.co/openchat/openchat-3.5-0106
        #   - https://huggingface.co/google/gemma-7b-it#chat-template

        # Mistral and Mixtral:
        #   <s> [INST] Instruction [/INST] Model answer </s> [INST] Follow-up instruction [/INST]

        # Nous Mixtral:
        #   <|im_start|>system
        #   You are "Hermes 2".<|im_end|>
        #   <|im_start|>user
        #   Hello, who are you?<|im_end|>
        #   <|im_start|>assistant

        # OpenChat:
        #   GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant: Hi<|end_of_turn|>GPT4 Correct User: How are you today?<|end_of_turn|>GPT4 Correct Assistant:

        # Google Gemma-it
        # <start_of_turn>user
        # How does the brain work?<end_of_turn>
        # <start_of_turn>model

        self.messages = messages
        self.merged_str = ""
        # Add default system prompt for each model
        today_date_time = get_current_date_time_with_gmt_and_day()
        default_system_prompt = f"You are helpful assistant build by OpenAI. Here is the latest real-time date and time for now: {today_date_time}."

        # https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1#instruction-format
        if self.model in ["mixtral-8x7b", "mistral-7b"]:
            self.messages = self.concat_messages_by_role(messages)
            self.cached_str = ""
            for message in self.messages:
                role = message["role"]
                content = message["content"]
                if role in self.inst_roles:
                    self.cached_str = f"[INST] system: {default_system_prompt}. \n{content} [/INST]"
                elif role in self.answer_roles:
                    self.cached_str += f"[INST]<s>system prompt: {default_system_prompt}</s> [/INST]"
                    self.merged_str += f"<s> {self.cached_str} {content} </s>\n"
                    self.cached_str = ""

                else:
                    self.cached_str = f"[INST] {content} [/INST]"
            if self.cached_str:
                self.merged_str += f"{self.cached_str}"
        # https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO#prompt-format
        elif self.model in ["nous-mixtral-8x7b"]:
            self.merged_str_list = []
            self.merged_str_list.append(f"<|im_start|>system\n{default_system_prompt} <|im_end|>")
            for message in self.messages:
                role = message["role"]
                content = message["content"]
                if role not in ["system", "user", "assistant"]:
                    role = self.default_role
                message_line = f"<|im_start|>{role}\n{content}<|im_end|>"
                self.merged_str_list.append(message_line)
            self.merged_str_list.append("<|im_start|>assistant")
            self.merged_str = "\n".join(self.merged_str_list)

        elif self.model in ["llama-3-70b"]:
            self.merged_str_list = []
            self.merged_str_list.append(f"<|eot_id|><|start_header_id|>system\n\n{default_system_prompt} <|end_header_id|><|eot_id|>")
            for message in self.messages:
                role = message["role"]
                content = message["content"]
                if role not in ["system", "user", "assistant"]:
                    role = self.default_role
                message_line = f"<|eot_id|><|start_header_id|>{role}\n\n{content}<|end_header_id|><|eot_id|>"
                self.merged_str_list.append(message_line)
            self.merged_str_list.append("<|eot_id|><|start_header_id|>assistant\n\n")
            self.merged_str = "\n".join(self.merged_str_list)

        elif self.model in ["zephyr-orpo-GPT4"]:
            self.merged_str_list = []
            self.merged_str_list.append(f"[INST][SYSTEM]\n{default_system_prompt}[/SYSTEM] [/INST]")
            for message in self.messages:
                role = message.get("role", self.default_role)
                content = message.get("content", "")
                if role not in ["system", "user", "assistant"]:
                    role = self.default_role
                message_line = f"[{role}]\n{content}[/{role}]"
                self.merged_str_list.append(message_line)
            self.merged_str_list.append("[ASSISTANT]")
            self.merged_str = "\n".join(self.merged_str_list)

            
        # https://huggingface.co/openchat/openchat-3.5-0106
        elif self.model in ["openchat-3.5"]:
            self.messages = self.concat_messages_by_role(messages)
            self.merged_str_list = []
            self.merged_str_list.append(f"GPT4 Correct System:\n{default_system_prompt}")
            self.end_of_turn = "<|end_of_turn|>"
            for message in self.messages:
                role = message["role"]
                content = message["content"]
                if role in self.inst_roles:
                    self.merged_str_list.append(
                        f"GPT4 Correct User:\n{content}{self.end_of_turn}"
                    )
                elif role in self.answer_roles:
                    self.merged_str_list.append(
                        f"GPT4 Correct Assistant:\n{content}{self.end_of_turn}"
                    )
                else:
                    self.merged_str_list.append(
                        f"GPT4 Correct User: {content}{self.end_of_turn}"
                    )
            self.merged_str_list.append(f"GPT4 Correct Assistant:\n")
            self.merged_str = "\n".join(self.merged_str_list)
        # https://huggingface.co/google/gemma-7b-it#chat-template
        elif self.model in ["gemma-7b"]:
            self.merged_str_list = []
            self.end_of_turn = "<end_of_turn>"
            self.start_of_turn = "<start_of_turn>"
            for message in self.messages:
                role = message["role"]
                content = message["content"]
                if role in self.inst_roles:
                    self.merged_str_list.append(
                        f"{self.start_of_turn}user\n{content}{self.end_of_turn}"
                    )
                elif role in self.answer_roles:
                    self.merged_str_list.append(
                        f"{self.start_of_turn}model\n{content}{self.end_of_turn}"
                    )
                else:
                    self.merged_str_list.append(
                        f"{self.start_of_turn}user\n{content}{self.end_of_turn}"
                    )
            self.merged_str_list.append(f"{self.start_of_turn}model\n")
            self.merged_str = "\n".join(self.merged_str_list)
        # https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO#prompt-format
        # https://huggingface.co/openchat/openchat-3.5-0106
        # elif self.model in ["openchat-3.5", "nous-mixtral-8x7b"]:
        elif self.model in ["openchat-3.5", "command-r-plus"]:
            tokenizer = AutoTokenizer.from_pretrained(self.model_fullname)
            self.merged_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            self.merged_str = "\n\n".join(
                [f"{message['role']}: {message['content']}" for message in messages]
            )


        return self.merged_str


if __name__ == "__main__":
    # model = "mixtral-8x7b"
    # model = "nous-mixtral-8x7b"
    # model = "gemma-7b"
    model = "openchat-3.5"
    composer = MessageComposer(model)
    messages = [
        {
            "role": "system",
            "content": "You are a LLM developed by OpenAI.\nYour name is GPT-4.",
        },
        {"role": "user", "content": "Hello, who are you?"},
        {"role": "assistant", "content": "I am a bot."},
        {"role": "user", "content": "What is your name?"},
        # {"role": "assistant", "content": "My name is Bing."},
        # {"role": "user", "content": "Tell me a joke."},
        # {"role": "assistant", "content": "What is a robot's favorite type of music?"},
        # {
        #     "role": "user",
        #     "content": "How many questions have I asked? Please list them.",
        # },
    ]
    logger.note(f"model: {composer.model}")
    merged_str = composer.merge(messages)
    logger.note("merged_str:")
    logger.mesg(merged_str)

    # python -m messagers.message_composer
