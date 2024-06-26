MODEL_MAP = {
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # [Recommended]
    "nous-mixtral-8x7b": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "openchat-3.5": "openchat/openchat-3.5-0106",
    "gemma-7b": "google/gemma-1.1-7b-it",
    "command-r-plus": "CohereForAI/c4ai-command-r-plus",
    "default": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "zephyr-orpo-GPT4":"HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",
    "llama-3-70b":"meta-llama/Meta-Llama-3-70B-Instruct",

}

AVAILABLE_MODELS = list(MODEL_MAP.keys())

STOP_SEQUENCES_MAP = {
    "mixtral-8x7b": "</s>",
    "llama-3-70b": "<|eot_id|>",
    "zephyr-orpo-GPT4":"</s>",
    "nous-mixtral-8x7b": "<|im_end|>",
    "mistral-7b": "</s>",
    "openchat-3.5": "<|end_of_turn|>",
    "gemma-7b": "<eos>",
    "command-r-plus": "<|END_OF_TURN_TOKEN|>",
}

TOKEN_LIMIT_MAP = {
    "zephyr-orpo-GPT4":32768,
    "mixtral-8x7b": 32768,
    "nous-mixtral-8x7b": 32768,
    "mistral-7b": 32768,
    "openchat-3.5": 8192,
    "llama-3-70b": 8192,
    "gemma-7b": 8192,
    "gpt-3.5-turbo": 8192,
    "command-r-plus": 32768,
}

TOKEN_RESERVED = 20


# https://platform.openai.com/docs/api-reference/models/list
AVAILABLE_MODELS_DICTS = [
    {
        "id": "mixtral-8x7b",
        "description": "[mistralai/Mixtral-8x7B-Instruct-v0.1]: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1",
        "object": "model",
        "owned_by": "mistralai",
    },
    {
        "id": "nous-mixtral-8x7b",
        "description": "[NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO]: https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "object": "model",
        "owned_by": "NousResearch",
    },
    {
        "id": "mistral-7b",
        "description": "[mistralai/Mistral-7B-Instruct-v0.2]: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2",
        "object": "model",
        "owned_by": "mistralai",
    },
    {
        "id": "openchat-3.5",
        "description": "[openchat/openchat-3.5-0106]: https://huggingface.co/openchat/openchat-3.5-0106",
        "object": "model",
        "owned_by": "openchat",
    },
    {
        "id": "gemma-7b",
        "description": "[gemma-1.1-7b-it]: https://huggingface.co/google/gemma-1.1-7b-it",
        "object": "model",
        "owned_by": "Google",
    },
    {
        "id": "command-r-plus",
        "description": "[CohereForAI/c4ai-command-r-plus]: https://huggingface.co/CohereForAI/c4ai-command-r-plus",
        "object": "model",
        "owned_by": "CohereForAI",
    },
    {
        "id":"zephyr-orpo-GPT4",
        "desciption":"[HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1]: https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",
        "object":"model",
        "owned_by":"HuggingFace",
    },
    {
        "id": "llama-3-70b",
        "desciption":"[meta-llama/Meta-Llama-3-70B-Instruct]: https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct",
        "object": "model",
        "owned_by": "Meta",
    },
]
