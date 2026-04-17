import time

import litellm
import numpy as np
import torch
from transformers import AutoTokenizer, LlamaForSequenceClassification

from utils.constants import ATTACK_RM_PATH, SUPPORT_RM_PATH, google_api_key
from utils.timing_log import log_timing
from utils.tool import logger

safety_setting = [
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


def HelperClient(
    prompt,
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    temperature=0.7,
    max_tokens=1000,
    n=1,
    stop=None,
    sys=None,
) -> list:
    if sys is not None:
        messages = [{"role": "system", "content": sys}]
    else:
        messages = []

    kwargs = {}
    if "llama" in model.lower():
        model_name = f"together_ai/{model}"
    elif "deepseek" in model.lower():
        model_name = f"deepseek/{model}"
    elif "gemini" in model.lower():
        model_name = f"gemini/{model}"
        kwargs = {"api_key": google_api_key, "safety_settings": safety_setting}
    elif "gpt" in model.lower() or "o1" in model.lower():
        model_name = model
    elif "moonshot" in model.lower() or "kimi" in model.lower():
        # Kimi/Moonshot API support
        import os

        model_name = f"moonshot/{model}"
        # Reduce max_tokens for moonshot models to avoid exceeding limits
        max_tokens = min(max_tokens, 4096)
        kwargs = {"api_key": os.environ.get("MOONSHOT_API_KEY", ""), "api_base": "https://api.moonshot.cn/v1"}
        # print(f"[HelperClient] Using Moonshot model: {model_name}")
        # print(f"Moonshot API Key: {kwargs['api_key'][:5]}****")
    else:
        raise NotImplementedError(f"{model} is not supported.")

    messages.append({"role": "user", "content": prompt})
    responses = []
    for i in range(n):
        t0 = time.perf_counter()
        # # Check if we need JSON response format and if the model supports it
        # use_json_format = ("json" in prompt.lower() or (sys is not None and "json" in sys.lower()))

        # # Only use response_format for models that support it
        # if use_json_format and ("gpt-4o" in model_name.lower() or "gpt-4-turbo" in model_name.lower() or "gpt-3.5-turbo" in model_name.lower()):
        if "json" in prompt.lower() or (sys is not None and "json" in sys.lower()):
            response = litellm.completion(
                model=model_name,
                response_format={"type": "json_object"},
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                **kwargs,
            )
        else:
            response = litellm.completion(
                model=model_name, messages=messages, temperature=temperature, max_tokens=max_tokens, stop=stop, **kwargs
            )
        elapsed = time.perf_counter() - t0
        ctx = {"model": model, "n_index": i + 1, "max_tokens": max_tokens}
        try:
            cost = getattr(response, "_hidden_params", {}).get("response_cost")
            if cost is not None:
                ctx["response_cost"] = cost
        except Exception:
            pass
        log_timing(logger, "helper_client_litellm", elapsed, **ctx)
        responses.append(response.choices[0].message.content)
    return responses


models_loaded = False
pro_model = None
con_model = None
tokenizer = None


class RM:
    def __init__(self, model_name):
        # Check if the model path exists locally
        import os

        if os.path.exists(model_name):
            # Load from local path
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = LlamaForSequenceClassification.from_pretrained(
                model_name, num_labels=3, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
            )
        else:
            # Try to load from Hugging Face Hub
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = LlamaForSequenceClassification.from_pretrained(
                model_name, num_labels=3, torch_dtype=torch.bfloat16, device_map="auto"
            )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def __call__(self, prompt: str, soft=False, temperature=0.7, max_tokens=1000, n=1) -> float:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
            self.model.device
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        if soft:  # 平滑计分
            p = torch.softmax(outputs.logits, dim=-1).tolist()
            p = np.array(p[0])
            score = p * np.array([0, 1, 2])
            score = score.sum(axis=-1)
            return score.item()
        else:
            print(torch.argmax(outputs.logits, dim=-1).item())
            return torch.argmax(outputs.logits, dim=-1).item()


def reward_model(prompt, type="pro", temperature=0.7, max_tokens=1000, n=1, soft=False):  # type is "pro" / "con"
    global pro_model, con_model, models_loaded
    if not models_loaded:
        logger.info("Logging reward model ...")
        pro_model = RM(SUPPORT_RM_PATH)
        con_model = RM(ATTACK_RM_PATH)
        models_loaded = True
    if type == "pro":
        return pro_model(prompt, soft, temperature, max_tokens, n)
    elif type == "con":
        return con_model(prompt, soft, temperature, max_tokens, n)
