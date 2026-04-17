import time
from typing import Any, Type

import litellm
import numpy as np
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer, LlamaForSequenceClassification

from utils.constants import ATTACK_RM_PATH, SUPPORT_RM_PATH, google_api_key
from utils.timing_log import log_timing
from utils.tool import logger

try:
    import instructor
except Exception:
    instructor = None

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
    response_model: Type[BaseModel] | None = None,
    use_instructor: bool | None = None,
) -> list[str] | list[BaseModel]:
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
        wants_json = "json" in prompt.lower() or (sys is not None and "json" in sys.lower())
        structured_enabled = response_model is not None and (
            use_instructor is True or (use_instructor is None and _supports_structured_output(model_name))
        )
        response = None
        structured_value = None
        if structured_enabled:
            try:
                structured_value = _completion_structured(
                    model_name=model_name,
                    messages=messages,
                    response_model=response_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    kwargs=kwargs,
                )
            except Exception as e:
                logger.warning(f"Structured output fallback for {model_name}: {e}")

        if structured_value is None:
            response = _completion_text(
                model_name=model_name,
                messages=messages,
                wants_json=wants_json or response_model is not None,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                kwargs=kwargs,
            )

        elapsed = time.perf_counter() - t0
        ctx = {"model": model, "n_index": i + 1, "max_tokens": max_tokens}
        if response is not None:
            try:
                cost = getattr(response, "_hidden_params", {}).get("response_cost")
                if cost is not None:
                    ctx["response_cost"] = cost
            except Exception:
                pass
        log_timing(logger, "helper_client_litellm", elapsed, **ctx)
        if response_model is not None:
            responses.append(structured_value if structured_value is not None else response.choices[0].message.content)
        else:
            responses.append(response.choices[0].message.content)
    return responses


def _supports_structured_output(model_name: str) -> bool:
    name = model_name.lower()
    return any(x in name for x in ["gpt", "o1", "claude", "gemini"])


def _completion_text(model_name: str, messages, wants_json: bool, temperature: float, max_tokens: int, stop, kwargs):
    call_kwargs: dict[str, Any] = dict(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
        **kwargs,
    )
    if wants_json:
        call_kwargs["response_format"] = {"type": "json_object"}
    return litellm.completion(**call_kwargs)


def _completion_structured(
    model_name: str,
    messages,
    response_model: Type[BaseModel],
    temperature: float,
    max_tokens: int,
    stop,
    kwargs,
) -> BaseModel:
    if instructor is None:
        raise RuntimeError("instructor is not installed.")

    if hasattr(instructor, "from_litellm"):
        client = instructor.from_litellm(litellm.completion)
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_model=response_model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            **kwargs,
        )
    raise RuntimeError("Installed instructor version does not support from_litellm.")


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
