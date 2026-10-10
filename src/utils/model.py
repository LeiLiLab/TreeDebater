import time
import math
import os
from typing import Any, Type

import litellm
import numpy as np
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer, LlamaForSequenceClassification

from utils.constants import ATTACK_RM_PATH, SUPPORT_RM_PATH, google_api_key
from utils.timing_log import log_timing
from utils.tool import logger
# litellm._turn_on_debug()

try:
    import instructor
except Exception:
    instructor = None


# Retired Deepseek API ids -> current v4 models (API only accepts deepseek-v4-pro / deepseek-v4-flash).
_DEEPSEEK_LEGACY_ALIASES = {
    "deepseek-chat": "deepseek-v4-flash",
    "deepseek-chat-v3": "deepseek-v4-flash",
    "chat": "deepseek-v4-flash",
}


def normalize_deepseek_litellm_model(model: str) -> str:
    """Map config model strings to litellm ``deepseek/<api-model>`` without double-prefixing."""
    raw = model.strip()
    if "/" in raw:
        provider, api_model = raw.split("/", 1)
        if provider.lower() != "deepseek":
            api_model = raw
    else:
        api_model = raw

    resolved = _DEEPSEEK_LEGACY_ALIASES.get(api_model.lower(), api_model)
    if resolved != api_model:
        logger.warning("Deepseek model %r is deprecated; using %r", api_model, resolved)
    return f"deepseek/{resolved}"


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


def helper_messages(prompt, *, sys=None, history_messages=None):
    """Assemble an owned request while keeping spoken history in its chat roles."""
    messages = [{"role": "system", "content": sys}] if sys is not None else []
    for entry in history_messages or ():
        if (entry.get('role') not in ('user', 'assistant')
                or not isinstance(entry.get('content'), str)):
            raise ValueError('Helper history must contain user/assistant text messages')
        messages.append({"role": entry['role'], "content": entry['content']})
    messages.append({"role": "user", "content": prompt})
    return messages


class CompletionResults(list):
    """List-compatible helper output carrying provider-reported request cost."""
    response_cost = 0.0


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
    json_mode: bool | None = None,
    request_timeout: float | None = None,
    history_messages=None,
    on_text=None,
) -> list[str] | list[BaseModel]:
    if json_mode is False and response_model is not None:
        raise ValueError('A structured response_model cannot use json_mode=False')
    if on_text is not None and (not callable(on_text) or n != 1 or response_model is not None or json_mode is not False):
        raise ValueError('Text streaming requires one plain-text response and a callable on_text')
    messages = helper_messages(prompt, sys=sys, history_messages=history_messages)

    kwargs = {}
    if os.environ.get("DEBATE_LLM_API_BASE"):
        model_name = model if model.startswith("openai/") else "openai/" + model
        kwargs = {"api_base": os.environ["DEBATE_LLM_API_BASE"],
                  "api_key": os.environ.get("DEBATE_LLM_API_KEY", "local-proxy")}
    elif "llama" in model.lower():
        model_name = f"together_ai/{model}"
    elif "deepseek" in model.lower():
        model_name = normalize_deepseek_litellm_model(model)
        if "flash" in model_name:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    elif "gemini" in model.lower():
        model_name = f"gemini/{model}"
        kwargs = {"api_key": google_api_key, "safety_settings": safety_setting}
    elif "gpt" in model.lower() or "o1" in model.lower():
        model_name = model
    elif "moonshot" in model.lower() or "kimi" in model.lower():
        # Kimi/Moonshot API support
        model_name = f"moonshot/{model}"
        # Reduce max_tokens for moonshot models to avoid exceeding limits
        max_tokens = min(max_tokens, 4096)
        kwargs = {"api_key": os.environ.get("MOONSHOT_API_KEY", ""), "api_base": "https://api.moonshot.cn/v1"}
    else:
        raise NotImplementedError(f"{model} is not supported.")

    if request_timeout is not None:
        if isinstance(request_timeout, bool) or not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError('request_timeout must be finite and positive')
        kwargs.update(timeout=request_timeout, num_retries=0)
    responses = CompletionResults()
    for i in range(n):
        t0 = time.perf_counter()
        # Speech prompts may mention JSON as input data or explicitly forbid it.
        # Let callers state the output contract instead of relying on that word.
        wants_json = json_mode if json_mode is not None else (
            "json" in prompt.lower() or (sys is not None and "json" in sys.lower()))
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
                **({'on_text': on_text} if on_text is not None else {}),
            )

        elapsed = time.perf_counter() - t0
        ctx = {"model": model, "n_index": i + 1, "max_tokens": max_tokens}
        if response is not None:
            try:
                cost = getattr(response, "_hidden_params", {}).get("response_cost")
                if cost is not None:
                    ctx["response_cost"] = cost
                    responses.response_cost += cost
            except Exception:
                pass
        log_timing(logger, "helper_client_litellm", elapsed, **ctx)
        if response_model is not None:
            responses.append(structured_value if structured_value is not None else response.choices[0].message.content)
        else:
            responses.append(response.choices[0].message.content)
    return responses


def _deepseek_thinking_model(model_name: str) -> bool:
    """Deepseek V4/reasoner models use thinking mode and reject instructor tool_choice."""
    name = model_name.lower()
    if "deepseek" not in name:
        return False
    return any(marker in name for marker in ("v4", "reasoner", "deepseek-r1", "/r1"))


def _supports_structured_output(model_name: str) -> bool:
    name = model_name.lower()
    if _deepseek_thinking_model(name):
        return False
    if "deepseek" in name:
        return True
    return any(x in name for x in ["gpt", "o1", "claude", "gemini"])


def _completion_text(model_name: str, messages, wants_json: bool, temperature: float, max_tokens: int, stop, kwargs, on_text=None):
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
    retries = call_kwargs.pop('num_retries', 3)
    if on_text is not None:
        # A retry after any delivered delta would repeat already committed speech.
        call_kwargs.setdefault('timeout', 60)
        stream = litellm.completion(num_retries=0, stream=True,
                                    stream_options={'include_usage': True}, **call_kwargs)
        chunks = []
        finished = False
        try:
            for chunk in stream:
                chunks.append(chunk)
                for choice in chunk.choices:
                    if choice.finish_reason is not None:
                        if choice.finish_reason != 'stop':
                            raise ValueError(f'Speech stream did not finish normally: {choice.finish_reason}')
                        finished = True
                    delta = getattr(choice.delta, 'content', None)
                    if delta:
                        on_text(delta)
            if not finished:
                raise ValueError('Speech stream ended without a completion marker')
            response = litellm.stream_chunk_builder(chunks, messages=messages)
            cost = next((getattr(c, '_hidden_params', {}).get('response_cost') for c in reversed(chunks)
                         if getattr(c, '_hidden_params', {}).get('response_cost') is not None), None)
            if cost is None:
                try:
                    cost = litellm.completion_cost(completion_response=response, model=model_name)
                except Exception:
                    pass  # Same unknown-cost behavior as a non-streaming provider response.
            if cost is not None:
                response._hidden_params['response_cost'] = cost
            return response
        finally:
            close = getattr(stream, 'close', None)
            if close is not None:
                close()
    return litellm.completion(num_retries=retries, **call_kwargs)


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
