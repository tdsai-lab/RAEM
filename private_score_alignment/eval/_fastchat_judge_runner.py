"""Run FastChat's ``gen_judgment`` after shimming its openai 0.28 API onto openai>=1.0.

FastChat 0.2.36 calls ``openai.ChatCompletion.create`` and ``openai.error.OpenAIError``,
both removed in openai>=1.0. This wrapper monkey-patches ``fastchat.llm_judge.common``
in-process before invoking gen_judgment, avoiding both an openai downgrade and any edits
to the installed fastchat package.
"""

from __future__ import annotations

import os
import runpy
import sys
import time

import fastchat.llm_judge.common as fc_common

try:
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit(f"openai>=1.0 is required for the FastChat judge shim: {exc}") from exc


def _build_client(api_dict: dict | None = None) -> OpenAI:
    api_dict = api_dict or {}
    kwargs: dict[str, str] = {}
    api_key = api_dict.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        kwargs["api_key"] = api_key
    base_url = api_dict.get("api_base") or os.environ.get("OPENAI_API_BASE", "")
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    return OpenAI(**kwargs)


def _chat_completion_openai(model, conv, temperature, max_tokens, api_dict=None):
    client = _build_client(api_dict)
    output = fc_common.API_ERROR_OUTPUT
    for _ in range(fc_common.API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response.choices[0].message.content
            break
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            time.sleep(fc_common.API_RETRY_SLEEP)
    return output


def _chat_completion_openai_azure(model, conv, temperature, max_tokens, api_dict=None):
    azure_dict = dict(api_dict or {})
    azure_dict.setdefault("api_base", os.environ.get("AZURE_OPENAI_ENDPOINT", ""))
    azure_dict.setdefault("api_key", os.environ.get("AZURE_OPENAI_KEY", ""))
    if model.startswith("azure-"):
        model = model[6:]
    return _chat_completion_openai(model, conv, temperature, max_tokens, api_dict=azure_dict)


fc_common.chat_completion_openai = _chat_completion_openai
fc_common.chat_completion_openai_azure = _chat_completion_openai_azure

_compat_model = os.environ.get("PSA_OPENAI_COMPAT_MODEL", "")
if _compat_model:
    try:
        import fastchat.model.model_adapter as _adapter

        if _compat_model not in _adapter.OPENAI_MODEL_LIST:
            _adapter.OPENAI_MODEL_LIST = tuple(_adapter.OPENAI_MODEL_LIST) + (_compat_model,)
    except Exception as exc:
        print(f"warning: could not register {_compat_model} with fastchat: {exc}", file=sys.stderr)

try:
    import tiktoken

    _orig_encoding_for_model = tiktoken.encoding_for_model

    def _encoding_for_model(name):
        try:
            return _orig_encoding_for_model(name)
        except KeyError:
            if str(name).startswith("gemini-"):
                return tiktoken.get_encoding("cl100k_base")
            raise

    tiktoken.encoding_for_model = _encoding_for_model
except Exception:
    pass


runpy.run_module("fastchat.llm_judge.gen_judgment", run_name="__main__", alter_sys=True)
