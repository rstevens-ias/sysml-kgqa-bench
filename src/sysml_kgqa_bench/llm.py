"""Chat-completion clients for the OpenAI API and OpenAI-compatible endpoints.

Backends:
    openai   OpenAI API, needs OPENAI_API_KEY.
    private  Any OpenAI-compatible endpoint (e.g. vLLM), needs PRIVATE_API_KEY and PRIVATE_ENDPOINT.
"""

import json
import os
import time

import dotenv
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, OpenAI, RateLimitError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from sysml_kgqa_bench.paths import ROOT

dotenv.load_dotenv(ROOT / ".env")
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
PRIVATE_API_KEY = os.getenv("PRIVATE_API_KEY")
PRIVATE_ENDPOINT = os.getenv("PRIVATE_ENDPOINT")

BACKENDS = ["openai", "private"]
JSON_ONLY_SUFFIX = "\n\nRespond with strictly valid JSON only, no other text."
_RETRYABLE_EXCEPTIONS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


def extract_json(text):
    """Parse the first balanced JSON object found in `text`."""
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object found in model output: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"unbalanced JSON object in model output: {text[:200]!r}")


def _parse_content(content, label, finish_reason):
    if not content:
        raise ValueError(f"empty content from {label} (finish_reason={finish_reason})")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return extract_json(content)


class LLMClient:
    def complete_json(self, prompt: str, temperature: float) -> dict:
        raise NotImplementedError

    def provenance(self) -> dict:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """Synchronous client that requests JSON output and returns the parsed object."""

    def __init__(
        self, api_key, model, base_url=None, backend_label="openai", timeout=120, max_tokens=4096, extra_body=None
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        self.model = model
        self.backend_label = backend_label
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        self._resolved_version = model

    def complete_json(self, prompt, temperature):
        label = f"{self.backend_label}/{self.model}"
        t0 = time.monotonic()
        print(f"    [{label}] calling (timeout={self.client.timeout}s, max_tokens={self.max_tokens})...", flush=True)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                extra_body=self.extra_body,
            )
        except Exception as e:
            print(f"    [{label}] {type(e).__name__}: {e}, retrying without response_format...", flush=True)
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt + JSON_ONLY_SUFFIX}],
                temperature=temperature,
                extra_body=self.extra_body,
                max_tokens=self.max_tokens,
            )
        self._resolved_version = getattr(resp, "model", None) or self.model
        content = resp.choices[0].message.content
        finish_reason = resp.choices[0].finish_reason
        reasoning_content = getattr(resp.choices[0].message, "reasoning_content", None)
        self.last_finish_reason = finish_reason
        self.last_content_chars = len(content or "")
        self.last_reasoning_chars = len(reasoning_content) if reasoning_content else None
        print(
            f"    [{label}] responded in {time.monotonic() - t0:.1f}s "
            f"(finish_reason={finish_reason}, {len(content or '')} content chars)",
            flush=True,
        )
        if finish_reason == "length":
            print(f"    WARNING: response truncated at max_tokens={self.max_tokens}.", flush=True)
        return _parse_content(content, label, finish_reason)

    def provenance(self):
        return {"backend": self.backend_label, "model": self.model, "resolved_version": self._resolved_version}


class AsyncOpenAIClient(LLMClient):
    """Asynchronous client with retries; optionally uses structured output via a pydantic model."""

    def __init__(
        self,
        api_key,
        model,
        base_url=None,
        backend_label="openai",
        timeout=120,
        max_tokens=4096,
        extra_body=None,
        max_retries=6,
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        self.model = model
        self.backend_label = backend_label
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        self._resolved_version = model
        self.max_retries = max_retries

    async def acomplete_json(self, prompt, temperature, response_model=None):
        retrying = AsyncRetrying(
            wait=wait_exponential_jitter(initial=1, max=60),
            stop=stop_after_attempt(self.max_retries),
            retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                return await self._call_once(prompt, temperature, response_model)

    async def _call_once(self, prompt, temperature, response_model=None):
        label = f"{self.backend_label}/{self.model}"
        t0 = time.monotonic()
        print(
            f"    [{label}] calling async (timeout={self.client.timeout}s, max_tokens={self.max_tokens})...", flush=True
        )

        parsed_obj = None
        resp = None
        if response_model is not None:
            try:
                resp = await self.client.chat.completions.parse(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                    response_format=response_model,
                    extra_body=self.extra_body,
                )
                if resp.choices[0].message.parsed is not None:
                    parsed_obj = resp.choices[0].message.parsed.model_dump()
            except Exception as e:
                print(f"    [{label}] structured output failed ({type(e).__name__}: {e}), using json_object mode...")
                resp = None

        if parsed_obj is None:
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    extra_body=self.extra_body,
                )
            except Exception as e:
                print(f"    [{label}] json_object mode failed ({type(e).__name__}: {e}), retrying without...")
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt + JSON_ONLY_SUFFIX}],
                    temperature=temperature,
                    extra_body=self.extra_body,
                    max_tokens=self.max_tokens,
                )

        self._resolved_version = getattr(resp, "model", None) or self.model
        content = resp.choices[0].message.content
        finish_reason = resp.choices[0].finish_reason
        print(
            f"    [{label}] responded in {time.monotonic() - t0:.1f}s "
            f"(finish_reason={finish_reason}, {len(content or '')} content chars)",
            flush=True,
        )
        if finish_reason == "length":
            print(f"    WARNING: response truncated at max_tokens={self.max_tokens}.", flush=True)
        if parsed_obj is not None:
            return parsed_obj
        return _parse_content(content, label, finish_reason)

    def provenance(self):
        return {"backend": self.backend_label, "model": self.model, "resolved_version": self._resolved_version}


def _credentials(backend):
    if backend == "openai":
        if not API_KEY:
            raise RuntimeError("--backend openai needs OPENAI_API_KEY (see .env.example)")
        return API_KEY, None
    if backend == "private":
        if not (PRIVATE_API_KEY and PRIVATE_ENDPOINT):
            raise RuntimeError("--backend private needs PRIVATE_API_KEY and PRIVATE_ENDPOINT (see .env.example)")
        return PRIVATE_API_KEY, PRIVATE_ENDPOINT
    raise ValueError(f"unknown backend: {backend}")


def build_llm_client(backend, model, timeout=120, max_tokens=4096, extra_body=None):
    api_key, base_url = _credentials(backend)
    return OpenAIClient(
        api_key,
        model,
        base_url=base_url,
        backend_label=backend,
        timeout=timeout,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )


def build_async_llm_client(backend, model, timeout=120, max_tokens=4096, extra_body=None, max_retries=6):
    api_key, base_url = _credentials(backend)
    return AsyncOpenAIClient(
        api_key,
        model,
        base_url=base_url,
        backend_label=backend,
        timeout=timeout,
        max_tokens=max_tokens,
        extra_body=extra_body,
        max_retries=max_retries,
    )
