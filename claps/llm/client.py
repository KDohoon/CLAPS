"""
LLM client for the CLAPS cognitive-symptom generation pipeline.

All CLAPS LLM components (Descriptor, Initializer, Updater, Appraiser, Generator)
are GPT-4o mini instances called through a single OpenAI-compatible chat-completions
endpoint. The endpoint, API key, and decoding parameters are read from environment
variables so that no secret is ever committed to the repository.

Environment variables
----------------------
CLAPS_LLM_API_URL   chat-completions endpoint (required)
CLAPS_LLM_API_KEY   API key (required)
CLAPS_LLM_AUTH      "apikey" (default) sends the key in an `apikey` header;
                    "bearer" sends `Authorization: Bearer <key>`.

The defaults reproduce the setup used in the paper (GPT-4o mini served through an
OpenAI-compatible proxy). To call the public OpenAI API instead, set:

    CLAPS_LLM_API_URL=https://api.openai.com/v1/chat/completions
    CLAPS_LLM_API_KEY=sk-...
    CLAPS_LLM_AUTH=bearer
"""

import os
import re
import json
import time
from urllib import request, error


class LLMClient:
    def __init__(self, api_url=None, api_key=None, auth=None,
                 temperature=0.2, max_tokens=4096, retry=3,
                 sleep_between_calls=1.0, timeout=120):
        self.api_url = api_url or os.getenv("CLAPS_LLM_API_URL")
        self.api_key = api_key or os.getenv("CLAPS_LLM_API_KEY")
        self.auth = (auth or os.getenv("CLAPS_LLM_AUTH", "apikey")).lower()
        if not self.api_url or not self.api_key:
            raise ValueError(
                "Set CLAPS_LLM_API_URL and CLAPS_LLM_API_KEY (see .env.example)."
            )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry = retry
        self.sleep_between_calls = sleep_between_calls
        self.timeout = timeout

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.auth == "bearer":
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            headers["apikey"] = self.api_key
        return headers

    def chat(self, system_prompt, user_prompt, label=""):
        """Single chat completion. Returns the assistant text, or None on failure."""
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        for attempt in range(self.retry):
            try:
                req = request.Request(self.api_url, data=data,
                                      headers=self._headers(), method="POST")
                with request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                if "choices" not in body or not body["choices"]:
                    raise ValueError(f"Unexpected response: {body}")
                content = body["choices"][0]["message"]["content"]
                if isinstance(content, list):  # some proxies return a list of parts
                    content = "".join(p.get("text", "") for p in content
                                      if isinstance(p, dict))
                return content
            except error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")
                print(f"    [{label}] attempt {attempt + 1} HTTP {e.code}: {err_body[:200]}")
            except Exception as e:
                print(f"    [{label}] attempt {attempt + 1} failed: {e}")
            if attempt < self.retry - 1:
                time.sleep(5 * (attempt + 1))
        return None


def extract_json(text):
    """Best-effort extraction of a single JSON object from an LLM response."""
    if not text:
        return None
    stripped = text.strip()
    candidates = []
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    for pattern in (r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            candidates.append(match.group(1).strip())
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last != -1 and first < last:
        candidates.append(text[first:last + 1].strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None
