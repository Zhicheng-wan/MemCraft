"""
brain.py - LLM interface for the MemCraft agent.
Calls the UCSD TritonAI API (Llama-4-Scout or similar).
"""

import json
import time
import requests
from typing import Optional


class Brain:
    """LLM interface that handles API calls and token tracking."""

    def __init__(self, api_key: str, api_url: str, model: str,
                 max_tokens: int = 4096, temperature: float = 0.3):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Token tracking for budget management
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.total_latency = 0.0

    def query(self, system_prompt: str, user_prompt: str,
              json_mode: bool = False) -> dict:
        """
        Send a query to the LLM and return the parsed response.

        Returns:
            dict with 'content' (str), 'tokens_used' (dict), 'latency' (float)
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        start = time.time()
        try:
            response = requests.post(self.api_url, headers=headers,
                                     json=payload, timeout=30)
            latency = time.time() - start

            if response.status_code != 200:
                return {
                    "content": "",
                    "error": f"API error {response.status_code}: {response.text}",
                    "tokens_used": {"prompt": 0, "completion": 0},
                    "latency": latency
                }

            data = response.json()
            content = data["choices"][0]["message"]["content"] or ""

            # Track tokens
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_calls += 1
            self.total_latency += latency

            return {
                "content": content,
                "error": None,
                "tokens_used": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens
                },
                "latency": latency
            }

        except requests.exceptions.Timeout:
            return {
                "content": "",
                "error": "Request timed out (30s)",
                "tokens_used": {"prompt": 0, "completion": 0},
                "latency": time.time() - start
            }
        except Exception as e:
            return {
                "content": "",
                "error": str(e),
                "tokens_used": {"prompt": 0, "completion": 0},
                "latency": time.time() - start
            }

    def parse_json_response(self, content: str) -> Optional[dict]:
        """Try to parse JSON from LLM response, handling markdown fences and thinking blocks."""
        if not content:
            return None
        content = content.strip()

        # Strip <think>...</think> reasoning blocks (Llama-4 / reasoning models)
        import re
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        # Strip markdown code fences
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines).strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Find the outermost balanced JSON object or array
            for open_char, close_char in [('{', '}'), ('[', ']')]:
                start = content.find(open_char)
                if start < 0:
                    continue
                depth = 0
                in_string = False
                escape_next = False
                for i, ch in enumerate(content[start:], start):
                    if escape_next:
                        escape_next = False
                        continue
                    if ch == '\\' and in_string:
                        escape_next = True
                        continue
                    if ch == '"':
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if ch == open_char:
                        depth += 1
                    elif ch == close_char:
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(content[start:i + 1])
                            except json.JSONDecodeError:
                                break
            return None

    def get_stats(self) -> dict:
        """Return usage statistics."""
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "avg_latency": (self.total_latency / self.total_calls
                           if self.total_calls > 0 else 0),
        }