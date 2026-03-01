"""
LLM client for TritonAI API (OpenAI-compatible endpoint).
Handles retries, JSON parsing, token tracking, and budget enforcement.
"""
import json
import time
import re
import requests
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    API_KEY, API_BASE_URL, MODEL_NAME, TokenTracker
)


class LLMClient:
    """Wrapper around TritonAI chat completions API."""
    
    def __init__(self, agent_name: str = "unknown", task_name: str = "unknown"):
        self.agent_name = agent_name
        self.task_name = task_name
        self.step_counter = 0
        
        if not API_KEY:
            raise ValueError(
                "TRITONAI_API_KEY not set. "
                "Export it or add to .env file."
            )
    
    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
        retries: int = 3,
    ) -> dict:
        """
        Send a chat completion request and parse JSON response.
        
        Returns:
            Parsed JSON dict from the LLM response.
            On failure, returns {"action": "noop", "reasoning": "LLM call failed"}.
        """
        # Budget check
        if not TokenTracker.check_budget():
            print("[BUDGET] Budget exhausted! Returning noop.")
            return {"action": "noop", "reasoning": "Budget exhausted"}
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        for attempt in range(retries):
            try:
                resp = requests.post(
                    API_BASE_URL, headers=headers, json=payload, timeout=60
                )
                
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    print(f"[LLM] Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                
                resp.raise_for_status()
                data = resp.json()
                
                # Track tokens
                usage = data.get("usage", {})
                input_tok = usage.get("prompt_tokens", 0)
                output_tok = usage.get("completion_tokens", 0)
                TokenTracker.log(
                    self.agent_name, self.task_name,
                    self.step_counter, input_tok, output_tok
                )
                
                # Extract content
                content = data["choices"][0]["message"]["content"]
                return self._parse_json(content)
                
            except requests.exceptions.Timeout:
                print(f"[LLM] Timeout on attempt {attempt+1}/{retries}")
                time.sleep(2 ** attempt)
            except requests.exceptions.RequestException as e:
                print(f"[LLM] Request error: {e}")
                time.sleep(2 ** attempt)
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                print(f"[LLM] Parse error: {e}")
                if attempt < retries - 1:
                    time.sleep(1)
        
        return {"action": "noop", "reasoning": "LLM call failed after retries"}
    
    def _parse_json(self, text: str) -> dict:
        """Extract JSON from LLM response, handling markdown fences."""
        # Strip markdown code fences
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        
        # Fallback: extract action from text
        return {
            "action": "noop",
            "reasoning": f"Could not parse JSON from: {text[:200]}"
        }
    
    def set_step(self, step: int):
        self.step_counter = step
