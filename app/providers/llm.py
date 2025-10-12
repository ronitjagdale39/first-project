import os
from typing import List, Dict, Optional, AsyncGenerator

import aiohttp

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

class LLMProvider:
    async def complete(
        self,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = 512,
    ) -> str:
        # Fallback to echo if key missing
        if not OPENAI_API_KEY:
            # minimal echo/mock for local dev
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            return f"[mock] {last_user[:200]}"

        payload = {
            "model": OPENAI_MODEL,
            "messages": [{"role": "system", "content": system}] + messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{OPENAI_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=60) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"OpenAI error {resp.status}: {text}")
                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
                return content.strip()

    async def stream_complete(
        self,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = 512,
    ) -> AsyncGenerator[str, None]:
        """Yield tokens as they arrive. Falls back to single-chunk mock."""
        # Fallback streaming: just echo the last user message in one chunk
        if not OPENAI_API_KEY:
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            yield f"[mock] {last_user[:200]}"
            return

        payload = {
            "model": OPENAI_MODEL,
            "messages": [{"role": "system", "content": system}] + messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{OPENAI_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=None) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"OpenAI error {resp.status}: {text}")

                async for raw in resp.content:
                    try:
                        line = raw.decode("utf-8").strip()
                    except Exception:
                        continue
                    if not line:
                        continue
                    # OpenAI streams Server-Sent Events lines prefixed with 'data: '
                    # Each event may contain a JSON with delta tokens or [DONE]
                    if line.startswith("data: "):
                        data_str = line[len("data: "):].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            import json

                            data_obj = json.loads(data_str)
                            delta = data_obj["choices"][0]["delta"].get("content")
                            if delta:
                                yield delta
                        except Exception:
                            # Ignore malformed lines
                            continue
