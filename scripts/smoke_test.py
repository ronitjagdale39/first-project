import os
import sys
import asyncio

# Ensure we use mock fallback (no external calls)
os.environ.pop("OPENAI_API_KEY", None)

# Add project root to sys.path for module imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.main import root
from app.router.chat import chat, ChatRequest


def test_root():
    data = root()
    assert data["status"] == "ok"
    print("root:", data)


async def test_chat_calc():
    resp = await chat(ChatRequest(session_id="s1", message="2+3*4"))
    print("calc:", resp.reply)
    assert resp.reply == str(2 + 3 * 4)


async def test_chat_llm():
    resp = await chat(ChatRequest(session_id="s1", message="Hello there"))
    print("llm:", resp.reply)
    assert resp.reply.startswith("[mock]")


async def main():
    test_root()
    await test_chat_calc()
    await test_chat_llm()
    print("SMOKE OK")


if __name__ == "__main__":
    asyncio.run(main())
