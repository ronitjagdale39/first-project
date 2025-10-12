from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from ..providers.llm import LLMProvider
from ..skills.calculator import try_calculate
from ..memory.session import SessionMemory

router = APIRouter()
llm = LLMProvider()
memory = SessionMemory()

class ChatRequest(BaseModel):
    session_id: str
    message: str
    temperature: Optional[float] = 0.3

class ChatResponse(BaseModel):
    reply: str

@router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    # retrieve context
    history = memory.get_history(req.session_id)

    # simple intent route: calculator first
    calc = try_calculate(req.message)
    if calc is not None:
        reply = str(calc)
        memory.append(req.session_id, req.message, reply)
        return ChatResponse(reply=reply)

    # otherwise call LLM
    system_prompt = (
        "You are a concise, helpful assistant. If the user asks to calculate, respond with the result only."
    )
    reply = await llm.complete(
        system=system_prompt,
        messages=history + [{"role": "user", "content": req.message}],
        temperature=req.temperature or 0.3,
    )

    memory.append(req.session_id, req.message, reply)
    return ChatResponse(reply=reply)


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    history = memory.get_history(req.session_id)

    calc = try_calculate(req.message)
    if calc is not None:
        reply = str(calc)
        # Store immediately and stream a single chunk
        memory.append(req.session_id, req.message, reply)

        async def gen_calc():
            yield reply

        return StreamingResponse(gen_calc(), media_type="text/plain")

    system_prompt = (
        "You are a concise, helpful assistant. If the user asks to calculate, respond with the result only."
    )

    async def token_generator():
        full_chunks = []
        async for token in llm.stream_complete(
            system=system_prompt,
            messages=history + [{"role": "user", "content": req.message}],
            temperature=req.temperature or 0.3,
        ):
            full_chunks.append(token)
            yield token
        # After streaming completes, store in memory
        memory.append(req.session_id, req.message, "".join(full_chunks))

    return StreamingResponse(token_generator(), media_type="text/plain")
