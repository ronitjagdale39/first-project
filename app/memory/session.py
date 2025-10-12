from collections import defaultdict
from typing import Dict, List

Message = Dict[str, str]

class SessionMemory:
    def __init__(self, max_messages: int = 15) -> None:
        self._store: Dict[str, List[Message]] = defaultdict(list)
        self._max_messages = max_messages

    def get_history(self, session_id: str) -> List[Message]:
        history = self._store.get(session_id, [])
        if len(history) > self._max_messages:
            return history[-self._max_messages :]
        return history

    def append(self, session_id: str, user_message: str, assistant_reply: str) -> None:
        self._store[session_id].append({"role": "user", "content": user_message})
        self._store[session_id].append({"role": "assistant", "content": assistant_reply})
        if len(self._store[session_id]) > self._max_messages:
            self._store[session_id] = self._store[session_id][-self._max_messages :]
