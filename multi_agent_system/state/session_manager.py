# multi-agent-system/state/session_manager.py
import uuid
import time
from typing import Dict, Any

class InMemorySessionService:
    """
    Simple in-memory sessions for short-lived interactions.
    """
    def __init__(self):
        self.sessions = {}

    def create_session(self) -> str:
        sid = str(uuid.uuid4())
        self.sessions[sid] = {"created_at": time.time(), "data": {}}
        return sid

    def get(self, sid: str):
        return self.sessions.get(sid)

    def update(self, sid: str, key: str, value: Any):
        if sid not in self.sessions:
            raise KeyError("session not found")
        self.sessions[sid]["data"][key] = value

    def delete(self, sid: str):
        if sid in self.sessions:
            del self.sessions[sid]
