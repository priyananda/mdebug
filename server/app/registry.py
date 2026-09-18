"""In-process session registry.

Sessions are server-side state, which is a real constraint on a scale-to-zero
platform: an instance reclaimed between requests takes its sessions with it.
The client has a `session_not_found` recovery path for exactly that, and the
deployment should set min-instances to 1.
"""

from __future__ import annotations

import asyncio
import os

from .runtime import Runtime, get_runtime
from .session import Session, SessionConfig

MAX_SESSIONS = int(os.environ.get("MDEBUG_MAX_SESSIONS", 16))


class SessionNotFound(KeyError):
    def __init__(self, session_id: str) -> None:
        super().__init__(session_id)
        self.session_id = session_id


class Registry:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    @property
    def runtime(self) -> Runtime:
        return get_runtime()

    async def create(self, config: SessionConfig) -> Session:
        async with self._lock:
            # Each session holds a KV cache and captured attention, so the count
            # is bounded rather than left to grow until the process dies.
            while len(self._sessions) >= MAX_SESSIONS:
                oldest = next(iter(self._sessions))
                await self._close(oldest)
            session = Session(self.runtime, config)
            self._sessions[session.id] = session
            return session

    def get(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFound(session_id)
        return session

    def list(self) -> list[Session]:
        return list(self._sessions.values())

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            if session_id not in self._sessions:
                raise SessionNotFound(session_id)
            await self._close(session_id)

    async def _close(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            await session.close()

    async def shutdown(self) -> None:
        for session_id in list(self._sessions):
            await self._close(session_id)


registry = Registry()
