"""Orchestrates one tutoring turn: retrieve course material, search the web, call the model."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from . import policy
from .curriculum import retrieve
from .llm import LLM
from .profile import StudentProfile
from .store import Store
from .web import SearchResult, WebSearch, WebSearchError

log = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 20


class StudentNotFound(KeyError):
    pass


@dataclass
class ChatResult:
    reply: str
    sources: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class TutorService:
    def __init__(self, store: Store, llm: LLM, web: WebSearch | None = None) -> None:
        self.store = store
        self.llm = llm
        self.web = web or WebSearch()

    def create_student(self, profile: StudentProfile) -> StudentProfile:
        self.store.save_student(profile)
        return profile

    def update_student(self, student_id: str, profile: StudentProfile) -> StudentProfile:
        self._require(student_id)
        profile.id = student_id
        self.store.save_student(profile)
        return profile

    def get_student(self, student_id: str) -> StudentProfile:
        return self._require(student_id)

    def add_material(self, student_id: str, title: str, text: str) -> None:
        self._require(student_id)
        if not text.strip():
            raise ValueError("material text is empty")
        self.store.add_material(student_id, title, text)

    def chat(
        self,
        student_id: str,
        message: str,
        history: Sequence[dict] = (),
        use_web: bool = True,
    ) -> ChatResult:
        profile = self._require(student_id)
        message = message.strip()
        if not message:
            raise ValueError("message is empty")

        warnings: list[str] = []
        history = list(history)[-MAX_HISTORY_TURNS:]
        self._validate_history(history)

        syllabus = retrieve(message, self.store.list_materials(student_id))

        web_results: list[SearchResult] = []
        if use_web and self.web.enabled:
            try:
                web_results = self.web.search(message)
            except WebSearchError as exc:
                log.warning("web search unavailable: %s", exc)
                warnings.append("Web search was unavailable, so this answer uses only your course material and general knowledge.")

        directive = policy.tutoring_directive(message, history)
        system = policy.build_system_prompt(profile, directive, syllabus, web_results)
        messages = [*history, {"role": "user", "content": message}]
        reply = self.llm.complete(system, messages)

        sources = [{"type": "course", "title": c.title} for c in _unique_by_title(syllabus)]
        sources += [{"type": "web", "title": r.title, "url": r.url} for r in web_results]
        return ChatResult(reply=reply, sources=sources, warnings=warnings)

    def _require(self, student_id: str) -> StudentProfile:
        profile = self.store.get_student(student_id)
        if profile is None:
            raise StudentNotFound(student_id)
        return profile

    @staticmethod
    def _validate_history(history: Sequence[dict]) -> None:
        for turn in history:
            if turn.get("role") not in ("user", "assistant") or not isinstance(turn.get("content"), str):
                raise ValueError("history turns need role 'user' or 'assistant' and string content")


def _unique_by_title(chunks):
    seen, out = set(), []
    for chunk in chunks:
        if chunk.title not in seen:
            seen.add(chunk.title)
            out.append(chunk)
    return out
