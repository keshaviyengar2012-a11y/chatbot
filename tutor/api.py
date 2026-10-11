"""FastAPI application exposing student profiles, course material, and chat."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .llm import AnthropicLLM
from .profile import StudentProfile
from .service import StudentNotFound, TutorService
from .store import Store

STATIC_DIR = Path(__file__).parent / "static"


class StudentIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    grade_level: str = Field(min_length=1, max_length=50)
    subject: str = Field(min_length=1, max_length=100)
    curriculum: str = Field(default="", max_length=200)
    teacher_expectations: str = Field(default="", max_length=4000)
    learning_style: str = "step-by-step"
    explanation_depth: str = "standard"
    strengths: list[str] = Field(default_factory=list, max_length=20)
    struggles: list[str] = Field(default_factory=list, max_length=20)
    language: str = Field(default="English", max_length=50)


class MaterialIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)


class TurnIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=10_000)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[TurnIn] = Field(default_factory=list, max_length=40)
    use_web: bool = True


def _to_profile(body: StudentIn) -> StudentProfile:
    try:
        return StudentProfile(**body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def create_app(service: TutorService | None = None) -> FastAPI:
    if service is None:
        store = Store(os.environ.get("DATABASE_PATH", "tutor.db"))
        service = TutorService(store, AnthropicLLM())

    app = FastAPI(title="Student Learning Assistant")
    app.state.service = service

    def _not_found(_student_id: str) -> HTTPException:
        return HTTPException(status_code=404, detail="student not found")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "web_search": service.web.enabled}

    @app.post("/api/students", status_code=201)
    def create_student(body: StudentIn) -> dict:
        return service.create_student(_to_profile(body)).to_dict()

    @app.get("/api/students/{student_id}")
    def get_student(student_id: str) -> dict:
        try:
            return service.get_student(student_id).to_dict()
        except StudentNotFound as exc:
            raise _not_found(student_id) from exc

    @app.put("/api/students/{student_id}")
    def update_student(student_id: str, body: StudentIn) -> dict:
        try:
            return service.update_student(student_id, _to_profile(body)).to_dict()
        except StudentNotFound as exc:
            raise _not_found(student_id) from exc

    @app.post("/api/students/{student_id}/materials", status_code=201)
    def add_material(student_id: str, body: MaterialIn) -> dict:
        try:
            service.add_material(student_id, body.title, body.text)
        except StudentNotFound as exc:
            raise _not_found(student_id) from exc
        return {"status": "stored", "title": body.title}

    @app.post("/api/students/{student_id}/chat")
    def chat(student_id: str, body: ChatIn) -> dict:
        try:
            result = service.chat(
                student_id,
                body.message,
                history=[t.model_dump() for t in body.history],
                use_web=body.use_web,
            )
        except StudentNotFound as exc:
            raise _not_found(student_id) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"reply": result.reply, "sources": result.sources, "warnings": result.warnings}

    return app
