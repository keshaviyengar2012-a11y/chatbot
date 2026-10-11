"""Per-student profile: who the student is and how they like to learn."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

LEARNING_STYLES = ("visual", "verbal", "example-driven", "step-by-step")
DEPTHS = ("brief", "standard", "detailed")


@dataclass
class StudentProfile:
    name: str
    grade_level: str
    subject: str
    curriculum: str = ""  # e.g. "CBSE Class 10", "AP Calculus BC", "IB Biology HL"
    teacher_expectations: str = ""  # free text: rubric notes, required methods, notation
    learning_style: str = "step-by-step"
    explanation_depth: str = "standard"
    strengths: list[str] = field(default_factory=list)
    struggles: list[str] = field(default_factory=list)
    language: str = "English"
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        for label, value in (("name", self.name), ("grade_level", self.grade_level), ("subject", self.subject)):
            if not str(value).strip():
                raise ValueError(f"{label} is required")
        if self.learning_style not in LEARNING_STYLES:
            raise ValueError(f"learning_style must be one of {LEARNING_STYLES}")
        if self.explanation_depth not in DEPTHS:
            raise ValueError(f"explanation_depth must be one of {DEPTHS}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StudentProfile":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)
