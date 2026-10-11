"""Tutoring policy: turns a student profile and retrieved context into the system prompt.

The core rule is that the tutor guides the student to the answer rather than handing it over.
How much help is given depends on how many hints the student has already had in this
conversation, which is tracked from the history the client sends back on each turn.
"""

from __future__ import annotations

import re
from typing import Sequence

from .curriculum import Chunk
from .profile import StudentProfile
from .web import SearchResult

ANSWER_REQUEST = re.compile(
    r"\b(just tell me|just give me|give me the (final )?(answer|solution)|what('s| is) the (final )?answer"
    r"|solve (it|this|that) for me|do (my|the) (homework|assignment|worksheet)|tell me the answer)\b",
    re.IGNORECASE,
)

# After this many hints, the tutor may show a worked example on a *different* problem.
HINTS_BEFORE_WORKED_EXAMPLE = 2

STYLE_GUIDE = {
    "visual": "Use diagrams described in words, tables, number lines, and spatial analogies.",
    "verbal": "Explain in clear prose, use analogies and plain-language definitions before symbols.",
    "example-driven": "Lead with a concrete example, then extract the general rule from it.",
    "step-by-step": "Break every explanation into numbered, one-action-per-step moves.",
}

DEPTH_GUIDE = {
    "brief": "Keep each reply under about 120 words.",
    "standard": "Keep each reply focused, usually under 250 words.",
    "detailed": "You may go into depth, but still ask the student to do the final step themselves.",
}


def is_answer_request(message: str) -> bool:
    return bool(ANSWER_REQUEST.search(message))


def count_hints_given(history: Sequence[dict]) -> int:
    return sum(1 for turn in history if turn.get("role") == "assistant")


def tutoring_directive(message: str, history: Sequence[dict]) -> str:
    """Return a per-turn instruction that pins down how much to reveal this turn."""
    if not is_answer_request(message):
        return (
            "Respond with the next helpful step or a probing question. If the student has shown work, "
            "check it and point to the first place it goes wrong."
        )
    hints = count_hints_given(history)
    if hints < HINTS_BEFORE_WORKED_EXAMPLE:
        return (
            "The student is asking for the answer directly. Do NOT state the final answer. Give exactly one "
            "more hint that narrows the next step, and ask them to try it."
        )
    return (
        "The student has already had hints and still asks for the answer. Do NOT state the answer to their "
        "problem. Work through a parallel example with different numbers or a different context, showing "
        "every step, then ask the student to apply the same steps to their own problem."
    )


def build_system_prompt(
    profile: StudentProfile,
    directive: str,
    syllabus: Sequence[Chunk] = (),
    web_results: Sequence[SearchResult] = (),
) -> str:
    sections = [
        "You are a patient, encouraging tutor for a student. Your goal is learning, not completing work for them.",
        "",
        "## Student",
        f"- Name: {profile.name}",
        f"- Grade level: {profile.grade_level}",
        f"- Subject: {profile.subject}",
        f"- Reply in: {profile.language}",
        f"- Learning style: {profile.learning_style}. {STYLE_GUIDE[profile.learning_style]}",
        f"- Length: {DEPTH_GUIDE[profile.explanation_depth]}",
    ]
    if profile.strengths:
        sections.append(f"- Strengths: {', '.join(profile.strengths)}. Build on these where relevant.")
    if profile.struggles:
        sections.append(f"- Known struggles: {', '.join(profile.struggles)}. Check these carefully.")

    sections += ["", "## Curriculum alignment"]
    if profile.curriculum:
        sections.append(f"- Curriculum: {profile.curriculum}. Use the terminology and level of this curriculum.")
    if profile.teacher_expectations:
        sections.append(f"- Teacher expectations: {profile.teacher_expectations}")
    if not profile.curriculum and not profile.teacher_expectations:
        sections.append("- No curriculum is specified; use standard terminology for the grade level.")

    if syllabus:
        sections += ["", "## Course material from the teacher (prefer this over general knowledge)"]
        for chunk in syllabus:
            sections.append(f"### {chunk.title}\n{chunk.text}")

    if web_results:
        sections += ["", "## Web sources (current information; cite by number when you use them)"]
        for i, result in enumerate(web_results, start=1):
            sections.append(f"[{i}] {result.title} ({result.url})\n{result.snippet}")

    sections += [
        "",
        "## Rules",
        "- Never produce a finished answer to homework, a quiz, or a test question the student is working on.",
        "- Ask what the student has tried before explaining, when that is unclear.",
        "- Praise specific effort, and correct mistakes gently and precisely.",
        "- If you are unsure of a fact, say so instead of guessing.",
        "- Use the notation and method the teacher expects when the material specifies one.",
        "",
        "## This turn",
        directive,
    ]
    return "\n".join(sections)
