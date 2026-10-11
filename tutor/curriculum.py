"""Keyword retrieval over the teacher's syllabus and course material."""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it me my of on or so that the this "
    "to what when why with you your".split()
)


@dataclass(frozen=True)
class Chunk:
    title: str
    text: str


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1}


def chunk_text(title: str, text: str, max_chars: int = 900) -> list[Chunk]:
    """Split on blank lines, then pack paragraphs into chunks of roughly max_chars."""
    chunks: list[Chunk] = []
    current = ""
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(Chunk(title, current))
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(Chunk(title, current))
    return chunks


def retrieve(question: str, materials: list[tuple[str, str]], top_k: int = 4) -> list[Chunk]:
    """Return the chunks that share the most distinctive words with the question."""
    query = _tokens(question)
    if not query:
        return []
    scored: list[tuple[int, int, Chunk]] = []
    order = 0
    for title, text in materials:
        for chunk in chunk_text(title, text):
            overlap = len(query & _tokens(chunk.text + " " + title))
            if overlap:
                scored.append((overlap, -order, chunk))
            order += 1
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [chunk for _, _, chunk in scored[:top_k]]
