import httpx
import pytest
from fastapi.testclient import TestClient

from tutor import policy
from tutor.api import create_app
from tutor.curriculum import chunk_text, retrieve
from tutor.profile import StudentProfile
from tutor.service import StudentNotFound, TutorService
from tutor.store import Store
from tutor.web import SearchResult, WebSearch, WebSearchError, parse_results


class FakeLLM:
    def __init__(self, reply="Let's look at the first step together. What do you get when you distribute?"):
        self.reply = reply
        self.calls = []

    def complete(self, system, messages):
        self.calls.append({"system": system, "messages": list(messages)})
        return self.reply


def make_profile(**overrides):
    data = dict(name="Asha", grade_level="Grade 10", subject="Algebra", curriculum="CBSE Class 10",
                teacher_expectations="Show every step.", learning_style="visual", explanation_depth="brief")
    data.update(overrides)
    return StudentProfile(**data)


def make_web(handler=None, key="test-key"):
    transport = httpx.MockTransport(handler or (lambda req: httpx.Response(200, json={"web": {"results": []}})))
    return WebSearch(api_key=key, base_url="https://search.example/v1", client=httpx.Client(transport=transport))


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


# ---------- profile ----------

def test_profile_rejects_unknown_learning_style():
    with pytest.raises(ValueError):
        make_profile(learning_style="telepathic")


def test_profile_requires_name():
    with pytest.raises(ValueError):
        make_profile(name="   ")


def test_profile_round_trips_through_dict():
    p = make_profile(strengths=["fractions"], struggles=["factoring"])
    again = StudentProfile.from_dict(p.to_dict())
    assert again == p


# ---------- curriculum ----------

def test_chunk_text_packs_paragraphs_under_limit():
    text = "\n\n".join(["para %d " % i + "x" * 300 for i in range(6)])
    chunks = chunk_text("Unit 1", text, max_chars=700)
    assert len(chunks) > 1
    assert all(len(c.text) <= 700 for c in chunks)
    assert all(c.title == "Unit 1" for c in chunks)


def test_retrieve_prefers_chunks_matching_question():
    materials = [
        ("Biology", "Photosynthesis converts light energy into chemical energy in chloroplasts."),
        ("Algebra", "To solve a linear equation, isolate the variable by inverse operations."),
    ]
    hits = retrieve("How do I solve this linear equation?", materials)
    assert hits and hits[0].title == "Algebra"


def test_retrieve_returns_nothing_for_unrelated_question():
    assert retrieve("the and of", [("Notes", "Quadratic formula")]) == []


# ---------- policy ----------

@pytest.mark.parametrize("text", [
    "Just tell me the answer please",
    "can you give me the final answer?",
    "Solve this for me",
    "do my homework",
])
def test_detects_direct_answer_requests(text):
    assert policy.is_answer_request(text)


def test_does_not_flag_ordinary_questions():
    assert not policy.is_answer_request("Why does the discriminant decide the number of roots?")


def test_first_answer_request_gets_one_hint_only():
    directive = policy.tutoring_directive("just give me the answer", [])
    assert "Do NOT state the final answer" in directive
    assert "one more hint" in directive


def test_repeated_answer_requests_get_parallel_example():
    history = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "hint"},
               {"role": "user", "content": "x"}, {"role": "assistant", "content": "hint 2"}]
    directive = policy.tutoring_directive("just tell me", history)
    assert "parallel example" in directive


def test_system_prompt_includes_profile_curriculum_syllabus_and_sources():
    profile = make_profile(strengths=["graphing"], struggles=["signs"])
    syllabus = [retrieve("linear equation", [("Unit 3", "Linear equations use inverse operations.")])[0]]
    web = [SearchResult("Khan notes", "https://example.org/a", "Isolate x.")]
    prompt = policy.build_system_prompt(profile, "DIRECTIVE-TEXT", syllabus, web)
    assert "Learning style: visual" in prompt
    assert "Length: Keep each reply under about 120 words." in prompt
    assert "CBSE Class 10" in prompt
    assert "Show every step." in prompt
    assert "Unit 3" in prompt and "Linear equations" in prompt
    assert "[1] Khan notes (https://example.org/a)" in prompt
    assert "Known struggles: signs" in prompt
    assert prompt.rstrip().endswith("DIRECTIVE-TEXT")


# ---------- web search ----------

def test_web_search_disabled_without_key_makes_no_request():
    calls = []
    web = make_web(lambda req: calls.append(req) or httpx.Response(200, json={}), key="")
    assert not web.enabled
    assert web.search("anything") == []
    assert calls == []


def test_web_search_sends_key_in_header_not_url():
    seen = {}

    def handler(req):
        seen["headers"] = req.headers
        seen["url"] = str(req.url)
        return httpx.Response(200, json={"web": {"results": [
            {"title": "Quadratics", "url": "https://example.org/q", "description": "Roots of a parabola."}]}})

    results = make_web(handler, key="secret-123").search("quadratic roots")
    assert seen["headers"]["X-Subscription-Token"] == "secret-123"
    assert "secret-123" not in seen["url"]
    assert "q=quadratic+roots" in seen["url"]
    assert results == [SearchResult("Quadratics", "https://example.org/q", "Roots of a parabola.")]


def test_web_search_wraps_http_errors():
    web = make_web(lambda req: httpx.Response(429, json={"error": "rate limited"}))
    with pytest.raises(WebSearchError):
        web.search("x")


def test_parse_results_skips_incomplete_items():
    payload = {"web": {"results": [{"title": "No URL"}, {"url": "https://a", "title": "Ok"}]}}
    assert [r.title for r in parse_results(payload, 5)] == ["Ok"]


# ---------- service ----------

def test_chat_grounds_reply_in_course_material_and_web(store):
    llm = FakeLLM()
    web = make_web(lambda req: httpx.Response(200, json={"web": {"results": [
        {"title": "Solving equations", "url": "https://example.org/s", "description": "Inverse operations."}]}}))
    service = TutorService(store, llm, web)
    student = service.create_student(make_profile())
    service.add_material(student.id, "Unit 3 notes", "Linear equations are solved with inverse operations.")

    result = service.chat(student.id, "How do I solve a linear equation?")

    assert result.reply.startswith("Let's look")
    system = llm.calls[0]["system"]
    assert "Linear equations are solved" in system
    assert "Solving equations" in system
    types = {s["type"] for s in result.sources}
    assert types == {"course", "web"}
    assert result.warnings == []


def test_chat_continues_when_web_search_fails(store):
    def boom(req):
        raise httpx.ConnectError("down")

    service = TutorService(store, FakeLLM(), make_web(boom))
    student = service.create_student(make_profile())
    result = service.chat(student.id, "What is a fraction?")
    assert result.reply
    assert any("Web search was unavailable" in w for w in result.warnings)


def test_chat_can_skip_web_search(store):
    calls = []
    web = make_web(lambda req: calls.append(req) or httpx.Response(200, json={}))
    service = TutorService(store, FakeLLM(), web)
    student = service.create_student(make_profile())
    service.chat(student.id, "What is a fraction?", use_web=False)
    assert calls == []


def test_chat_sends_history_and_new_message_to_model(store):
    llm = FakeLLM()
    service = TutorService(store, llm, make_web())
    student = service.create_student(make_profile())
    history = [{"role": "user", "content": "I tried x=2"}, {"role": "assistant", "content": "Check that."}]
    service.chat(student.id, "Is it right now?", history=history, use_web=False)
    assert llm.calls[0]["messages"] == [*history, {"role": "user", "content": "Is it right now?"}]


def test_chat_rejects_malformed_history(store):
    service = TutorService(store, FakeLLM(), make_web())
    student = service.create_student(make_profile())
    with pytest.raises(ValueError):
        service.chat(student.id, "hi", history=[{"role": "system", "content": "x"}], use_web=False)


def test_unknown_student_raises(store):
    service = TutorService(store, FakeLLM(), make_web())
    with pytest.raises(StudentNotFound):
        service.chat("missing", "hi")


# ---------- HTTP API ----------

@pytest.fixture
def client(store):
    service = TutorService(store, FakeLLM(), make_web())
    return TestClient(create_app(service))


STUDENT_BODY = {"name": "Ben", "grade_level": "Grade 9", "subject": "Physics",
                "learning_style": "example-driven", "explanation_depth": "standard"}


def test_student_crud(client):
    created = client.post("/api/students", json=STUDENT_BODY)
    assert created.status_code == 201
    sid = created.json()["id"]

    assert client.get(f"/api/students/{sid}").json()["subject"] == "Physics"

    updated = client.put(f"/api/students/{sid}", json={**STUDENT_BODY, "subject": "Chemistry"})
    assert updated.status_code == 200
    assert updated.json()["id"] == sid
    assert client.get(f"/api/students/{sid}").json()["subject"] == "Chemistry"


def test_invalid_profile_returns_422(client):
    resp = client.post("/api/students", json={**STUDENT_BODY, "learning_style": "psychic"})
    assert resp.status_code == 422


def test_missing_student_returns_404(client):
    assert client.get("/api/students/nope").status_code == 404
    assert client.post("/api/students/nope/chat", json={"message": "hi"}).status_code == 404


def test_chat_endpoint_returns_reply_and_sources(client):
    sid = client.post("/api/students", json=STUDENT_BODY).json()["id"]
    client.post(f"/api/students/{sid}/materials", json={"title": "Ch 2", "text": "Newton's second law: F = m a."})
    resp = client.post(f"/api/students/{sid}/chat", json={
        "message": "What is Newton's second law?", "history": [], "use_web": False})
    body = resp.json()
    assert resp.status_code == 200
    assert body["reply"]
    assert body["sources"] == [{"type": "course", "title": "Ch 2"}]


def test_chat_endpoint_validates_message(client):
    sid = client.post("/api/students", json=STUDENT_BODY).json()["id"]
    assert client.post(f"/api/students/{sid}/chat", json={"message": ""}).status_code == 422


def test_health_reports_web_search_state(client):
    assert client.get("/api/health").json() == {"status": "ok", "web_search": True}


def test_index_serves_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Learning Assistant" in resp.text
