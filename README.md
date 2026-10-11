# chatbot — Student Learning Assistant

An AI tutor that adapts to each student and guides them toward answers instead of handing them over.

## What it does

- **Understands each student.** Each student has a profile: grade level, subject, curriculum, teacher
  expectations, learning style (step-by-step, example-driven, visual, or verbal), explanation length,
  strengths, and known struggles. The profile shapes every reply.
- **Teaches, doesn't just answer.** The tutor asks what the student has tried and gives one hint at a time.
  If a student keeps asking for the answer, it switches to a worked example on different numbers and asks
  them to apply the same steps. It never states the final answer to the student's own problem.
- **Aligns with the curriculum.** Teachers' notes and course material are uploaded per student. The tutor
  retrieves the most relevant passages for each question and prefers them over general knowledge. It also
  follows the curriculum terminology and any method the teacher requires.
- **Pulls current information from the web.** Web search uses an API key from the environment. Results are
  listed as sources under each reply. If search is unavailable, the tutor says so and carries on.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in the keys
export $(grep -v '^#' .env | xargs)
uvicorn tutor.api:create_app --factory --reload
```

Open http://127.0.0.1:8000, set up a student profile, and start asking questions.

### Keys

| Variable | Purpose |
| --- | --- |
| `ANTHROPIC_API_KEY` | Required. Used for tutoring replies. |
| `WEB_SEARCH_API_KEY` | Optional. Enables web lookups (Brave Search by default). |
| `WEB_SEARCH_URL` | Optional. Alternate search endpoint with the same query parameters. |
| `TUTOR_MODEL` | Optional. Defaults to `claude-sonnet-5-5`. |
| `DATABASE_PATH` | Optional. SQLite file path, defaults to `tutor.db`. |

Keys are read only from the environment. They are never written to the database, logs, or responses.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/students` | Create a student profile |
| `GET` / `PUT` | `/api/students/{id}` | Read or update a profile |
| `POST` | `/api/students/{id}/materials` | Add teacher notes or syllabus text (`title`, `text`) |
| `POST` | `/api/students/{id}/chat` | Ask a question. Body: `message`, `history`, `use_web`. Returns `reply`, `sources`, `warnings`. |
| `GET` | `/api/health` | Status, and whether web search is configured |

The client keeps the conversation and sends it back as `history` on each turn. The hint count used for the
"no answer" policy is taken from that history.

## Layout

```
tutor/
  profile.py     student profile model
  store.py       SQLite storage for profiles and course material
  curriculum.py  chunking and keyword retrieval over course material
  web.py         web search client (reads WEB_SEARCH_API_KEY)
  policy.py      tutoring rules and system prompt construction
  llm.py         Anthropic Messages API wrapper
  service.py     one chat turn: retrieve, search, prompt, reply
  api.py         FastAPI routes
  static/        single-page chat UI
tests/           offline tests with a fake model and mocked search
```

## Tests

```bash
python -m pytest -q
```

The tests use a fake model and a mocked HTTP transport, so they need no network access or API keys.

## Limitations

- Retrieval is keyword-based. It works well for focused teacher notes but is not semantic search.
- Hint escalation counts assistant turns in the conversation, not per problem. A long session on several
  problems will escalate sooner than one problem alone would.
- Authentication is not included. Anyone who can reach the server can read profiles. Add auth before
  deploying it for real students.
