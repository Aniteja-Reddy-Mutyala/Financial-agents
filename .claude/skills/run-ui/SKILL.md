---
name: run-ui
description: Use when asked to start, launch, run, open, or preview this project's Streamlit web UI — the Compliance Agent (main page), the Research Agent page, or the Orchestrator Agent page. One dev server serves all three; launch it via the Claude Code browser preview tool's pre-configured compliance-ui entry, no manual setup or asking the user first.
---

## How to start it

Call the preview tool with `{"name": "compliance-ui"}` — this is already
configured in `.claude/launch.json` (`python3 -m streamlit run app_ui.py`,
port 8501) and opens `http://localhost:8501` in the Browser pane. Don't
ask the user whether to start it; just do it when this skill is invoked.

- Main page (`/`) = Compliance Agent (`app_ui.py`).
- `/Research_Agent` = the Research Agent page (`pages/1_Research_Agent.py`),
  auto-discovered by Streamlit's multipage nav — it's the same server,
  not a second launch.
- `/Orchestrator_Agent` = the top-level Orchestrator page
  (`pages/2_Orchestrator_Agent.py`), same auto-discovery, same server.
  It builds its own Compliance and Research agent instances, so opening
  it incurs at least the same real Bedrock retrieval cost as the main
  page, plus its own plan/compose LLM calls — see
  `edit-orchestrator-agent`.

If a server is already running on this port, the tool reuses it rather
than starting a duplicate — check `preview_list` first if unsure whether
one's already up.

## What to expect once it's open

- Both pages' "Use live Bedrock" toggle defaults to **off** (Groq), not
  Bedrock — see `CLAUDE.md`/git history for why.
- Groq needs `GROQ_API_KEY` in the environment the server was started
  with. **Nothing auto-loads `.env`** here — this is a deliberate,
  repeated project convention (see `CLAUDE.md`, `run-evaluation-suite`),
  not an oversight. If it's not already set in the shell this skill
  launches the server from, Groq calls raise `GroqUnavailable` and
  gracefully fall back to the templated draft rather than crash — that's
  expected behavior, not a bug, and not something to silently "fix."
- **Don't bake `.env`-sourcing into this skill's launch** (e.g. a wrapper
  script `.claude/launch.json` points at instead of running Streamlit
  directly) — this was tried before as a temporary, explicitly-reverted
  testing hack and was deliberately never committed, because it breaks
  the server for anyone without a `.env` file and contradicts the
  project's own documented "nothing auto-loads it" design. If the user
  wants a real Groq answer and the key isn't loaded, ask them whether to
  source `.env` for this session — same as any other time `.env` comes
  up — rather than wiring it in permanently.
- Retrieval for the Compliance Agent is always a real, billed Bedrock
  embedding call regardless of the toggle — there's no fully free path
  through that page. The Research Agent's web search (`ddgs`) is always
  free regardless of the toggle.
- `data/embeddings.npz` must already exist for the Compliance Agent to
  answer anything (it does in this repo) — see `bedrock-rag-retrieval`
  if it's ever missing.
- **Editing a file under `src/` while the server is already running
  does NOT take effect**, even though Streamlit reruns the page script
  on every interaction/file-save: the page scripts (`app_ui.py`,
  `pages/*.py`) only `import` modules like `orchestrator_agent` once
  per process, and Python caches that import in `sys.modules` — a
  Streamlit "rerun" re-executes the page script, not a fresh import of
  everything it imports. Confirmed directly while testing a fix to
  `orchestrator_agent.py`: the UI kept showing the old (buggy)
  behavior until the server process itself was stopped
  (`preview_stop`) and restarted (`preview_start`), not just
  re-navigated or rerun. Always restart the server after editing
  anything under `src/`, not just after editing a `pages/*.py` file or
  `app_ui.py` itself (those two *do* get fully re-executed per rerun,
  since Streamlit treats the page script specially).
