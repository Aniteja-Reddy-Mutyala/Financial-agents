---
name: session-handoff
description: Compact the current conversation into a handoff brief and actually start a new session from it, so the user moves to a fresh session with the relevant context already loaded instead of a file they'd have to feed in themselves. Use this proactively whenever the user says something like "wrap up", "hand off", "let's end here", "I'm done for now", or otherwise signals the session is ending — not just on an explicit /session-handoff invocation.
argument-hint: "What will the next session be used for?"
---

## What this does

1. Review the current conversation and compile a handoff brief:
   - What's done (reference commits/PRs/files by path or identifier —
     don't restate their content; a fresh session can read the diff
     itself).
   - What's pending or deferred, and why.
   - Any standing decisions/preferences the user set this session that
     aren't already written down elsewhere (CLAUDE.md, memory, a skill)
     — check those first so nothing gets duplicated.
   - A concrete immediate next step, if there is one.
   - A "suggested skills" list: which of this project's skills the new
     session should load before continuing.
   - Redact anything sensitive (API keys, passwords, PII) before this
     brief goes anywhere.
   - If the user passed arguments when invoking this skill, treat them as
     what the next session should focus on and tailor the brief to that,
     rather than trying to cover everything.

2. **Start the new session from that brief** — call
   `mcp__ccd_session__start_session` (load it via `ToolSearch` first if
   it's not already loaded):
   - `initiation: "user_asked"` — invoking this skill *is* the explicit
     ask for a new session now.
   - `context: "fresh"`, deliberately **not** `"fork"` — the whole point
     of a handoff is a compact brief, not carrying this session's full
     transcript (and its token cost) into the new one. `"fork"` would
     defeat the purpose.
   - `use_worktree: false` unless the current session itself is in its
     own worktree — this is a continuation of the same ongoing work in
     the same checkout, not an isolated parallel task that needs its own
     branch.
   - `prompt`: the task-oriented part of the brief (what's done, what's
     pending, the immediate next step) — put file paths and identifiers
     in backticks.
   - `background`: at least 80 characters covering the overall goal,
     decisions, and constraints so far — written for someone who hasn't
     seen this conversation at all.
   - `title`: short (under 60 chars), names the task being continued.

3. Tell the user the new session has started and name it/link it so they
   can switch to it — don't just say "done," confirm where to go.

## Optional: also save a durable copy

The compiled brief can also be written to the OS temp directory
(`$TMPDIR`, else `/tmp`; `%TEMP%` on Windows) as a backup reference, the
same way the original skill this is based on always did — useful if the
user wants a copy outside any session, or the brief needs to be longer
than what's practical to pass directly. This is secondary now, not the
primary mechanism: the new session itself is the handoff, not a file
someone has to go find and paste in.

## Attribution

Based on
https://github.com/mattpocock/skills/blob/main/skills/productivity/handoff/SKILL.md
(MIT licensed) — specifically its instructions for *compiling* a handoff
brief (what to include, what to skip, redaction, argument handling) are
preserved. The delivery mechanism is a real deviation from the original:
Pocock's version always wrote a passive file to disk for a human or a
different agent to manually feed into a new session; this version
instead uses this environment's own session-starting tool
(`mcp__ccd_session__start_session`) to actually move the user into a new
session pre-loaded with the brief. That capability doesn't exist in
every host this skill might run in — if `start_session` (or an
equivalent) isn't available, fall back to writing the brief to the temp
directory as the original did, and tell the user where it is and that
they'll need to paste or reference it themselves.
