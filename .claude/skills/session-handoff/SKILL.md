---
name: session-handoff
description: Compact the current conversation into a handoff brief and start a fresh session pre-loaded with it. Use this proactively whenever the user says something like "wrap up", "hand off", "let's end here", "I'm done for now", or otherwise signals the session is ending — not just on an explicit /session-handoff invocation.
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

2. **Start a new session pre-loaded with the brief** —
   `mcp__ccd_session__start_session` with `initiation: "user_asked"`
   (the user explicitly asked for a handoff), `context: "fresh"`, a
   short `title` describing the task, `prompt` set to the compiled
   brief, and `background` (required with `context: "fresh"`, at least
   80 characters) summarizing the overall goal and how this handoff
   fits — the call is refused without it. Omit `use_worktree` so it
   inherits this project's default unless the user's request implies
   otherwise.

   This hands the brief to the new session directly as its task —
   no disk buffer, no hook, no `/clear` round-trip. The new session is
   a separate row the user can open; it is not the same session id as
   this one.

3. Tell the user, in this same turn, that the new session has started
   and link it (`[title](#local_<id>)` using the id `start_session`
   returned) so they can open it.

## Optional: also save a durable copy

The compiled brief can also be written to the OS temp directory
(`$TMPDIR`, else `/tmp`; `%TEMP%` on Windows) as a backup reference, the
same way the original skill this is based on always did — useful if the
user wants a copy outside the new session itself.

## Attribution

Based on
https://github.com/mattpocock/skills/blob/main/skills/productivity/handoff/SKILL.md
(MIT licensed) — specifically its instructions for *compiling* a handoff
brief (what to include, what to skip, redaction, argument handling) are
preserved. The delivery mechanism differs from the original, which
always wrote a passive file to disk for a human or a different agent to
manually feed into a new session; this version instead hands the brief
directly to a freshly started session via `mcp__ccd_session__start_session`.

An earlier version of this skill tried to keep the user in the *same*
session by buffering the brief to disk, running this environment's
`/clear`-equivalent tool (`mcp__ccd_session_mgmt__clear_session`), and
relying on a user-level `SessionStart` hook
(`~/.claude/hooks/session-handoff-inject.py`) to re-inject the buffer as
the cleared session's starting context. That didn't work: in this
environment (the Claude Code desktop app's "Code tab"), `clear_session`
is an app-native RPC that does not go through the CLI's hooks.json
`SessionStart` pipeline, so the hook never fired — confirmed by a
buffer file left unconsumed on disk and zero `SessionStart`/hook-related
lines in the app's own log, next to it logging its own internal hook
events constantly. That hook script and its wiring in
`~/.claude/settings.json` (`hooks.SessionStart` matcher `"clear"`) have
been removed as dead weight; this version uses the
`start_session`-based approach instead, which is natively supported
here.
