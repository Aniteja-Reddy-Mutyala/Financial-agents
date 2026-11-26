---
name: session-handoff
description: Compact the current conversation into a handoff brief, buffer it, then run /clear and have a SessionStart hook feed the brief back in as the fresh session's starting context — so the user keeps working in the *same* session (same id, same window) instead of being moved to a new one. Use this proactively whenever the user says something like "wrap up", "hand off", "let's end here", "I'm done for now", or otherwise signals the session is ending — not just on an explicit /session-handoff invocation.
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

2. **Write the brief to the handoff buffer**, keyed to the current
   working directory so it survives the clear below:

   ```bash
   KEY=$(python3 -c "import hashlib,os; print(hashlib.md5(os.getcwd().encode()).hexdigest())")
   mkdir -p ~/.claude/session-handoff-buffers
   cat > ~/.claude/session-handoff-buffers/"$KEY".md <<'EOF'
   <the compiled brief>
   EOF
   ```

   This pairs with the `SessionStart` hook at
   `~/.claude/hooks/session-handoff-inject.py` (wired in
   `~/.claude/settings.json` on the `"clear"` source) — it watches for a
   buffer file keyed by the *new* session's cwd and, if found, feeds its
   contents back in as `additionalContext` and deletes it. If that hook
   isn't present on this machine (check
   `~/.claude/hooks/session-handoff-inject.py` exists and
   `~/.claude/settings.json` wires it under `hooks.SessionStart` with
   `"matcher": "clear"`), set it up first — see the hook's own docstring
   — or fall back to the temp-directory approach in the "Optional" section
   below, since without the hook nothing will re-inject the buffer.

3. Tell the user, in this same turn, what's about to happen (this is the
   user's last chance to read anything from this session before the
   transcript clears — say it *before* step 4, not after):
   - A short confirmation that the brief is buffered.
   - That you're about to run `/clear` and the next turn in this same
     session will open with that brief already loaded.

4. **Clear the session** — call `mcp__ccd_session_mgmt__clear_session`
   with `session_id: "self"` (load it via `ToolSearch` first if it's not
   already loaded). Per that tool's own description, the clear happens
   once this turn ends, so nothing you say after this call is seen by
   anyone — step 3's message is the actual handoff notice to the user,
   not this call.

## Optional: also save a durable copy

The compiled brief can also be written to the OS temp directory
(`$TMPDIR`, else `/tmp`; `%TEMP%` on Windows) as a backup reference, the
same way the original skill this is based on always did — useful if the
user wants a copy outside the buffer mechanism, or as the fallback when
the SessionStart hook isn't set up on this machine (see step 2). This is
secondary, not the primary mechanism.

## Attribution

Based on
https://github.com/mattpocock/skills/blob/main/skills/productivity/handoff/SKILL.md
(MIT licensed) — specifically its instructions for *compiling* a handoff
brief (what to include, what to skip, redaction, argument handling) are
preserved. The delivery mechanism is a deviation from both the original
and an earlier version of this skill: Pocock's version always wrote a
passive file to disk for a human or a different agent to manually feed
into a new session; an earlier version of this skill started a brand
new session (`mcp__ccd_session__start_session`) pre-loaded with the
brief. This version instead keeps the user in the *same* session by
buffering the brief to disk, running this environment's own
`/clear`-equivalent tool (`mcp__ccd_session_mgmt__clear_session`), and
relying on a user-level `SessionStart` hook
(`~/.claude/hooks/session-handoff-inject.py`) to re-inject the buffer as
the cleared session's starting context. That hook is environment setup,
not something this skill invocation can create on the fly if missing —
see step 2's fallback.
