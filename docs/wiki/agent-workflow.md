# Agent Workflow

This page defines the required workflow for LLM agents working in this repository.

## Before Starting Any Substantial Task

1. Read `docs/wiki/index.md` and the relevant wiki pages.
2. **Search before writing.** For any new class, function, config, or wrapper:
   - `grep -r "ClassName\|function_name" myosuite/` — does it already exist?
   - Check `docs/wiki/library-usage.md` — does a library already provide this?
   - Check `docs/wiki/repository-map.md` — where is the canonical home for this kind of code?
   If something similar exists within ~80% similarity, extend or import it. Never duplicate.
3. For non-trivial tasks (3+ steps or architectural decisions): enter plan mode, write a plan to `tasks/todo.md`, and verify with the user before implementing.
4. For bug reports: diagnose root cause and fix directly — no hand-holding needed.

## Core Work Loop

1. Read relevant wiki pages and code before making changes.
2. Implement changes following `docs/wiki/engineering-standards.md`.
3. Verify correctness (tests, logs, diffs) before declaring done.
4. Update wiki pages affected by the change.
5. Append an entry to `docs/wiki/log.md`.

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items.
2. **Track Progress**: Mark items complete as you go.
3. **Capture Lessons**: After any user correction, update `tasks/lessons.md` with the pattern to avoid repeating it.

## Verification Before Done

- Never mark a task complete without proving it works.
- Run the relevant test suites (see `CLAUDE.md` for the required commands).
- Ask yourself: "Would a staff engineer approve this?"

## Subagent Strategy

- Use subagents to keep the main context window clean.
- Offload research, exploration, and parallel analysis to subagents.
- One focused task per subagent.
- **Subagents have no memory of sibling sessions.** A subagent writing mjlab backend code does not know what another subagent wrote in a sibling file. Always scope task prompts to include: *"Search for existing implementations of X before writing anything new."* Cross-file replication is the most common agent failure mode in this repo.

## When Wiki Updates Are Required

Update wiki pages whenever a change impacts:

- Repository structure or module ownership boundaries
- Architecture, contracts, or extension points
- Coding standards, testing expectations, or workflow policies
- Agent operating assumptions about where logic should live

## Update Rules

- Keep edits incremental; do not rewrite unrelated wiki pages.
- Prefer short, factual updates over speculative prose.
- If unsure where to put information, update the closest existing page and cross-link.
- Do not leave TODO placeholders without context and owner intent.

## Query and Navigation Pattern

- Start at `wiki/index.md`.
- Read the most relevant thematic page(s).
- Only then drill into code files.
- Ground claims in concrete file paths and tests when possible.

## Lint / Health Checks

During maintenance passes, look for:

- Stale statements that no longer match repository reality
- Broken or missing links between wiki pages
- Contradictions between `CLAUDE.md` and wiki guidance

If contradictions exist, align wiki to `CLAUDE.md` and log the correction.
