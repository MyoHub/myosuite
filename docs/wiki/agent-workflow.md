# Agent Workflow

## Before Starting

1. Read `docs/wiki/index.md`, then the relevant thematic page(s).
2. **Search before writing** — `grep -r "keyword" myosuite/` and check `library-usage.md`. If something similar exists (≥ 80%), extend or import it.
3. For tasks with 3+ steps or architectural decisions: write a plan to `tasks/todo.md` and confirm with the user.

## Work Loop

1. Implement following `engineering-standards.md`.
2. Verify correctness (tests, diffs). Run the commands in `CLAUDE.md`.
3. Update any wiki page affected by the change.
4. Append an entry to `docs/wiki/log.md`.

## After a User Correction

Update `tasks/lessons.md` with the pattern to avoid repeating it.

## Subagent Strategy

- One focused task per subagent — subagents have no memory of sibling sessions.
- Always scope subagent prompts with: *"Search for existing implementations of X before writing anything new."*
- Use subagents for research, exploration, and parallel analysis to keep the main context clean.

## Wiki Maintenance

Update wiki pages when a change affects: repository structure, architecture contracts, coding standards, or agent operating assumptions. Keep edits factual and incremental.
