# Repository Wiki Index

This wiki is the persistent knowledge layer for LLM agents working in this repository.
Agents must read this index first, then open the linked pages before making substantial changes.

## Core Pages

- `docs/wiki/repository-map.md` - High-level structure of the repository and where major concerns live.
- `docs/wiki/engineering-standards.md` - Coding, testing, review, and architecture rules to follow.
- `docs/wiki/agent-workflow.md` - Required workflow for ingesting changes, updating this wiki, and logging work.
- `docs/wiki/library-usage.md` - **Approved library feature map. Read before writing any helper, wrapper, or utility.**
- `docs/wiki/log.md` - Chronological append-only change log for wiki maintenance events.
- `docs/wiki/writing-term-functions.md` - How to write backend-agnostic obs/reward/action term functions.
- `docs/wiki/mjlab-design-guide.md` - **Canonical mjlab patterns. Read before writing any mjlab backend code.** Covers entity structure, state writes, DR, vectorization, contacts, new-task checklist, and anti-pattern reference.
- `docs/wiki/cross-backend-contract.md` - **Obs/action/timing invariants for CPU↔mjlab↔mujoco_wasm (mjswan) policy portability.** Read before exporting a policy for browser or cross-backend eval.

## Source of Truth Priority

When there is a conflict, follow this order:

1. Actual code and tests in the repository
2. `CLAUDE.md`
3. This wiki

If this wiki is stale, update it in the same change set as the code update.
