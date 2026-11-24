# Review rules

These rules apply to every reviewer of a MyoSuite pull request: people, Claude Code Review, Codex, Antigravity/Gemini and the local
`scripts/review/local_review.sh`. `AGENTS.md` (imported by `CLAUDE.md`) points here; edit this file, not copies of it.
The mirror file in `myo_sim` states the same ownership split from the library side.

## Scope

- Automated code review covers defects in the proposed source, tests and documentation. **Do not emit findings about commit Author, Committer, AI attribution or CLA identity, and do not request attribution rewrites.** Those checks belong to the commit-creation hooks and maintainer verification of GitHub's PR commit list, not the code-review sandbox.
- A review service may reconstruct a patch as a new local commit with a different SHA and service identity. Even `git show --format=fuller` on that object proves only the reconstructed object's metadata, not the submitted PR's authorship. Never treat local `HEAD`, a synthetic merge or a SHA absent from GitHub's PR commit list as the contributor's commit.
- The human-attribution requirement in `AGENTS.md` still applies when **creating commits**. This review scope does not waive it.

## Ownership and reuse checks

- Apply the ownership and minimum-change review below to the actual diff, not just the PR description. For each new model transform or abstraction, search the existing primitives and cite the canonical symbol/path when flagging duplication or the wrong owner. Model scaling, mass/inertia and muscle/tendon transforms belong in `myo_sim`; source processing, subject-fitting workflows, task wiring and provenance may remain in MyoSuite. A thin compatibility facade is allowed; a copied algorithm behind a fallback is not a second canonical implementation.
- Check whether configuration, composition or a small extension can replace new code, and whether superseded code/callers are removed within a compatible migration. Flag unnecessary parallel solvers, wrappers or frameworks only with concrete equivalent behavior/constraints; do not demand consolidation of different numerical objectives or use line count alone as a defect.
- Check all applicable contracts and validation rules in this file and the linked engineering guides: dependency/version gates, public imports/serialization/checkpoint compatibility, scientific provenance and diagnostics, supported-backend parity, and method-specific training/artifact requirements. Inspect the implementation and evidence; passing lint or absent CI does not prove compliance. If another repository or experiment is unavailable, say that it is unverified rather than assuming it passed.
- Keep defect findings actionable with the affected code, existing primitive or violated contract, consequence and safe correction. Separate optional simplification from defects; use severity based on impact, never inflate it to force visibility. In a review summary, when supported, state ownership/reuse, compatibility and checks run/not run; do not treat a bot's clean review as human design or implementation approval.
- **Catalog coverage and reuse:** before adding a term or feature, search `myosuite-terms --grep <pattern>` and the source for reuse. When adding, moving, renaming or removing public terms/features, verify catalog coverage and update discovery mappings and tests in the same PR. Treat the catalog as a discovery aid, not an exhaustive substitute for source search. Reviewers check coverage and justified exclusions; do not maintain a second feature registry.

## Ownership and minimum-change rules

- Decide the owner before adding code. `myo_sim` owns reusable model construction and anatomical/physical transforms: geometry and segment scaling, mass/inertia, muscle/tendon parameters and model-only calibration. MyoSuite owns measurement/source adapters, scientific analysis and experiment workflows, task configuration and CPU/mjlab wiring. A workflow estimating subject parameters may stay in MyoSuite; reusable anatomical/physical model transformations must use the canonical `myo_sim` API. Task settings, sensor calibration and experimental control parameters remain with their workflow. The split holds both ways: a pull request to `myo_sim` must not bring task, reward, training or experiment logic into the library.
- Search both repositories and the relevant dependency APIs before adding a helper, class, wrapper or solver. Record the existing symbols considered and why configuration, composition or a small extension does not suffice. Review behavior and constraints, not a guessed similarity percentage.
- Prefer the smallest compatible change. Reuse the current composition/physics primitives and public APIs; remove superseded code and update callers within the compatible migration; document any staged compatibility period. Keep only deliberate compatibility facades, not a second algorithm hidden behind a fallback. Preserve distinct numerical objectives when equivalence has not been demonstrated.
- If the required library API is not released, link the prerequisite `myo_sim` PR and gate the consumer on a version that provides it. Do not copy the implementation into MyoSuite to bypass that dependency.
- Every code PR and updated review summary must state the owner, reused primitives, necessary new logic and code removed/simplified (or why none can be removed), with compatibility and validation evidence. Missing evidence needs revision before approval. Added/deleted line counts inform review; they are not a correctness quota.
- Reviewers must check the actual diff and imports against that evidence. Existing lint/tests do not establish absence of duplication or correct ownership; do not describe them as enforcing this semantic review gate.

## Checks, approval and authorship

- Tests, lint and this review run locally through pre-commit (`pre-commit install` installs the pre-push review). State
  in the PR what ran and what did not.
- Documentation (`docs/**`, `*.md`) may auto-merge, except `REVIEW.md`, `AGENTS.md`, `CLAUDE.md` and files under `.claude/`, `.agents/`,
  `.github/` and `scripts/review/`, which take an approval like code. Everything else needs one approval from a person other than the one
  who opened the pull request.

- A pull request is opened by a person, under their own GitHub account, who owns its design and answers for it. Agents may
  write code locally for that person, and `scripts/review/local_review.sh --fix` may prepare
  follow-up fixes for the author to commit, but an agent
  or bot account must not open the pull request. Commits keep the human attribution rules of `AGENTS.md` (no AI co-author
  trailers: CLA).
- A bot's or an agent's clean review is not a human approval of design or implementation.

## What a review summary states

- Owner (`myo_sim` or MyoSuite) and why; the existing primitives reused, with paths; the new logic that was necessary; the code
  removed or simplified, or why none could be.
- Compatibility consequences, and the checks run versus not run.
