# Proposal: agent-assisted PR review

Draft for discussion with the core team; this does not change current rules or enable automation. Intended for MyoSuite and myo_sim.

## Merge policy

| Change | Proposed gate |
|---|---|
| Routine docs, changelog, wording | Auto-merge after relevant validation and agent review |
| Bug fix preserving supported contracts | Auto-merge after verified reproduction, regression checks, and agent review |
| Correct fix with intentional compatibility or behavior changes | Human approval of the consequences, then auto-merge after checks |
| New feature, architecture, design or agent instructions | Human agreement on scope/design before implementation; human implementation approval before merge |
| Unproven bug, uncertain impact, or newly introduced regression | Block pending evidence or resolution |

Mixed changes take the stricter gate. Auto-merge eligibility initially applies only to core-team contributions and explicitly authorized agent work.

## Prove bugs before fixing them

Every bug-fix PR must provide:

- Expected behavior and its basis: documented contract, approved design, reference calculation, or established supported behavior.
- Minimal reproduction: command, smallest input, dependencies, seed where relevant, and exact target/proposed commits.
- The same reproduction failing for the claimed reason before the fix and passing afterward; retain a focused regression test where practical.
- Relevant regression results and any checks not run.

A test assertion alone does not establish correct behavior. Setup/import failures do not prove an unrelated bug. GPU, stochastic, and rendering defects may use a reproducible script with measured results. Unexecuted evidence is unverified and cannot qualify for automatic merging.

Agent reviewers must also substantiate blocking bug findings with a reproduction or verifiable counterexample; unsupported concerns remain questions.

## Compatibility requiring human review

- MyoSuite: observation/action contracts, rewards, termination, reset distribution, timing, CPU/mjlab parity, published checkpoints.
- myo_sim: behavior-changing joint, tendon, muscle, inertia, contact, or composition edits; downstream policy compatibility.
- Both: public API/support changes, dependencies, licensing, CI/releases, design/agent/review rules, changed baselines or relaxed tolerances.

A correct scientific change can still break existing policies or comparisons. Accidental regressions block merge; intentional changes require explicit acceptance and compatibility notes.

## Proposed enforcement

1. Existing CI verifies relevant behavior; an agent publishes one updated summary: category, reproduction, compatibility, checks, decision.
2. A required `review-policy` check evaluates trusted classification/evidence and human approvals. Labels and PR checkboxes alone are insufficient.
3. Low-risk PRs pass automatically. Other lanes require approval from an eligible human maintainer other than the author; agents acting through human accounts do not satisfy this gate.
4. New commits invalidate implementation approval and trigger re-evaluation; base changes invalidate affected validation. Design/scope changes require renewed agreement.
5. GitHub auto-merge waits for this gate and existing required checks. Keep release approval separate.

Candidate implementation: a small GitHub App shared by both repositories, with policy configuration protected from PR edits and no branch-rule bypass. Run PR code in unprivileged CI, separate from merge credentials. Selective approval would be enforced by the policy check rather than a blanket human-approval requirement on integration branches.

## Decisions to settle

- Agree on categories and the initial auto-merge scope.
- Define core-team eligibility, human approvers, and explicit design approval records.
- Choose the agent reviewer and trusted evidence format; confirm how human decisions are distinguished from agent actions.
- Choose GitHub App versus trusted workflow enforcement. Start in reporting-only mode, then enable routine docs and proven compatible fixes.
