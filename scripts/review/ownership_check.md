Judge one question about this MyoSuite pull request: does the code it adds belong in MyoSuite, and does the description (commit message bodies, or the PR description) give
the ownership and reuse evidence that REVIEW.md asks for? Go by what the diff and the code on the branch and on the base branch do, not by
what the title or description claims.

Ownership, as REVIEW.md and docs/wiki/repository-map.md state it:
- `myo_sim` owns reusable model construction and anatomical/physical transforms: geometry and segment scaling, mass and
  inertia, muscle and tendon parameters, model-only calibration, MJCF/XML model definitions and their composition.
- MyoSuite owns measurement and source adapters (`myosuite/integrations/`), shared numerical helpers (`myosuite/physics/`),
  scientific analysis and subject-fitting solvers (`myosuite/biomechanics/`), task configuration, rewards, observations,
  runtime policy inference, visualization (`myosuite/viz/`) and CPU/mjlab wiring (`myosuite/envs/`).
- Out-of-wheel `training/<method>/` owns RL trainers, replay buffers, world models and method-specific architectures
  (`docs/wiki/training-guidelines.md`); out-of-wheel `tools/<tool>/` owns heavy CV/perception inference stacks. Runtime
  modules under `myosuite/` must never import `training`, `tools`, `scripts` or `myosuite.tests`.
- Inside `myosuite/`, new runtime code belongs in its active canonical subpackage following the upward import DAG in
  `docs/wiki/repository-map.md` — never in closed compatibility shim namespaces (`myosuite/scenes/`, `myosuite/logger/`,
  migrated `myosuite/terms/` and `myosuite/utils/` modules), and runtime policy loading/export belongs in
  `myosuite/policies/` rather than `myosuite/benchmarks/`.
- A thin compatibility facade or re-export shim is fine. A copy of a `myo_sim` algorithm, a fallback that
  re-implements one, or a new model transform written in MyoSuite is not.

pass: every new model transform in the diff calls the `myo_sim` API (or a linked, version-gated `myo_sim` PR), nothing in the
diff duplicates an existing primitive on the base branch, and the description names the owner, the existing symbols considered or
reused, and the code removed or why none could be. A change with no model code at all (docs, tests, task wiring, training)
passes when it duplicates no primitive.
fail: anything else, or you cannot tell.

Report each problem with file:line. Title it by kind and file: "wrong owner: <file>",
"duplicate of <existing symbol>: <file>", or, with no file, "missing ownership evidence". One finding per file and kind; name
the other lines in its text. Say in the text which existing symbol or `myo_sim` API should be used instead.
Severity: high for a model transform or duplicated algorithm in MyoSuite; medium for missing ownership/reuse evidence in the
description; low for a facade that should be thinner.
