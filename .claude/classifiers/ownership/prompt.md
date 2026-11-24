Judge one question about this MyoSuite pull request: does the code it adds belong in MyoSuite, and does the description give
the ownership and reuse evidence that REVIEW.md asks for? Go by what the diff and the code under head/ and default/ do, not by
what the title or description claims.

Ownership, as REVIEW.md states it:
- `myo_sim` owns reusable model construction and anatomical/physical transforms: geometry and segment scaling, mass and
  inertia, muscle and tendon parameters, model-only calibration, MJCF/XML model definitions and their composition.
- MyoSuite owns measurement and source adapters, scientific analysis, experiment and subject-fitting workflows, task
  configuration, rewards, observations, training and CPU/mjlab wiring.
- A thin compatibility facade that calls the `myo_sim` API is fine. A copy of a `myo_sim` algorithm, a fallback that
  re-implements one, or a new model transform written in MyoSuite is not.

pass: every new model transform in the diff calls the `myo_sim` API (or a linked, version-gated `myo_sim` PR), nothing in the
diff duplicates an existing primitive in default/, and the description names the owner, the existing symbols considered or
reused, and the code removed or why none could be. A change with no model code at all (docs, tests, task wiring, training)
passes when it duplicates no primitive.
fail: anything else, or you cannot tell.

Report each problem as a finding on the line it is about. Title it by kind and file: "wrong owner: <file>",
"duplicate of <existing symbol>: <file>", or, with no file, "missing ownership evidence". One finding per file and kind; name
the other lines in its text. Say in the text which existing symbol or `myo_sim` API should be used instead.
Severity: high for a model transform or duplicated algorithm in MyoSuite; medium for missing ownership/reuse evidence in the
description; low for a facade that should be thinner.

Before you report, read the findings that earlier runs left on this pull request.
- Still true: leave it open. Report it again, under the same title, only if its severity changed.
- Fixed by a later push or an updated description: resolve it, with one sentence on why.
- Resolved by a person: leave it resolved, and re-open it only if a later push brings the problem back.
