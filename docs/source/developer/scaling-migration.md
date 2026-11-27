# Canonical Model Scaling

Geometry, OpenSim scale files and physical parameter edits belong to `myo_sim`.
MyoSuite keeps task integration, IK, marker retargeting and participant calibration.
`myosuite.core.body_scaling.scale_model` remains a compatibility facade and compiles
the specification returned by `myo_sim.scaling.scale_spec`.

The library API returns a copy. Both the CPU walk and its mjlab twin assign that
return value before compiling. The nominal source specification stays unchanged;
discarding the return value would silently discard subject scaling.

Subject scaling requires a `myo_sim` release that ships `myo_sim.scaling` ([myo_sim#149](https://github.com/MyoHub/myo_sim/pull/149); legacy `myo-sim==0.2.5` does not include this module and raises an actionable import error when scaling is requested). Nominal environments do not import the optional scaling module.

The existing `retarget`, `scale_spec`, `scale_model` and checkpoint digest names
resolve through `myosuite.core.body_scaling`. Private `_scale_spec` returns the
copied spec; internal consumers use the public `scale_spec` spelling. Skin factor
validation delegates to the same model library rather than retaining another
implementation.

Anisotropic scaling may change a muscle's wrap branch. The library reports these
tendons in `scale_report(...).skipped` with `changes its path`. Peak strength is
preserved, but force at the assembled default pose need not be identical for those
tendons. Inspect the report before interpreting calibration or comparing frozen
policies across morphologies.

The model-layer XML writer repair landed in
[myo_sim#185](https://github.com/MyoHub/myo_sim/pull/185), and the scaling rules,
unit tests and pinned OpenSim `ModelScaler` comparisons live in `myo_sim`
([myo_sim#149](https://github.com/MyoHub/myo_sim/pull/149)). MyoSuite retains
the compatibility facade, `make_env(..., scales=..., total_mass=...)` across CPU
and mjlab, skin binding, checkpoint manifest provenance and task integration tests.
