# Canonical model scaling

Geometry, OpenSim scale files and physical parameter edits belong to `myo_sim`.
MyoSuite keeps task integration, IK, marker retargeting and participant calibration.
`myosuite.core.body_scaling.scale_model` remains a compatibility facade and compiles
the specification returned by `myo_sim.scaling.scale_spec`.

The library API returns a copy. Both the CPU walk and its mjlab twin assign that
return value before compiling. The nominal source specification stays unchanged;
discarding the return value would silently discard subject scaling.

This integration requires [myo_sim#149](https://github.com/MyoHub/myo_sim/pull/149).
Published `myo-sim==0.2.5` does not provide the scaling module. The feature raises
an actionable import error when requested with that release. Nominal environments
do not import the optional scaling module. Keep this PR draft until the dependency
is available and the supported dependency constraint can name its released version.
Do not install an unpinned development branch to make a production release pass.

The existing `retarget`, `scale_model`, checkpoint digest and private OpenSim names
continue to resolve at their old paths. Private `_scale_spec` now returns a copy;
internal consumers use the public `scale_spec` spelling. Skin factor validation
delegates to the same model library rather than retaining another implementation.

Anisotropic scaling may change a muscle's wrap branch. The library reports these
tendons in `scale_report(...).skipped` with `changes its path`. Peak strength is
preserved, but force at the assembled default pose need not be identical for those
tendons. The tests preserve the tight force comparison for every other actuator
and bound the number of reported exceptions. Inspect the report before interpreting
calibration or comparing frozen policies across morphologies.

The model-layer XML writer repair is tracked in
[myo_sim#185](https://github.com/MyoHub/myo_sim/pull/185). The independent OpenSim
fixtures and task integration tests remain in MyoSuite; moving the implementation
does not turn software regression tests into independent human validation.
