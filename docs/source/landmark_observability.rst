Landmark observability
======================

``myosuite.biomechanics.observability.landmark_observability`` evaluates which
infinitesimal joint motions a set of landmark positions can distinguish at one
posture. A small IK residual alone does not establish this. Axial landmarks, for
example, may fit while leaving forearm rotation unconstrained.

Reuse ``retarget.Landmark`` with named sites or body-local offsets in metres::

    from myosuite.biomechanics.observability import (
        independent_dofs, landmark_observability,
    )
    from myosuite.biomechanics.retarget import Landmark

    scope = [d.index for d in independent_dofs(model)
             if d.joint in {"shoulder", "elbow"}]
    result = landmark_observability(
        model, qpos, [Landmark("wrist", "", site="wrist_marker")],
        target_dofs=scope, translation_scale_m=0.1, rotation_scale_rad=1.0,
    )
    report = result.record()  # strict JSON; deficient condition_number is null

Indices address independent MuJoCo tangent velocities, after active polynomial
joint couplings. Omitted DOFs remain free nuisance variables: the analysis removes
landmark motion they can explain before assessing target rank. Only
``fixed_dofs`` explicitly assumes known motion. A missing finger landmark therefore
need not invalidate an arm scope, but a free root that mimics arm motion can.

Characteristic translation/rotation scales (metres/radians, optionally overridden
by ``dof_scales``) make mixed units comparable. Landmark weights are relative
multipliers; ``landmark_scale_m`` normalizes positions. These scales are not noise
estimates. Reports contain scope names, units, scales, rank, singular values,
conditioning and ambiguous coordinate groups. Nullspace loadings use scaled target
coordinates; they are not probabilities. Conditioning depends on chosen scales.

This local geometric diagnostic excludes joint limits, posture priors, orientations,
temporal evidence and sensor noise. It establishes neither global uniqueness,
statistical uncertainty nor tracking accuracy. Evaluate representative postures and
validate the tracking system separately.
