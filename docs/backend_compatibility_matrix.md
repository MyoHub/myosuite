# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

# Backend Compatibility Matrix (Gym vs MJX vs MJLab)

This document defines a **prescriptive backend support policy** for MyoSuite.
It separates:

- what is registered in Gymnasium (broad catalog), from
- what is intentionally supported across backends (narrow stable set), and
- what numerical parity guarantees currently exist.

## 1) Environment Catalog Size (Current Codebase)

From registration modules:

- Gym/MyoBase base IDs: 36 (`myosuite/envs/myo/myobase/__init__.py`)
- Gym/MyoChallenge base IDs: 19 (`myosuite/envs/myo/myochallenge/__init__.py`)
- Gym/MyoMimic named tasks: 4 (`myosuite/envs/myo/myomimic/__init__.py`)

Total base IDs before automatic variants: **59**.

Note: many Gym IDs also auto-register muscle-condition variants (`Sarc`, `Fati`,
and some `Reaf`) via `register_env_variant`, which expands the practical Gym
catalog beyond 145.

> **MyoChallenge registration (v4+):** All MyoChallenge environments
> (`Bimanual`, `TableTennis`, `Soccer`, `RunTrack`, `ChaseTag`) now register
> directly against their Gymnasium-native implementations. The transitional
> `*Native-v0` duplicate IDs have been removed.

## 2) Prescriptive Cross-Backend Support Tiers

Use this tiering for docs, CI expectations, and user-facing recommendations.

### Tier A (recommended, cross-backend focus)

These are the envs with the strongest backend story today:

1. `myoElbowPose1D6MFixed-v0`
2. `myoLegWalk-v0`
3. `myoSarcLegWalk-v0` (MJLab + Gym; MJX variant not yet exposed)

Rationale:

- Explicitly exercised in MJLab integration tests.
- Covered by backend parity tests (Gym vs MJX and/or Gym vs MJLab).
- Already used as backend benchmark anchors.

### Tier B (MJX-supported but narrower parity coverage)

MJX env names (`myosuite/envs/myo/backends/mjx/__init__.py`):

- `MjxElbowPoseFixed-v0`
- `MjxElbowPoseRandom-v0`
- `MjxFingerPoseFixed-v0`
- `MjxFingerPoseRandom-v0`
- `MjxHandReachFixed-v0`
- `MjxHandReachRandom-v0`
- `MjxLegWalk-v0`
- `MjxMimicBimanual-v0` (alias: `MjxMuscleMimicBimanual-v0`)
- `MjxMimicFullbody-v0` (alias: `MjxMuscleMimicFullbody-v0`)

Policy:

- Treat elbow and leg walk as the parity reference pairings with Gym.
- Treat the remaining MJX envs as supported for MJX usage, but with
  limited strict cross-backend numerical contracts today.

### Tier C (Gym-only broad catalog)

All remaining Gym-only environments (including most MyoChallenge and MyoEdits
variants) should be considered Gym-first unless promoted into Tier A/B by:

- backend implementation maturity, and
- added parity tests with stable numerical tolerances.

## 3) Backend Compatibility Matrix (Prescriptive)

| Canonical task | Gym | MJX | MJLab | Status |
|---|---|---|---|---|
| Elbow pose fixed (`myoElbowPose1D6MFixed-v0`) | Yes | Yes (`MjxElbowPoseFixed-v0`) | Yes | **Tier A** |
| Leg walk (`myoLegWalk-v0`) | Yes | Yes (`MjxLegWalk-v0`) | Yes | **Tier A** |
| Leg walk sarcopenia (`myoSarcLegWalk-v0`) | Yes | No dedicated MJX ID | Yes | **Tier A (Gym+MJLab)** |
| Elbow pose random (`myoElbowPose1D6MRandom-v0`) | Yes | Yes (`MjxElbowPoseRandom-v0`) | Registered in MJLab tasks | Tier B |
| Finger pose fixed/random | Yes | Yes (`MjxFingerPose*`) | Partial (`FingerPoseCfg` task mapping) | Tier B |
| Finger reach random | Yes | Yes (`MjxFingerReachRandom-v0`) | Partial (`FingerReachCfg` task mapping) | Tier B |
| Hand pose random | Yes | Yes (`MjxHandPoseRandom-v0`) | Partial (`HandPoseCfg` task mapping) | Tier B |
| Hand reach fixed/random | Yes | Yes (`MjxHandReach*`) | Partial (`HandReachCfg` task mapping) | Tier B |
| Challenge baoding (`myoChallengeBaodingP2-v1`) | Yes | No | Registered in MJLab tasks | Experimental |
| Mimic bimanual (`myoMimicBimanual-v0`) | Yes | Yes (`MjxMimicBimanual-v0`) | Yes (`myoMimicBimanual-v0`) | Tier B |
| Mimic full-body (`myoMimicFullbody-v0`) | Yes | Yes (`MjxMimicFullbody-v0`) | Yes (`myoMimicFullbody-v0`) | Tier B |
| Remaining MyoBase/MyoChallenge/MyoEdits | Yes | Mostly No | Mostly No | Gym-only |

## 4) Numerical Differences and Guarantees (Current Tests)

### Gym vs MJX

From `myosuite/tests/test_backends.py`:

- Elbow pose trajectory parity:
  - `qpos` tolerance: `atol = 5e-3`
- Elbow pose reward parity:
  - cumulative reward relative tolerance: `<= 5%`
- Reach env:
  - currently smoke/health checks (finite values, expected reward sign),
    not strict stepwise parity.

From `myosuite/tests/test_parity.py`:

- CPU baseline parity tests validate consistency with stored rollout baselines.
- MyoChallenge-specific parity tests (`test_myochallenge_*_parity.py`) have
  been removed following the v4 migration to Gymnasium-native implementations.

### Gym vs MJLab

From `myosuite/tests/test_backends.py`, `test_myo_leg_walk_*_parity.py`, and
`test_tier_a_backend_mappings.py`:

- Elbow SAR parity is currently **loose**:
  - max joint-angle difference bound: `<= 1.5 rad`
  - reward boundedness check (finite and `|reward| <= 10`), not strict equality.
- Leg walk reward/state tests are primarily contract/sanity checks and
  diagnostic comparisons; strict numerical equivalence is not yet enforced.
- MJLab registry now includes both `myoLegWalk-v0` and `myoSarcLegWalk-v0`
  (`WalkCfg` mapping + task registration path).

Interpretation:

- MJLab is functionally integrated for key tasks, but numerical parity to Gym
  remains broader/tighter for MJX than for MJLab at present.

## 5) Practical Recommendation

For users who need backend portability and predictable numbers:

- Default to **Tier A** envs.
- Prefer **MJX** when strict numerical parity with Gym is required.
- Treat **MJLab** as compatible-but-evolving for now; use existing tests as
  guardrails and expand strict parity incrementally per task.

## 6) Promotion Criteria (to move a task into Tier A)

Require all of:

1. Task exists in Gym + MJX or Gym + MJLab.
2. End-to-end `make -> reset -> step` test coverage on target backend.
3. Explicit numerical parity contract:
   - state tolerance and/or reward tolerance documented in tests.
4. Stability in CI (no flaky backend-specific skips outside known platform
   constraints, such as optional GPU-only test gates).
