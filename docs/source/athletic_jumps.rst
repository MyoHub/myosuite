Athletic field events: long jump and high jump
==============================================

``long_jump`` and ``high_jump`` complete the athletic track-and-field set. Both start like the sprint (run pose at
the origin, the same lane rules on the runway), then World Athletics' event rules take over, as far as a
simulation can apply them. The rules, the rewards and the benchmark metrics live in one place,
``myosuite.envs.myo.tasks.basic.full_body.jumps``: :class:`JumpSpec` (geometry and weights) and
:class:`JumpJudge` (an online judge, vectorised over worlds, that needs only the body colliders as segments and
the whole-body centre of mass).

Events
------

.. list-table::
   :header-rows: 1
   :widths: 16 42 42

   * -
     - Long jump
     - High jump
   * - Geometry
     - take-off line 20 m down a 1.22 m runway; landing on the ground plane (no sand)
     - 30 mm bar along the side axis 15 m down the runway, standards 4 m apart; bar = static collider
   * - Valid attempt
     - the flight that leaves the ground behind the line and lands beyond it
     - every body collider passes the bar plane above the bar, without touching it, after a one-foot take-off
   * - Fails
     - foul (a supporting foot's front surface past the line, or another body part on the ground beyond it,
       before the jump); runway endings (fall, off lane, not facing, stalled)
     - knocked (collider touches the bar, tested on the swept motion between control steps), under / through
       (a body part passes below the bar), run-out past the standards, ground contact beyond the bar plane
       before clearing, two-foot take-off, runway endings
   * - Score
     - mark: line to the nearest ground-footprint point of any body part within 1 s of landing (a timed
       flat-ground surrogate of the "nearest break")
     - clearance rate per bar height over the ladder 0.4, 0.6, ..., 2.0 m (``high_jump_hXXX``); best height
       cleared, reported with its attempt budget
   * - Observation
     - + distance to the take-off line
     - + distance to the bar, bar height
   * - Reward
     - sprint terms + ``lambda_mark`` x mark once at the end of a valid jump
     - sprint terms with the approach speed paid up to 6 m/s, + COM rise inside the 3 m take-off zone,
       + ``lambda_clear`` once on a clearance

Lane, posture and fall rules are suspended from a legal take-off on (long jump: behind the line; high jump:
inside the 3 m take-off zone); the stall ending, the time limit and the numerical check stay active. Each
attempt record carries ``judge_version``.

Ground contact is "collider lowest point below the ground". Bolt's penalty contact pushes only when a collider
penetrates, and with that threshold the judge agrees with the plugin's ground reaction force (> 1 N) on 99.8 %
of control steps, with no missed contacts; a 1 cm threshold flagged 6 % of airborne steps as contact. That check
compares whole-body contact flags at control rate on non-terminal steps; it does not validate per-foot contact,
contact location or sub-step transitions, so the one-foot take-off rule is judged at control rate.

A body part passing *below* the bar fails the attempt even without contact: the colliders do not cover the
whole body (the neck between the trunk capsules and the head sphere), and an earlier version scored a runner who
ducked under a 1.6 m bar as a clearance.

Metrics
-------

``eval_jumps.py`` scores each world's first episode and writes the attempt records and a summary
(``jumps.summarize``).

Long jump: ``mean_score_m`` (headline: every attempt, fouls and endings scored 0), ``mark_m`` (valid jumps), ``effective_m`` (take-off toe to landing, i.e. mark + board gap),
``board_gap_m`` (distance lost behind the line), ``takeoff_vx`` / ``takeoff_vz`` / ``takeoff_angle_deg`` (COM at
take-off, angle against the total horizontal speed; vertical speed corrected by half a control step of gravity,
error at most 0.16 m/s at 30 Hz if the COM is ballistic over the release step; positions are the last contact
sample),
``flight_time_s`` (airborne control steps x 1/30 s), ``com_rise_m``, ``single_leg_rate``; per run ``valid_rate``,
``foul_rate`` and ``jump_rate``: valid attempts that are not a running stride over the line (flight under
0.2 s and take-off angle under 10 deg are flagged ``stride_like``; legal, but not a jump; a benchmark heuristic, not an athletics rule).

High jump: ``clear_rate`` per bar, ``best_height_cleared_m`` over the ladder, ``min_bar_gap`` (closest
collider-to-bar clearance while crossing; negative = contact), ``com_peak_up``, ``com_peak_minus_bar`` (peak of the judged flight, on
clearances; a flop clears with the COM near or below the bar), take-off angle and vertical speed,
``knocked_by`` (which collider hit the bar).

Running
-------

.. code-block:: bash

   python scripts/eval_jumps.py CKPT.pt --task long_jump --n 32 --out lj.json --trace lj
   python scripts/eval_jumps.py CKPT.pt --ladder --n 16 --out hj_ladder.json
   python scripts/render_jumps.py lj.json lj_long_jump.npz --out lj.mp4

A checkpoint trained without the event observation (a sprinter) runs blind: the event inputs are dropped.

Backends
--------

The judge runs on the fullbody C-plugin env (``CFullbodySimpleTD3Env(task="long_jump")``). The generic CPU
``ModularTaskEnv`` and mjlab terms do not apply it yet, so ``make_athletic("long_jump")`` raises
``NotImplementedError`` and the registration matrix skips both events, instead of registering a sprint that only
looks like a jump.
