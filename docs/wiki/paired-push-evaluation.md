# Paired push evaluation

Use the existing CPU policy adapters and `PerturbationWrapper` to compare pushed
and unpushed rollouts from the same native simulator, controller/history and RNG
snapshot. This evaluates inference; it does not train a new algorithm.

```bash
python scripts/eval_push_recovery.py --config scripts/configs/walking_push_48.json --output results/push-nominal
python scripts/eval_push_recovery.py --config scripts/configs/walking_push_48.json --output results/push-jitter --seeds 0 1 2 --velocity-jitter-natural 0.02
```

The first command is a deterministic 48-cell sweep: four world directions, four
measured baseline heel-strike phases, and three force magnitudes (including zero).
The second is a separate 144-trial experiment with independent seeded world-XY
initial velocity jitter, SD `0.02 sqrt(g L)`. Both arms share each sampled state.
Repeated nominal reset seeds are not independent trials; the CLI rejects them.
Three seeds are a bounded pilot, not a precise population robustness estimate.

The pinned legacy checkpoint drives the current CPU model at its actual 5 ms
control period. Its old GPU success percentage is not a result of this protocol.
The output records current physical model identity, native options/dependencies,
checkpoint bytes/source, ordered I/O where declared by the env, and adjacent
normalizer/agent metadata when available. Unknown legacy interface/normalization
details remain unknown; recording current provenance does not certify the old
training model as identical. `--checkpoint PATH` selects an explicit local SB3
`.zip` or RSL-RL `.pt` instead and records local byte identity.

Force is normalized by actual actor mass times gravity; time by `sqrt(L/g)`.
The example's `L=0.9 m` is a declared reference length, not an inferred anatomical
measurement. SI conditions are available through `PushProtocol`. Durations round
up, starts to the nearest control step. A body-local application point is measured
from body origin; omitted means COM. Body/world force coordinates are explicit.
The world wrench is sampled at each control boundary and held over its substeps.

Contact phase uses two baseline entries of the declared heel group after minimum
contact/swing dwell, ignoring initial standing contact. Phase zero is the selected
heel strike and one the next same-heel strike. The baseline bracket determines one
start step for both arms, without online phase retiming after the push. A missing
bracket is explicitly unavailable/unexposed. Reset-clock stride phase is a separate
option and never labelled measured gait phase. Contact force threshold and foot/
ground body sets are recorded; these are contact entries, not force-plate events.

Recovery means return to paired-baseline speed/heading for a continuous declared
hold interval, with no subsequent fall in the horizon. It is separate from target
speed tracking; the walking example requires its baseline to track 1 m/s within
the declared mean-error tolerance. Lateral drift is reported separately unless a
recovery tolerance is explicitly set. Height/tilt falls, task termination, horizon
truncation, baseline ineligibility and unrecovered censoring remain separate.
Delivered scalar/vector/angular impulses integrate actual trace wrenches, including
partial doses. Muscle excitation, activation and actuator force in N are separate
signals; none is labelled EMG. Foot-placement differences match the first threshold
contact entry after the push independently in each arm; absent events are null.

Each run produces JSON, outcome CSV, numeric NPZ traces and initial controller/RNG
JSON. Load NPZ with `allow_pickle=False`. The native integration state is meaningful
only with the matching model/runtime. The complete Python replay graph remains in
memory; artifacts are not an arbitrary env-object deserializer. Custom stateful
controllers use `ReplayPolicy(get_state=..., set_state=...)` and callback-owned
mutable state uses explicit `ReplayStateOwner`s. Existing target-body background
wrenches are rejected. Renderers/model edits require a different replay contract.

Per-condition intervals use SciPy exact binomial and percentile bootstrap of
paired speed differences across eligible, exposed initial-state seeds. Duplicated
seeds/states disable intervals; correlated grid cells are never pooled as independent
trials. Nominal sweeps have descriptive counts only. `complete.json` appears last;
its absence marks an interrupted run. Existing output folders are protected.
Timings separate checkpoint resolution, env startup, policy/provenance, paired
rollout and artifact I/O; interpreter/import startup and cache state are not measured.
