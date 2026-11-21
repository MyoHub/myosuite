# Performance cards

`myosuite.utils.performance_card` summarizes data-only measurements from any
training process. It imports NumPy/SciPy, with no trainer or RL dependency.
`RunRecord.from_dict(record)` validates the JSON schema; `summarize_runs(runs)`
returns a JSON-compatible card. The producer measures timings and quality;
this tool does not benchmark a trainer or change its defaults.

Each input JSON contains one record or a list. Required fields:

| Fields | Meaning |
|---|---|
| `config_id`, `env_id`, `seed` | Configuration, registered task and independent training seed |
| `contract_id` | Identity binding physical model, task, quality protocol and evaluation distribution |
| `startup_s`, `collection_s`, `learning_s`, `total_wall_s` | Finite nonnegative seconds; disjoint phases within positive total time |
| `transitions`, `peak_memory_bytes` | Nonnegative integer totals |
| `clock_origin`, `memory_scope` | Explicit measurement scopes, e.g. `script_entry` and `torch_allocations_excludes_warp_driver` |
| `quality_metric`, `target`, `quality_points` | Metric name, threshold and ordered `{wall_s, transitions, value}` events |

`higher_is_better` defaults to true. All event wall times use the declared clock
and lie within run totals. `script_entry` excludes OS/Python bootstrap; Torch
allocation peaks exclude Warp and driver memory and are not total VRAM. Record
hardware/software, timing synchronization and evaluation schedule in the
configuration artifact. A card rejects mixed task/physics/evaluation identities,
metric/target/direction, clocks or memory scopes, and duplicate seeds per config.

```bash
python scripts/report_performance.py seed*.json --output card.json --seed 0
```

Means and percentile bootstrap intervals resample independent training seeds
(default 2,000 resamples, 95% confidence), never checkpoints or evaluation
episodes. A single seed has no interval. Small seed counts provide weak
uncertainty estimates; evaluation noise is not a separate uncertainty model.
Throughput is the mean of per-run rates, not a ratio of pooled totals.

The first **observed** threshold crossing is retained even if quality later
drops. It is neither interpolated nor a sustained-success guarantee. Unreached
runs are right-censored at the last quality evaluation, which can precede exit.
Restricted crossing costs cap every run at the earliest final evaluation across
all configurations (separately in seconds/transitions); failures contribute
that horizon. These are capped observed-event costs, not continuous-time
survival estimates or successful-only time-to-target means. An empty quality
trace is `unobserved` and disables restricted costs for the whole comparison.
Raw quality events remain in the card. Inspect both wall time and transitions;
a short pilot establishes no asymptotic learning or hardware ranking.
