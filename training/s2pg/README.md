# Stochastic Stateful Policy Gradient PPO (`myosuite-s2pg`)

Stochastic Stateful Policy Gradient PPO ($\text{S}^2\text{PG}$-PPO) for MyoSuite musculoskeletal control tasks, implemented as a self-contained PyTorch training package under `training/s2pg/` (`myosuite_s2pg`) following `docs/wiki/training-guidelines.md`.

$\text{S}^2\text{PG}$ trains recurrent/stateful policies (`gru`, `lstm`, or `vanilla` MLP cells) **without** truncated backpropagation through time (BPTT) by treating the next hidden state $z_{t+1}$ as part of a stochastic joint action-and-state transition kernel and optimizing single-step shuffled minibatches with PPO.

## Attribution and Licensing

$\text{S}^2\text{PG}$ is **not** an original MyoSuite method. It was introduced by Davide Tateo, Firas Al-Hafez, Jan Peters, and collaborators:

> Firas Al-Hafez, Guoping Zhao, Jan Peters, and Davide Tateo. **Time-Efficient Reinforcement Learning with Stochastic Stateful Policies.** *International Conference on Learning Representations (ICLR)*, 2024.
> Reference repositories: [github.com/robfiras/s2pg](https://github.com/robfiras/s2pg) · [github.com/robfiras/loco-mujoco](https://github.com/robfiras/loco-mujoco) (`loco_mujoco/algorithms/experimental/s2pg_ppo_jax.py`)

Upstream LocoMuJoCo code is distributed under the MIT License (`Copyright (c) 2024 Al-Hafez`), preserved in [`LICENSE-loco-mujoco.txt`](LICENSE-loco-mujoco.txt). New MyoSuite/PyTorch code in this directory is licensed under Apache-2.0 ([`LICENSE`](LICENSE)).

| Component | Source | Notes |
|---|---|---|
| Method: stochastic internal state transition kernel, critic over $(o_t, z_t, a_{t-1})$, BPTT-free policy gradient | Al-Hafez, Zhao, Peters, Tateo (ICLR 2024) | Core $\text{S}^2\text{PG}$ formulation |
| `IndependentJointDist`, `ActorCriticS2PG` architecture (separate pre-RNN `obs` / `prev_action` embeddings, post-RNN `obs` skip-embedding, `gru`/`lstm`/`vanilla` transition cells, `ema_coef` hidden smoothing, orthogonal `gain=0.01` action head, separate `log_std_a` and `log_std_z`), and joint-ratio PPO update | Adapted from [`robfiras/loco-mujoco`](https://github.com/robfiras/loco-mujoco) (`loco_mujoco/algorithms/experimental/s2pg_ppo_jax.py`, MIT License) | Ported from JAX/Flax/Distrax to idiomatic PyTorch (`torch.nn` / `torch.distributions`) |
| `CpuVectorEnv` (`make_env`), `MjlabTensorVecEnv` (`s2pg-train-mjlab` zero-copy GPU adapter), apple-to-apple `FeedforwardPPOTrainer` (`baseline_feedforward_ppo`), `StatefulS2PGExportWrapper` (TorchScript / ONNX export), `s2pg-bench` (`FatigueWrapper` evaluation), and unit tests | New MyoSuite code (`Copyright (c) MyoSuite Authors`, Apache-2.0) | Consumes registered MyoSuite tasks without modifying `myosuite/` |

## Why $\text{S}^2\text{PG}$ Avoids Truncated BPTT

Standard recurrent PPO treats hidden transitions deterministically ($z_{t+1} = f_\theta(o_t, z_t, a_{t-1})$), which couples policy parameters across every step of an episode and requires storing sequential trajectory chunks and unrolling gradients backward through time (BPTT).

$\text{S}^2\text{PG}$ instead models a factorized diagonal Gaussian over both the environment action $a_t \in \mathbb{R}^{d_a}$ and the next latent state $z_{t+1} \in \mathbb{R}^{d_z}$:

$$
p_\theta(a_t, z_{t+1} \mid o_t, z_t, a_{t-1})
= \mathcal{N}\!\big(a_t;\, \mu_\theta^a(o_t, z_t, a_{t-1}),\, \operatorname{diag}(\sigma_a^2)\big)\,
  \mathcal{N}\!\big(z_{t+1};\, \mu_\theta^z(o_t, z_t, a_{t-1}),\, \operatorname{diag}(\sigma_z^2)\big)
$$

where $\mu_\theta^z(o_t, z_t, a_{t-1}) = (1 - \alpha) z_t + \alpha f_\theta(o_t, z_t, a_{t-1})$ with smoothing coefficient $\alpha = \text{ema\_coef} \in [0, 1]$ (`stored_hidden_dim` is $2 d_h$ for LSTM $[h_t, c_t]$ and $d_h$ for GRU or vanilla MLP).

1. **Rollout collection:** At step $t$, $(a_t, z_{t+1}) \sim p_{\theta_{\text{old}}}(\cdot \mid o_t, z_t, a_{t-1})$ is sampled and stored in a flat transition tuple $(o_t, z_t, a_{t-1}, a_t, z_{t+1}, \log p_{\theta_{\text{old}}})$. Whenever an episode ends (`terminated | truncated`), the carry fed into the next episode is reset to $(z_0 = 0, a_{-1} = 0)$, while preserving the sampled $z_{t+1}$ of the terminal transition for its log-density ratio.
2. **Temporal credit assignment:** Generalized Advantage Estimation ($\text{GAE}\text{-}\lambda$) bootstraps across non-terminal steps using the augmented critic $V_\phi(o_t, z_t, a_{t-1})$.
3. **Flat minibatch PPO update:** Because $z_{t+1}$ was sampled stochastically, PPO optimizes single-step shuffled minibatches using the **joint** log-probability ratio:

$$
r_t(\theta) = \exp\!\Big(
  \big[\log \pi_\theta(a_t \mid o_t, z_t, a_{t-1}) + \log q_\theta(z_{t+1} \mid o_t, z_t, a_{t-1})\big]
  - \big[\log \pi_{\theta_{\text{old}}}(a_t \mid o_t, z_t, a_{t-1}) + \log q_{\theta_{\text{old}}}(z_{t+1} \mid o_t, z_t, a_{t-1})\big]
\Big)
$$

with joint differential entropy bonus $\mathcal{H}(\pi_\theta) + \mathcal{H}(q_\theta)$, requiring zero sequence unrolling during gradient updates.
4. **Deterministic deployment:** At evaluation and export time, exploration noise on both $a_t$ and $z_{t+1}$ is disabled: starting from $(z_0 = 0, a_{-1} = 0)$, the policy steps deterministically via $(a_t, z_{t+1}) = (\mu_\theta^a, \mu_\theta^z)$.

## Installation

From a MyoSuite checkout (keeping method dependencies isolated from the root package):

```bash
pip install -e .
pip install -e "training/s2pg[test]"

# Optional GPU training extra (mjlab + tyro):
pip install -e ".[mjlab]"
pip install -e "training/s2pg[mjlab]"
```

Or directly from Git:

```bash
pip install myosuite
pip install "git+https://github.com/MyoHub/myosuite.git@dev#subdirectory=training/s2pg"
```

## Usage

### 1. Side-by-Side Benchmark (`s2pg-bench`)

Compare `PPO` (`FeedforwardPPOTrainer`, sharing identical Adam settings, LR schedule, minibatch count, GAE-$\lambda$, `RunningMeanStd`, and MLP widths) against `S2PG-PPO` (`rnn_type="gru"`) across seeds `0, 1, 2` on unmodified `myoElbowPose1D6MRandom-v0` and `myoHandPoseRandom-v0`:

```bash
s2pg-bench \
    --envs myoElbowPose1D6MRandom-v0 myoHandPoseRandom-v0 \
    --total-steps 32768 \
    --eval-interval 8192 \
    --seeds 0 1 2 \
    --n-envs 8 \
    --n-eval-episodes 20 \
    --out-json logs/s2pg/bench_unmodified.json
```

To evaluate under non-Markovian 3CC-r muscle fatigue (where unobserved muscle fatigue compartments make the observation stream partially observable; reported separately per `docs/wiki/training-guidelines.md`):

```bash
s2pg-bench \
    --envs myoElbowPose1D6MRandom-v0 myoHandPoseRandom-v0 \
    --fatigue \
    --seeds 0 1 2 \
    --out-json logs/s2pg/bench_fatigue.json
```

### 2. GPU `mjlab` Training (`s2pg-train-mjlab`)

Run `S2PGPPOTrainer` directly on `ManagerBasedRlEnv` GPU tensors (supporting `--feature` muscle stages just like `scripts/train_mjlab.py`):

```bash
s2pg-train-mjlab myoElbowPose1D6MRandom-v0 \
    --num-envs 1024 \
    --max-iterations 500 \
    --rnn-type gru \
    --hidden-state-dim 64

# With non-Markovian muscle fatigue enabled:
s2pg-train-mjlab myoElbowPose1D6MRandom-v0 \
    --num-envs 1024 \
    --feature fatigue \
    --rnn-type gru
```

### 3. Stateful TorchScript / ONNX Export

Export a trained `ActorCriticS2PG` with embedded `RunningMeanStd` statistics as a self-contained deterministic module `(obs, hidden_state, prev_action) -> (action, next_hidden_state)`:

```python
from pathlib import Path
from myosuite_s2pg import export_onnx_stateful, export_torchscript

scripted = export_torchscript(trainer.policy, output=Path("logs/s2pg/policy_stateful.pt"))
export_onnx_stateful(trainer.policy, output=Path("logs/s2pg/policy_stateful.onnx"))
```

## Benchmark against PPO

**One-sentence summary:** On GPU `mjlab` (`184,320` steps, `256` parallel envs, seeds `0, 1, 2`), $\text{S}^2\text{PG}$-PPO (`gru`, `hidden_state_dim=64`) trains recurrent GRU policies via flat shuffled minibatches without truncated BPTT at only **`+4.2%–16.3%` GPU wall-clock cost**, outperforming both MyoSuite's canonical `myo_ppo_runner_cfg` and matched `FeedforwardPPO` on 80-muscle 3D standing balance (`myoLegStandRandom-v0`: **`667.23 ± 26.69` return, `93.3% ± 2.4%` solved** in `22.75 s` vs. `639.04 ± 33.79` / `91.7%` for `FeedforwardPPO` and `541.51 ± 33.22` / `91.7%` for `myo_ppo_runner_cfg`), improving solved rate over stateless `FeedforwardPPO` on `myoFingerPoseRandom-v0` both unmodified (`28.3%` vs. `25.0%`) and by **`+5.0` percentage points under a Velocity-Masked POMDP** (`33.3% ± 4.7%` vs. `28.3% ± 2.4%` when `qvel` is masked), and edging stateless `PPO` on CPU 39-muscle `myoHandPoseRandom-v0` unmodified and under `FatigueWrapper` (`-317.97 ± 3.08` vs. `-318.41 ± 3.13`), while fixed-LR `S2PG`/`FeedforwardPPO` (`3e-4`) converge slower than KL-adaptive `myo_ppo_runner_cfg` on low-DOF 5–6 muscle tasks.

### 1. Multi-Task GPU `mjlab` Benchmark (`184,320` Steps, `256` Envs, Seeds `0, 1, 2`)

- **Hardware:** `NVIDIA GeForce RTX 4070 Ti SUPER`, Windows 10 (`10.0.26300`), Python `3.11.18`, PyTorch `2.15.0+cu132`, `mjlab 1.6.1`, `mujoco 3.11.0`, `num_envs = 256`, `30 × 24 × 256 = 184,320` steps per seed (`0, 1, 2`), evaluated deterministically over 20 seeded episodes (`eval_seed=10000..10019`) on unmodified CPU `make_env(env_id)`.

| Environment (Muscles) | Method | Env Steps | Final Return (mean ± std) | Final Solved % (mean ± std) | Train Wall-Clock (s) | Peak GPU VRAM (MB) |
|---|---|---:|---:|---:|---:|---:|
| `myoFingerPoseRandom-v0` (5M) | `myo_ppo_runner_cfg` *(KL-adaptive LR)* | 184,320 | **`243.54 ± 15.83`** | **`96.7% ± 2.4%`** | `15.34 ± 0.37` | `22.9` |
| `myoFingerPoseRandom-v0` (5M) | `FeedforwardPPO` *(matched)* | 184,320 | `-19.87 ± 24.44` | `25.0% ± 4.1%` | `7.44 ± 0.08` | `42.2` |
| `myoFingerPoseRandom-v0` (5M) | $\text{S}^2\text{PG}$-PPO (`gru`) | 184,320 | `-30.91 ± 5.55` | `28.3% ± 2.4%` | `8.64 ± 0.05` | `56.2` |
| `myoFingerReachRandom-v0` (5M) | `myo_ppo_runner_cfg` *(KL-adaptive LR)* | 184,320 | **`233.97 ± 62.23`** | **`33.3% ± 6.2%`** | `19.05 ± 0.64` | `25.6` |
| `myoFingerReachRandom-v0` (5M) | `FeedforwardPPO` *(matched)* | 184,320 | `10.18 ± 15.25` | `0.0% ± 0.0%` | `11.38 ± 0.21` | `45.3` |
| `myoFingerReachRandom-v0` (5M) | $\text{S}^2\text{PG}$-PPO (`gru`) | 184,320 | `-9.92 ± 3.05` | `0.0% ± 0.0%` | `12.51 ± 0.02` | `61.3` |
| `myoLegStandRandom-v0` (80M) | `myo_ppo_runner_cfg` *(KL-adaptive LR)* | 184,320 | `541.51 ± 33.22` | `91.7% ± 4.7%` | `28.55 ± 0.51` | `40.2` |
| `myoLegStandRandom-v0` (80M) | `FeedforwardPPO` *(matched)* | 184,320 | `639.04 ± 33.79` | `91.7% ± 2.4%` | `21.84 ± 0.08` | `63.3` |
| `myoLegStandRandom-v0` (80M) | $\text{S}^2\text{PG}$-PPO (`gru`) | 184,320 | **`667.23 ± 26.69`** | **`93.3% ± 2.4%`** | `22.75 ± 0.01` | `82.7` |

#### GPU `mjlab` Learning Curves vs. Environment Steps & Wall-Clock Time (Unmodified, 3-Seed Mean)

| Environment | Env Steps | `myo_ppo_runner_cfg` Return (Solved, Wall s) | `FeedforwardPPO` Return (Solved, Wall s) | $\text{S}^2\text{PG}$-PPO Return (Solved, Wall s) |
|---|---:|---:|---:|---:|
| `myoFingerPoseRandom-v0` | `0` | `-90.44` (`26.7%`, `0.0s`) | `-98.30` (`10.0%`, `0.0s`) | `-98.12` (`10.0%`, `0.0s`) |
| `myoFingerPoseRandom-v0` | `61,440` | `-30.43` (`46.7%`, `5.70s`) | `-25.14` (`18.3%`, `2.54s`) | `-73.53` (`16.7%`, `2.97s`) |
| `myoFingerPoseRandom-v0` | `122,880` | `109.96` (`76.7%`, `10.53s`) | `-25.04` (`23.3%`, `5.01s`) | `-34.89` (`23.3%`, `5.80s`) |
| `myoFingerPoseRandom-v0` | `184,320` | `243.54` (`96.7%`, `15.34s`) | `-19.87` (`25.0%`, `7.44s`) | `-30.91` (`28.3%`, `8.64s`) |
| `myoFingerReachRandom-v0` | `0` | `-23.12` (`1.7%`, `0.0s`) | `-23.65` (`0.0%`, `0.0s`) | `-23.64` (`0.0%`, `0.0s`) |
| `myoFingerReachRandom-v0` | `61,440` | `-2.98` (`0.0%`, `6.51s`) | `-13.54` (`1.7%`, `3.92s`) | `2.29` (`3.3%`, `4.26s`) |
| `myoFingerReachRandom-v0` | `122,880` | `94.41` (`18.3%`, `12.92s`) | `37.21` (`5.0%`, `7.64s`) | `-12.14` (`0.0%`, `8.40s`) |
| `myoFingerReachRandom-v0` | `184,320` | `233.97` (`33.3%`, `19.05s`) | `10.18` (`0.0%`, `11.38s`) | `-9.92` (`0.0%`, `12.51s`) |
| `myoLegStandRandom-v0` | `0` | `659.37` (`93.3%`, `0.0s`) | `658.91` (`95.0%`, `0.0s`) | `660.82` (`95.0%`, `0.0s`) |
| `myoLegStandRandom-v0` | `61,440` | `612.51` (`93.3%`, `10.34s`) | `656.26` (`93.3%`, `7.38s`) | **`692.04` (`93.3%`, `7.65s`)** |
| `myoLegStandRandom-v0` | `122,880` | `561.15` (`93.3%`, `19.38s`) | `645.92` (`91.7%`, `14.60s`) | `665.93` (`93.3%`, `15.20s`) |
| `myoLegStandRandom-v0` | `184,320` | `541.51` (`91.7%`, `28.55s`) | `639.04` (`91.7%`, `21.84s`) | **`667.23` (`93.3%`, `22.75s`)** |

#### Velocity-Masked POMDP Ablation (`qvel` Zeroed Out, Reported Separately per `training-guidelines.md`)

Zeroing joint velocities (`qvel` slice `4:8` on `myoFingerPoseRandom-v0`; `qvel` slice `35:69` on `myoLegStandRandom-v0`) tests recurrent state inference without truncated BPTT:

| Environment (`qvel` Masked) | Method | Env Steps | Final Return (mean ± std) | Final Solved % (mean ± std) | Train Wall-Clock (s) | Peak GPU VRAM (MB) |
|---|---|---:|---:|---:|---:|---:|
| `myoFingerPoseRandom-v0` (`qvel=0`) | `FeedforwardPPO` | 184,320 | `-7.68 ± 27.82` | `28.3% ± 2.4%` | `7.37 ± 0.06` | `54.9` |
| `myoFingerPoseRandom-v0` (`qvel=0`) | $\text{S}^2\text{PG}$-PPO (`gru`) | 184,320 | `-14.90 ± 15.43` | **`33.3% ± 4.7%`** | `8.57 ± 0.01` | `68.9` |
| `myoLegStandRandom-v0` (`qvel=0`) | `FeedforwardPPO` | 184,320 | **`708.95 ± 65.14`** | **`95.0% ± 0.0%`** | `21.67 ± 0.04` | `71.1` |
| `myoLegStandRandom-v0` (`qvel=0`) | $\text{S}^2\text{PG}$-PPO (`gru`) | 184,320 | `687.61 ± 43.50` *(`701.95` @ `61k`)* | `91.7% ± 2.4%` *(`95.0%` @ `61k`)* | `22.80 ± 0.08` | `90.5` |

### 2. CPU `s2pg-bench` Reference (`32,768` Steps, `8` Envs, Seeds `0, 1, 2`)

- **Hardware:** Windows 10 (`10.0.26300`), `AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD` (24 logical CPUs), Python `3.11.18`, MuJoCo `3.8.1`, PyTorch `2.13.0+cpu`, `CpuVectorEnv` × 8 envs, 20 deterministic seeded evaluation episodes (`eval_seed=10000..10019`).

#### CPU Unmodified Registered Environments (`seeds = 0, 1, 2`)

| Environment | Method | Env Steps | Final Return (mean ± std) | Final Success (mean ± std) | Train Wall-Clock (s) | Peak Memory (MB) |
|---|---|---:|---:|---:|---:|---:|
| `myoElbowPose1D6MRandom-v0` | PPO (feedforward) | 32,768 | `767.98 ± 3.53` | `100.0% ± 0.0%` | `12.28 ± 0.10` | `67.9` |
| `myoElbowPose1D6MRandom-v0` | $\text{S}^2\text{PG}$-PPO (`gru`) | 32,768 | `650.15 ± 125.69` | `75.0% ± 31.9%` | `15.56 ± 0.32` | `0.9` |
| `myoHandPoseRandom-v0` | PPO (feedforward) | 32,768 | `-316.08 ± 2.49` | `0.0% ± 0.0%` | `22.36 ± 0.28` | `4.3` |
| `myoHandPoseRandom-v0` | $\text{S}^2\text{PG}$-PPO (`gru`) | 32,768 | `-316.06 ± 2.35` | `0.0% ± 0.0%` | `25.81 ± 0.23` | `1.0` |

### Learning Curves vs. Environment Steps & Wall-Clock Time (Unmodified, 3-Seed Mean)

| Environment | Env Steps | `PPO` Return (Success, Train s) | $\text{S}^2\text{PG}$-PPO Return (Success, Train s) |
|---|---:|---:|---:|
| `myoElbowPose1D6MRandom-v0` | `0` | `1.03` (`5.0%`, `0.00s`) | `0.84` (`5.0%`, `0.00s`) |
| `myoElbowPose1D6MRandom-v0` | `8,192` | `384.27` (`50.0%`, `3.05s`) | `305.72` (`28.3%`, `3.89s`) |
| `myoElbowPose1D6MRandom-v0` | `16,384` | `733.59` (`96.7%`, `6.15s`) | `428.08` (`40.0%`, `7.75s`) |
| `myoElbowPose1D6MRandom-v0` | `24,576` | `762.90` (`100.0%`, `9.23s`) | `589.22` (`60.0%`, `11.68s`) |
| `myoElbowPose1D6MRandom-v0` | `32,768` | `767.98` (`100.0%`, `12.28s`) | `650.15` (`75.0%`, `15.56s`) |
| `myoHandPoseRandom-v0` | `0` | `-316.51` (`0.0%`, `0.00s`) | `-315.71` (`0.0%`, `0.00s`) |
| `myoHandPoseRandom-v0` | `8,192` | `-316.39` (`0.0%`, `5.62s`) | `-316.66` (`0.0%`, `6.51s`) |
| `myoHandPoseRandom-v0` | `16,384` | `-316.15` (`0.0%`, `11.27s`) | `-315.40` (`0.0%`, `12.85s`) |
| `myoHandPoseRandom-v0` | `24,576` | `-316.14` (`0.0%`, `16.84s`) | `-316.25` (`0.0%`, `19.33s`) |
| `myoHandPoseRandom-v0` | `32,768` | `-316.08` (`0.0%`, `22.36s`) | `-316.06` (`0.0%`, `25.81s`) |

### Non-Markovian Muscle Fatigue Ablation (`--fatigue`, `FatigueWrapper`, reported separately)

| Environment (`+ FatigueWrapper`) | Method | Env Steps | Final Return (mean ± std) | Final Success (mean ± std) | Train Wall-Clock (s) | Peak Memory (MB) |
|---|---|---:|---:|---:|---:|---:|
| `myoElbowPose1D6MRandom-v0` | PPO (feedforward) | 32,768 | `723.77 ± 44.72` | `96.7% ± 4.7%` | `20.68 ± 0.38` | `68.1` |
| `myoElbowPose1D6MRandom-v0` | $\text{S}^2\text{PG}$-PPO (`gru`) | 32,768 | `155.54 ± 114.11` | `18.3% ± 13.1%` | `23.85 ± 0.51` | `0.9` |
| `myoHandPoseRandom-v0` | PPO (feedforward) | 32,768 | `-318.41 ± 3.13` | `0.0% ± 0.0%` | `30.60 ± 0.36` | `4.4` |
| `myoHandPoseRandom-v0` | $\text{S}^2\text{PG}$-PPO (`gru`) | 32,768 | `-317.97 ± 3.08` | `0.0% ± 0.0%` | `33.83 ± 0.36` | `1.0` |

## Tests

```bash
pytest training/s2pg/tests/test_s2pg.py -v
```
