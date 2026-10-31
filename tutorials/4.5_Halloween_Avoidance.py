# %% [markdown]
# # MyoHalloween: load an environment, control muscles, inspect the result
#
# This lesson uses `myoHalloween-v0` with its explicit **TERRA showcase profile**.
# The environment owns the body, pumpkin, launch schedule, collisions and metrics.
# A separate pretrained controller selects left/right/squat reference motions and
# uses TERRA to produce 354 muscle controls. We run inference, not new training.
#
# Run the cells in order in Colab, or install `requirements-student.txt` in a fresh
# Python 3.12 environment and run this file. A GPU is optional. The first run
# downloads the public TERRA checkpoint; later runs reuse its cache.
#
# [Task and API documentation](https://github.com/MyoHub/myosuite/blob/feature/myohalloween-task/docs/halloween.md)
# · [Try the browser demo](https://spooky.myosuite.org/)

# %% [markdown]
# ## Hugging Face account and `HF_TOKEN` (recommended)
#
# The [TERRA-4B checkpoint](https://huggingface.co/merc-s/TERRA-4B) is public: you can run this lesson without an account or token. Signing in is recommended for higher download limits, especially on shared Colab connections.
#
# 1. [Create a free Hugging Face account](https://huggingface.co/join) and sign in.
# 2. Open [Access Tokens](https://huggingface.co/settings/tokens), select **Create new token**, and create a **Read** token. Write/admin permissions are unnecessary.
# 3. In Colab, click the **key icon (Secrets)** in the left sidebar. Add a secret named exactly **`HF_TOKEN`**, paste the token into its value, and enable **Notebook access** for this notebook.
# 4. Do this **before running the download cells**. The Hugging Face Hub library reads the Colab secret automatically. If you already attempted a download before adding/enabling the secret, restart the runtime and rerun the cells in order.
#
# Keep the token in Secrets; do not paste it into a code cell or print it. A missing/disabled secret still permits anonymous access to the public checkpoint; if you get a rate-limit error, enable the secret and retry after the limit resets. If a token is rejected, replace it with a valid Read token or disable/remove the invalid secret and restart for anonymous access.
#
# For local Python, set the **`HF_TOKEN` environment variable before starting Python**, or run `hf auth login` once. See the [official token guide](https://huggingface.co/docs/hub/security-tokens) and [download limits](https://huggingface.co/docs/hub/rate-limits). This lesson does not require access to gated motion datasets.

# %% [markdown]
# ## 1. Imports and settings
#
# Set the graphics backend before importing MuJoCo. EGL is the verified Colab/Linux
# path. Local desktop users can select their graphics backend before running Python.
# The simulation runs at 500 Hz, muscle controls at 100 Hz, and video at 25 fps.

# %%
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LP_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from pathlib import Path
import json

import numpy as np
from myosuite import make_env
from myosuite.utils.video_io import write_video
from myosuite.integrations.musclemimic import download_terrain_checkpoint
from myosuite.integrations.musclemimic.halloween import (
    COMMANDS,
    HalloweenController,
    download_halloween_assets,
)

OUTPUT = Path("myohalloween_output")
OUTPUT.mkdir(exist_ok=True)
SEED = 2027
COMMAND_OVERRIDE = None  # Try "idle", "left", "right", or "squat" later.
VIDEO_FPS = 25

# %% [markdown]
# ## 2. Load the registered task
#
# `profile="terra"` selects the calibrated model and three frontal throws used by
# the working demo. X is lateral; pumpkins approach along positive Y.
# Both profiles accept **354 muscle excitations** and return **535 observations**.
# The default profile is the separate six-second CPU/MJLab training benchmark.
# This 24-second showcase needs a larger episode limit and runs on CPU.
#
# `reset(seed=...)` restores a settled pose with active muscles, resets all task
# counters, and determines the launch timing and muscle-policy noise seed.

# %%
env = make_env(
    "myoHalloween-v0",
    profile="terra",
    max_episode_steps=3000,
    render_mode="rgb_array",
)
obs, info = env.reset(seed=SEED)
task = env.unwrapped
print("Observations:", obs.shape, "Muscle controls:", env.action_space.shape)
print("Control interval:", task.dt, "seconds")

# %% [markdown]
# ## 3. Load the pretrained controller
#
# The two small learned files are downloaded openly and verified by SHA-256.
# Their decoder creates reference poses; their selector scores four commands from
# the observed projectile intercept. The frozen TERRA checkpoint tracks those
# references through muscle controls. Reference poses never overwrite the actor.
#
# Call `controller.reset(info)` after each environment reset. This uses the seed
# supplied by the task so repeating the same reset reproduces policy noise too.
# Recovery after a side-step is scripted at 3.7 seconds after launch.

# %%
assets = download_halloween_assets()
checkpoint = download_terrain_checkpoint()
controller = HalloweenController(task, checkpoint, assets, command_override=COMMAND_OVERRIDE)
controller.reset(info)
print("Loaded TERRA:", checkpoint)
print("Available commands:", COMMANDS)

# %% [markdown]
# ## 4. Step physics and record video
#
# The loop has three operations: compute muscle actions, step the environment,
# and render every fourth control state. `info["halloween"]` reports launches and
# completed-throw metrics. A launch appears in the observation before the
# controller reacts, so avoidance cannot begin before that physical launch.
#
# `terminated` means a hit, fall, or completed curriculum; `truncated` means the
# external episode time limit was reached. We inspect the metrics in section 5.
# Collect frames, then use MyoSuite's video helper to save this short clip.
# At 640 × 480, the example keeps about 550 MB of frames in memory.

# %%
video_path = OUTPUT / "avoidance.mp4"
root_positions = []
timestamps = []
frames = []
control_steps = round(1 / (VIDEO_FPS * task.dt))
last_launch = 0
try:
    for step in range(3000):
        muscle_actions = controller(task.data, info)
        obs, reward, terminated, truncated, info = env.step(muscle_actions)
        timestamps.append(task.data.time)
        root_positions.append(task.data.qpos[:3].copy())
        if info["halloween"]["launch_id"] != last_launch:
            last_launch = info["halloween"]["launch_id"]
            print(f"Pumpkin {last_launch} launched at {task.data.time:.2f} s")
        if step % control_steps == 0:
            frames.append(env.render())
        if terminated or truncated:
            break
finally:
    env.close()
write_video(video_path, frames, fps=VIDEO_FPS)
del frames
print("Saved video:", video_path.resolve())

# %% [markdown]
# ## 5. Check the physical outcome
#
# Contact rows are checked at every 2 ms physics substep. Surface distance is
# sampled near the body at the 100 Hz control rate, so it is not a continuous
# collision guarantee. Recovery means finishing within 10 cm of home with an
# upright pelvis/head. The baseline should avoid all three throws and move
# laterally by about 0.32 m. A hit/fall must fail these checks.
#
# The selector can choose different successful commands. This checkpoint's
# unmodified example selects left, right, right; it does not demonstrate a jump.

# %%
report = {
    "env_id": "myoHalloween-v0",
    "profile": "terra",
    "seed": SEED,
    "events": info["halloween"]["events"],
    "decisions": controller.decisions,
    "current_event": info["halloween"]["current_event"],
    "hit": bool(info["rwd_dict"]["hit"]),
    "fallen": bool(info["rwd_dict"]["fallen"]),
    "terminated": terminated,
    "truncated": truncated,
}
(OUTPUT / "checks.json").write_text(json.dumps(report, indent=2) + "\n")
np.savez_compressed(OUTPUT / "root_motion.npz", time=timestamps, root=root_positions)
print(json.dumps(report, indent=2))
if COMMAND_OVERRIDE is None:
    assert info["halloween"]["complete"] and not truncated, "The curriculum did not finish"
    assert not info["rwd_dict"]["hit"] and not info["rwd_dict"]["fallen"], "Actor hit or fell"
    assert all(e["body_contacts"] == 0 and e["recovered"] for e in report["events"])
    assert max(e["max_home_distance_m"] for e in report["events"]) > 0.2

# %% [markdown]
# ## 6. Watch and experiment
#
# The MP4 plays the physical states sampled during section 4, at real-time speed.
# In Colab the next cell embeds it. Running the Python script saves it on disk.
#
# 1. Set `COMMAND_OVERRIDE = "idle"`; rerun from section 2. Does the actor get hit?
# 2. Restore `None`; repeat seed 2027 and compare the saved root trajectory.
# 3. Inspect `controller.decisions`: which command won for each intercept?
# 4. To change throws, pass an `AvoidanceConfig` to `make_env` as
#    `avoidance_config=...`; see the task documentation for a small example.
#
# Recreate the environment after closing it. The controller and task are reusable
# library components; their implementation is linked in the task documentation.
# Robustness to arbitrary threats, low throws, and jumping remain outside this
# curriculum. GPU benchmark training uses the default profile and its own policy.
