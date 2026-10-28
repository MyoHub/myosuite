# %% [markdown]
# # MyoHalloween: a muscle-driven avoidance experiment
#
# Run this file with Python after installing requirements-student.txt, or run
# the matching Colab cells from top to bottom. All experiment code is here.
# You will build a body and a physical pumpkin, select an avoidance command,
# let TERRA generate 354 muscle controls, measure contacts, and save a video.
#
# Three different things are easy to confuse:
# - TERRA is a pretrained muscle controller. We load it; we do not train PPO.
# - A small learned motion decoder supplies reference poses for four commands.
# - A learned selector scores those commands from the visible pumpkin state.
# Recovery timing is scripted. This lesson covers upper-body throws, not jumps.

# %% [markdown]
# ## 1. Imports and experiment settings
#
# Physics runs every 0.002 s (500 Hz). TERRA chooses controls every five
# physics steps (100 Hz). The video keeps every fourth control frame (25 fps).
# X is lateral. Pumpkins travel from negative Y toward the body.
# Change OUTPUT to keep several runs. Later exercises change COMMAND_OVERRIDE.

# %%
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LP_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from dataclasses import replace
from pathlib import Path
from urllib.request import urlretrieve
import hashlib
import json

import av
import joblib
import mujoco
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from myosuite.core.trajectory_io import motion_clip_from_states
from myosuite.integrations.musclemimic import (
    MIMIC_SITES,
    TerrainController,
    build_terrain_fullbody_spec,
    download_terrain_checkpoint,
    load_terrain_policy,
)

OUTPUT = Path("myohalloween_output")
OUTPUT.mkdir(exist_ok=True)
SIM_DT = 0.002
SUBSTEPS = 5
CONTROL_DT = SIM_DT * SUBSTEPS
COMMANDS = ("idle", "left", "right", "squat")
COMMAND_OVERRIDE = None  # Exercise: use "idle" to test standing still.
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
RADIUS = 0.065  # metres; keep the trained curriculum unchanged for the first run.

# %% [markdown]
# ## 2. Download the two small learned data files
#
# These are weights, not Python programs. The NPZ stores motion decoder
# coefficients and a settled starting state. The joblib file stores the trained
# scikit-learn command-value model. Their hashes pin the exact teaching example.
# Only load the trusted, hash-verified joblib file below: it is a pickle format.
# The much larger TERRA checkpoint is downloaded separately in section 5.

# %%
ASSET_REVISION = "8842a86052d46be13b489a1e755bd265bcbc78ae"
ASSET_URL = (
    "https://raw.githubusercontent.com/MyoHub/myosuite/"
    f"{ASSET_REVISION}/tutorials/files/halloween/"
)
HASHES = {
    "motion_decoder.npz": "b3f9a27ed5a9b3dd64f0efb8671eb5954fe1c0ce1ed681e3e5ba339d2468b78c",
    "selector.joblib": "c748be2fa397cd8aefb70bf7fa6eda20eb5e5ecefb49963245531cb41097db8e",
}
for filename, expected_hash in HASHES.items():
    destination = OUTPUT / filename
    if not destination.exists():
        urlretrieve(ASSET_URL + filename, destination)
    actual_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
    assert actual_hash == expected_hash, f"Unexpected file contents: {destination}"

weights = np.load(OUTPUT / "motion_decoder.npz", allow_pickle=False)
selector = joblib.load(OUTPUT / "selector.joblib")
print("Commands:", weights["commands"].tolist())
print("Frames per command:", {name: int(weights[name + "_frames"]) for name in COMMANDS})

# %% [markdown]
# ## 3. Build the model and add real collision geometry
#
# Use MyoSuite's calibrated TERRA body unchanged. Our MjSpec callback appends
# one free pumpkin. Its sphere has mass and collides; lobes and stem only draw
# its shape. Explicit pairs include the body's collision geoms whose collision
# masks are disabled because the original model uses selected contact pairs.
# The actor-only copy supplies TERRA observations; the scene advances physics.

# %%
def add_pumpkin(spec):
    """Append one free projectile without changing the actor's joints/muscles."""
    paired = {name for pair in spec.pairs for name in (pair.geomname1, pair.geomname2)}
    targets = [g.name for g in spec.geoms if g.name and (
        g.name in paired or g.contype or g.conaffinity
    )]
    body = spec.worldbody.add_body(name="pumpkin", pos=[8, 0, 0.5], gravcomp=1)
    body.add_freejoint(name="pumpkin_joint")
    body.add_geom(name="pumpkin_collision", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                  size=[RADIUS, 0, 0], mass=0.25, rgba=[1, 0.28, 0.015, 1],
                  contype=1, conaffinity=1, friction=[0.7, 0.01, 0.001])
    for j in range(8):
        angle = j * np.pi / 4
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                      pos=[0.28 * RADIUS * np.cos(angle), 0.28 * RADIUS * np.sin(angle), 0],
                      size=[0.78 * RADIUS, 0.78 * RADIUS, 0.92 * RADIUS],
                      mass=0, contype=0, conaffinity=0, rgba=[0.9, 0.19, 0.008, 1])
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, pos=[0, 0, RADIUS + 0.005],
                  size=[0.009, 0.018, 0], mass=0, contype=0, conaffinity=0,
                  rgba=[0.18, 0.28, 0.05, 1])
    for name in targets:
        spec.add_pair(geomname1="pumpkin_collision", geomname2=name)


actor = build_terrain_fullbody_spec().compile()
actor_data = mujoco.MjData(actor)
model = build_terrain_fullbody_spec(add_pumpkin).compile()
data = mujoco.MjData(model)
assert model.nu == 354 and model.opt.timestep == SIM_DT
print(f"Scene: {model.nq} positions, {model.nv} velocities, {model.nu} muscle controls")

pumpkin_body = model.body("pumpkin").id
pumpkin_geom = model.geom("pumpkin_collision").id
qa = int(model.joint("pumpkin_joint").qposadr[0])
va = int(model.joint("pumpkin_joint").dofadr[0])
paired_ids = set(map(int, model.pair_geom1)) | set(map(int, model.pair_geom2))
body_geoms = {
    i for i in range(model.ngeom)
    if model.geom_bodyid[i] not in (0, pumpkin_body)
    and (i in paired_ids or model.geom_contype[i] or model.geom_conaffinity[i])
}

# %% [markdown]
# ## 4. Turn a command into a reference trajectory
#
# This is the entire motion-decoder calculation: evaluate radial basis
# functions at each phase, multiply by learned coefficients, add a mean pose.
# MyoSuite then computes reference site positions/velocities from those poses.
# Translating the reference by an anchor lets a left/right step start where
# the previous step ended. Reference poses are targets, never actor state writes.
# Holding the exact final reference avoids a discontinuity during recovery.

# %%
reference_cache = {}


def command_reference(name, anchor):
    """Decode reference joint poses, then translate their root/site targets."""
    if name not in reference_cache:
        frames = int(weights[name + "_frames"])
        phase = np.linspace(0, 1, frames)[:, None]
        basis = rbf_kernel(phase, weights[name + "_centers"], gamma=float(weights["gamma"]))
        qpos = basis @ weights[name + "_coef"] + weights[name + "_base"]
        reference_cache[name] = motion_clip_from_states(actor, qpos, CONTROL_DT, MIMIC_SITES)
    clip = reference_cache[name]
    shift = np.r_[anchor, 0.0]
    positions = clip.qpos.copy()
    positions[:, :3] += shift
    return replace(clip, qpos=positions, site_xpos=clip.site_xpos + shift)


def hold_last_reference(clip):
    """Make an 11-second stationary reference at the last target pose."""
    return replace(clip, qpos=np.repeat(clip.qpos[-1:], 1101, axis=0),
                   qvel=np.zeros((1101, actor.nv)),
                   site_xpos=np.repeat(clip.site_xpos[-1:], 1101, axis=0),
                   site_xmat=np.repeat(clip.site_xmat[-1:], 1101, axis=0),
                   site_velocity=np.zeros((1101, *clip.site_velocity.shape[1:])))

# %% [markdown]
# ## 5. Initialize once and load the frozen muscle policy
#
# Start from the decoder's settled state, which includes muscle activations.
# An arbitrary upright pose with zero activation is not a balanced initial state.
# After this one reset, only muscle controls move the actor during simulation.
# TerrainController is the existing MyoSuite runner, including its observation
# timing correction; this tutorial does not reimplement neural-network inference.

# %%
rng = np.random.default_rng(2027)
policy_seed = int(rng.integers(2**31))
first_wait_steps = int(rng.integers(40, 81))
data.qpos[:actor.nq] = weights["initial_qpos"]
data.qvel[:actor.nv] = weights["initial_qvel"]
data.act[:] = weights["initial_act"]
mujoco.mj_forward(model, data)
home = data.qpos[:2].copy()
anchor = home.copy()
command = "idle"
reference = command_reference(command, anchor)
checkpoint = download_terrain_checkpoint()
policy = load_terrain_policy(checkpoint, actor, reference, stochastic=True, seed=policy_seed)
tracker = TerrainController(policy, actor, reference, SUBSTEPS)
tracker.reset()
reference_frame = 0
print("Loaded TERRA:", checkpoint)

# %% [markdown]
# ## 6. Define three reproducible throws
#
# These are the held-out example's physical parameters, not action labels.
# At launch the selector sees position and velocity and estimates the intercept.
# A projectile starts speed*flight_s metres in front of the target. Giving it
# vertical velocity g*flight_s/2 returns it to its starting height at arrival:
# z(t) = z(0) + vz*t - g*t**2/2. Every hit is checked against MuJoCo contacts.

# %%
throws = [
    dict(target_x=-0.11537105180914775, target_z=1.4608824024981797,
         flight_s=1.2563822360825192, speed=8.0),
    dict(target_x=0.13093328164781803, target_z=1.4949661043026456,
         flight_s=1.2074422533634466, speed=8.0),
    dict(target_x=0.02295215779072947, target_z=1.7203338380650184,
         flight_s=1.0910301086419278, speed=12.0),
]
trace = []
results = []

# %% [markdown]
# ## 7. Run the experiment: launch, choose, actuate, measure
#
# Read this loop in order. A new command changes a reference, not the body pose.
# Each call to tracker returns muscle controls. Five mj_step calls advance real
# physics; contacts are counted after each substep so a brief impact is retained.
# After a lateral action we request the opposite action 3.7 s after launch.
# The actor then has until 7.5 s after launch to recover. These timings reproduce
# the existing teaching baseline; they are not an online planning algorithm.

# %%
for throw_index, threat in enumerate(throws):
    wait_steps = first_wait_steps if throw_index == 0 else 50
    contacts = 0
    clearance = 10.0
    max_offset = 0.0
    min_height = float(data.qpos[2])
    selected = "idle"

    for step in range(wait_steps + 750):
        next_command = None
        if step == wait_steps:
            # Launch the free pumpkin. Only the projectile state is prescribed.
            t, speed = threat["flight_s"], threat["speed"]
            model.body_gravcomp[pumpkin_body] = 0
            data.qpos[qa:qa + 3] = [home[0] + threat["target_x"],
                                    home[1] + 0.14 - speed * t, threat["target_z"]]
            data.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
            data.qvel[va:va + 6] = [0, speed, 0.5 * 9.81 * t, 0, 2, 0]
            mujoco.mj_forward(model, data)

            # Predict value for each command from the estimated intercept.
            relative = (data.qpos[qa:qa + 3] - data.qpos[:3]).astype(np.float32)
            velocity = data.qvel[va:va + 3].astype(np.float32)
            intercept_time = -relative[1] / velocity[1]
            features = np.array([
                relative[0] + velocity[0] * intercept_time,
                relative[2] + np.float32(data.qpos[2]) + velocity[2] * intercept_time
                - 0.5 * 9.81 * intercept_time**2,
                intercept_time, 1.0,
            ], dtype=np.float32)
            candidates = np.c_[np.tile(features, (4, 1)), np.eye(4)]
            values = selector.predict(candidates)
            selected = COMMAND_OVERRIDE or COMMANDS[int(np.argmax(values))]
            next_command = selected
            print(f"Throw {throw_index + 1}: values={values.round(2)}, command={selected}")
        elif step == wait_steps + 370 and selected in ("left", "right"):
            next_command = "right" if selected == "left" else "left"

        if next_command is not None and next_command != "idle":
            assert command == "idle", "Wait for the current reference to finish."
            command = next_command
            reference = command_reference(command, anchor)
            policy.obs_adapter.reference = reference
            tracker = TerrainController(policy, actor, reference, SUBSTEPS)
            tracker.reset()
            reference_frame = 0

        # Supply the current physical actor state to the existing TERRA runner.
        actor_data.qpos[:] = data.qpos[:actor.nq]
        actor_data.qvel[:] = data.qvel[:actor.nv]
        actor_data.act[:] = data.act
        actor_data.time = data.time
        mujoco.mj_forward(actor, actor_data)
        controls = tracker(actor_data)
        reference_frame += 1
        if reference_frame >= reference.num_frames - 1:
            anchor = reference.qpos[-1, :2].copy()
            reference = hold_last_reference(reference)
            command = "idle"
            policy.obs_adapter.reference = reference
            tracker = TerrainController(policy, actor, reference, SUBSTEPS)
            tracker.reset()
            reference_frame = 0

        data.ctrl[:] = np.clip(controls, *model.actuator_ctrlrange.T)
        for substep in range(SUBSTEPS):
            mujoco.mj_step(model, data)
            if step >= wait_steps:
                for contact in data.contact:
                    if ((contact.geom1 == pumpkin_geom and int(contact.geom2) in body_geoms)
                            or (contact.geom2 == pumpkin_geom and int(contact.geom1) in body_geoms)):
                        contacts += 1
                        clearance = min(clearance, float(contact.dist))
        mujoco.mj_forward(model, data)
        trace.append((data.time, data.qpos.copy(), data.qvel.copy(), data.act.copy()))
        max_offset = max(max_offset, float(np.linalg.norm(data.qpos[:2] - home)))
        min_height = min(min_height, float(data.qpos[2]))
        if step >= wait_steps and np.linalg.norm(data.qpos[qa:qa + 3] - data.qpos[:3]) < 1:
            for geom in body_geoms:
                distance = mujoco.mj_geomDistance(model, data, pumpkin_geom, geom, 10.0, np.zeros(6))
                clearance = min(clearance, float(distance))

    head = data.body("head").xpos
    direction = head - data.qpos[:3]
    posture = float(direction[2] / np.linalg.norm(direction))
    final_offset = float(np.linalg.norm(data.qpos[:2] - home))
    recovered = final_offset < 0.10 and data.qpos[2] > 0.85 and head[2] > 1.45 and posture > 0.9
    results.append(dict(throw=throw_index + 1, command=selected, body_contacts=contacts,
                        min_surface_distance_m=clearance, min_pelvis_height_m=min_height,
                        max_home_distance_m=max_offset, final_home_distance_m=final_offset,
                        recovered=bool(recovered)))

# %% [markdown]
# ## 8. Save and inspect quantitative checks
#
# Collision counts are measured at 500 Hz. Surface distance is sampled near the
# body at 100 Hz; it is useful context, not a continuous collision guarantee.
# The first unmodified run should avoid all three throws, remain upright and
# finish each recovery within 10 cm of home. Do not replace a failed check with
# a hardcoded success value when experimenting.

# %%
report = dict(architecture="learned command selector + frozen TERRA muscle control",
              seed=2027, actor_resets=1, events=results)
(OUTPUT / "checks.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
time, qpos, qvel, activation = map(np.asarray, zip(*trace))
np.savez_compressed(OUTPUT / "rollout.npz", time=time, qpos=qpos, qvel=qvel, act=activation)
if COMMAND_OVERRIDE is None:
    assert all(r["body_contacts"] == 0 and r["recovered"] for r in results), "Inspect checks.json"
    assert max(r["max_home_distance_m"] for r in results) > 0.2, "No meaningful lateral dodge"
    assert min(r["min_pelvis_height_m"] for r in results) > 0.65, "Actor fell"

# %% [markdown]
# ## 9. Render the recorded physical rollout
#
# Physics has already finished. Setting render_data to each saved state only
# draws the recording; it does not generate or prescribe the simulated motion.
# The timestamps determine the playback speed: every fourth 100 Hz state at
# 25 fps is real time. Use a fixed camera so lateral movement stays visible.

# %%
model.vis.global_.offwidth = VIDEO_WIDTH
model.vis.global_.offheight = VIDEO_HEIGHT
model.vis.quality.numslices = 8  # Drawing detail only; physics geometry is unchanged.
model.vis.quality.numstacks = 4
render_data = mujoco.MjData(model)
camera = mujoco.MjvCamera()
camera.lookat[:] = [0, -0.6, 1.35]
camera.distance = 4.8
camera.azimuth = 115
camera.elevation = -10
options = mujoco.MjvOption()
options.geomgroup[:] = 0
options.geomgroup[:3] = 1
options.sitegroup[:] = 0
video_path = OUTPUT / "avoidance.mp4"
with mujoco.Renderer(model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH) as renderer:
    with av.open(str(video_path), mode="w") as video:
        stream = video.add_stream("libx264", rate=25)
        stream.width, stream.height = VIDEO_WIDTH, VIDEO_HEIGHT
        stream.pix_fmt = "yuv420p"
        stream.codec_context.thread_count = 1
        for frame in range(0, len(time), 4):
            if frame % 200 == 0:
                print(f"Rendering state {frame}/{len(time)}", flush=True)
            render_data.qpos[:] = qpos[frame]
            render_data.qvel[:] = qvel[frame]
            render_data.act[:] = activation[frame]
            mujoco.mj_forward(model, render_data)
            renderer.update_scene(render_data, camera=camera, scene_option=options)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            image = av.VideoFrame.from_ndarray(renderer.render(), format="rgb24")
            for packet in stream.encode(image):
                video.mux(packet)
        for packet in stream.encode():  # Flush the encoder's final frames.
            video.mux(packet)
print("Video:", video_path.resolve())

# %% [markdown]
# ## Try it yourself
#
# 1. Set COMMAND_OVERRIDE = "idle" and rerun. Do the same pumpkins actually hit?
# 2. Restore None; move one target_x by 0.02 m and compare command values and clearance.
# 3. Change camera.distance only. Metrics should stay identical: rendering is separate.
# 4. Inspect a left reference's root X coordinates and compare with physical root X.
#
# This example does not prove arbitrary-threat robustness, jumping, or that
# TERRA plans routes online. The separate raw-muscle myoHalloween-v0 benchmark
# has a different action/observation contract. No browser or training code is run.
