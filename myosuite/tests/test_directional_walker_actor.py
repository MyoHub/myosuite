# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The ChaseTag actor that starts as the tutorial 5.4 walker (546 -> 528 adapter + 5.4 network)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

from tensordict import TensorDict  # noqa: E402

from myosuite import make_env  # noqa: E402
from myosuite.envs.myo.tasks.mimic.policy import ActorCritic  # noqa: E402
from myosuite.integrations.musclemimic.directional_walker_actor import (  # noqa: E402
    DirectionalWalkerActor,
    FbvsDirectionalAdapter,
)
from myosuite.tests.support.optional_deps import (  # noqa: E402
    require_musclemimic_models,
)

pytestmark = pytest.mark.tier2

ENV_ID = "myoChallengeChaseTagFBVsTrain-v0"
CLASS = (
    "myosuite.integrations.musclemimic.directional_walker_actor:DirectionalWalkerActor"
)
N_ACT = 354


def _walker_checkpoint():
    from myosuite.envs.myo.backends.mjlab.tasks.chasetag_vs.rl_cfg import (  # noqa: PLC0415
        WALKER_CHECKPOINT,
    )

    if not WALKER_CHECKPOINT.exists():
        pytest.skip("tutorial 5.4 walker checkpoint not in this checkout")
    return WALKER_CHECKPOINT


def _fbvs_observations(n_steps: int = 30) -> np.ndarray:
    """Train-env observations of both roles (learner and opponent views)."""
    require_musclemimic_models()
    env = make_env(ENV_ID, opponent_policy=lambda o: np.full(N_ACT, 0.1))
    rows = []
    try:
        for seed in (0, 1, 2):
            obs, _ = env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            for _ in range(n_steps):
                rows += [obs, env.unwrapped.opponent_obs.copy()]
                obs, _, term, _, _ = env.step(rng.uniform(0.0, 0.3, N_ACT))
                if term:
                    break
    finally:
        env.close()
    return np.stack(rows).astype(np.float32)


def _actor(checkpoint: str) -> DirectionalWalkerActor:
    obs = TensorDict({"policy": torch.zeros(2, 546)}, batch_size=[2])
    dist = {"class_name": "GaussianDistribution", "init_std": 0.1, "std_type": "scalar"}
    return DirectionalWalkerActor(
        obs,
        {"actor": ["policy"]},
        "actor",
        N_ACT,
        distribution_cfg=dist,
        walker_checkpoint=checkpoint,
    ).eval()


def test_adapter_is_the_chasetag_heading_directional_obs() -> None:
    """Same 528 numbers as ``chasetag_heading_directional_obs`` on the player's own state."""
    import mujoco  # noqa: PLC0415

    from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_config import (  # noqa: PLC0415
        ChaseTagVsConfig,
    )
    from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_model import (  # noqa: PLC0415
        _build_fullbody_agent_spec,
    )
    from myosuite.envs.myo.tasks.challenge.chase_tag_vs.train_env import (  # noqa: PLC0415
        LEARNER,
        OPPONENT,
    )
    from myosuite.envs.myo.tasks.mimic.chasetag_obs import (  # noqa: PLC0415
        chasetag_heading_directional_obs,
    )

    require_musclemimic_models()
    single = _build_fullbody_agent_spec(ChaseTagVsConfig())[0].compile()
    data = mujoco.MjData(single)
    adapter = FbvsDirectionalAdapter()
    env = make_env(ENV_ID, opponent_policy=lambda o: np.full(N_ACT, 0.1))
    u = env.unwrapped
    try:
        roles = set()
        for seed in (0, 1, 2, 3):
            obs, info = env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            for _ in range(20):
                for agent, view in ((LEARNER, obs), (OPPONENT, u.opponent_obs)):
                    qs, vs = u.player_slices(agent)
                    other, _ = u.player_slices(
                        OPPONENT if agent == LEARNER else LEARNER
                    )
                    data.qpos[:], data.qvel[:] = u.data.qpos[qs], u.data.qvel[vs]
                    data.act[:] = u.data.act[u._act[agent]]
                    evade = (info["task"] == "EVADE") == (agent == LEARNER)
                    roles.add(evade)
                    expected = chasetag_heading_directional_obs(
                        single, data, u.data.qpos[other][:2], evade=evade
                    )
                    got = adapter(torch.as_tensor(view)[None])[0].numpy()
                    np.testing.assert_allclose(got, expected, atol=1e-5)
                obs, _, term, _, info = env.step(rng.uniform(0.0, 0.3, N_ACT))
                if term:
                    break
        assert roles == {True, False}
    finally:
        env.close()


def test_actor_starts_as_the_walker_and_its_exports_match() -> None:
    """Deterministic output = 5.4 walker(adapter(obs)); the JIT and ONNX copies agree."""
    ckpt = _walker_checkpoint()
    actor = _actor(str(ckpt))
    walker = ActorCritic.load(ckpt, obs_dim=528)
    obs = torch.as_tensor(_fbvs_observations())
    td = TensorDict({"policy": obs}, batch_size=[obs.shape[0]])
    with torch.no_grad():
        expected = walker.actor_mean(walker.trunk(FbvsDirectionalAdapter()(obs)))
        torch.testing.assert_close(actor(td), expected, atol=1e-5, rtol=0)
        torch.testing.assert_close(actor.as_jit()(obs), expected, atol=1e-5, rtol=0)
        torch.testing.assert_close(
            actor.as_onnx(verbose=False)(obs), expected, atol=1e-5, rtol=0
        )
    assert actor.distribution is not None  # PPO's Gaussian noise on top


def test_checkpoint_loads_and_exports_as_a_546_to_354_player(tmp_path) -> None:
    """``load_rslrl_policy`` and ``export_rslrl_to_onnx`` rebuild the adapter + network."""
    import onnxruntime as ort  # noqa: PLC0415
    import yaml  # noqa: PLC0415

    from myosuite.utils.export_onnx import export_rslrl_to_onnx  # noqa: PLC0415
    from myosuite.utils.rslrl_policy import load_rslrl_policy  # noqa: PLC0415

    actor = _actor(str(_walker_checkpoint()))
    run = tmp_path / "run"
    (run / "params").mkdir(parents=True)
    with (run / "params" / "agent.yaml").open("w") as f:
        yaml.safe_dump({"actor": {"class_name": CLASS, "hidden": 512, "layers": 6}}, f)
    torch.save({"actor_state_dict": actor.state_dict()}, run / "model_0.pt")

    policy = load_rslrl_policy(run / "model_0.pt", N_ACT)
    assert policy.obs_dim == 546
    obs = torch.as_tensor(_fbvs_observations(10))
    td = TensorDict({"policy": obs}, batch_size=[obs.shape[0]])
    with torch.no_grad():
        expected = actor(td)
        torch.testing.assert_close(policy(obs), expected, atol=1e-5, rtol=0)

    onnx_path = tmp_path / "player.onnx"
    export_rslrl_to_onnx(run / "model_0.pt", onnx_path, obs_dim=546, act_dim=N_ACT)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = session.run(None, {session.get_inputs()[0].name: obs.numpy()})[0]
    np.testing.assert_allclose(got, expected.numpy(), atol=1e-4)
