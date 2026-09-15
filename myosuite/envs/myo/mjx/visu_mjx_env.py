import time
import mujoco
import jax
import mujoco.viewer
from brax.io import model
from myosuite.envs.myo.mjx import make
import argparse
import os

from train_jax_ppo import load_env_and_network_factory
from brax.training.acme.running_statistics import normalize
# Visualize an MJX environment interactively

local_path = os.path.dirname(os.path.realpath(__file__))

def main(env_name, load_policy, use_reset):
    env, _, network_factory = load_env_and_network_factory(env_name, 'jax')

    # If we don't load a trained policy, we will transmit actions from the user inputs in the simulate window.
    ppo_network = None
    policy_fn = None
    if load_policy:
        ppo_network = network_factory(
            env.observation_size,
            env.action_size)
        params = model.load_params(local_path+"/"+"playground_params.pickle")

        def deterministic_policy(input_data):
            logits = ppo_network.policy_network.apply(*params[:2], input_data)
            brax_result = ppo_network.parametric_action_distribution.mode(logits)
            return brax_result

        policy_fn = jax.jit(deterministic_policy)

    # We could get the model from the env, but we want to make some edits for convenience
    spec = mujoco.MjSpec.from_file(env.xml_path)
    spec = env.preprocess_spec(spec)
    # Add in dummy sensor we can write to later to visualize values
    spec.add_sensor(name="reward+target", type=mujoco.mjtSensor.mjSENS_USER, dim=1+env.mjx_model.nq)
    m = spec.compile()

    d = mujoco.MjData(m)

    # Note: the first two steps will be performed much slower as the function is being jitted.
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    key =jax.random.PRNGKey(0)
    state = jit_reset(key)

    # We'll use a parallel CPU mujoco instance in sync with the MJX env
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()
            actions = policy_fn(state.obs) if load_policy else d.ctrl
            state = state.replace(data=state.data.replace(mocap_pos=d.mocap_pos, xfrc_applied=d.xfrc_applied))
            state = jit_step(state, actions)

            # We'll use the sensordata array to visualize key values as real-time bar-graphs (use F4 in viewer)
            d.sensordata[0] = state.reward
            d.sensordata[1:] = state.info['target_angles'] if "target_angles" in state.info else state.data.qpos

            if load_policy:
                d.ctrl = state.data.ctrl  # If we are using the policy, we'll visualize the excitation.

            if state.done and use_reset:
                _, key = jax.random.split(key)
                state = jit_reset(key)
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            d.qpos = state.data.qpos
            mujoco.mj_forward(m, d)  # We only do forward step for the kinematics, no need to integrate.

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train PPO agent with Brax")
    parser.add_argument(
        "--env_name",
        type=str,
        default="MjxFingerPoseRandom-v0",
    )

    parser.add_argument(
        "--load_policy",
        action="store_true",
    )

    parser.add_argument(
        "--use_reset",
        action="store_true",
    )

    args = parser.parse_args()
    main(args.env_name, args.load_policy, args.use_reset)
