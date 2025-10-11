import time
import mujoco
import jax
import mujoco.viewer
from myosuite.envs.myo.mjx import make
# Visualize an MJX environment interactively


def main():
    env = make("MjxElbowPoseFixed-v0")

    # We could get the model from the env, but we want to make some edits for convenience
    spec = mujoco.MjSpec.from_file(env.xml_path)
    spec = env.preprocess_spec(spec)
    # Add in dummy sensor we can write to later to visualize values
    spec.add_sensor(name="reward+target", type=mujoco.mjtSensor.mjSENS_USER, dim=1+env.mjx_model.nv)
    m = spec.compile()

    d = mujoco.MjData(m)

    # Note: the first two steps will be performed much slower as the function is being jitted.
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    state = jit_reset(jax.random.PRNGKey(0))

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()

            state = state.replace(data=state.data.replace(mocap_pos=d.mocap_pos, xfrc_applied=d.xfrc_applied))
            state = jit_step(state, d.ctrl)

            d.qpos = state.data.qpos
            mujoco.mj_forward(m, d)  # We only do forward step, no need to integrate.

            # We'll use the sensordata array to visualize key values as real-time bar-graphs (use F4 in viewer)
            d.sensordata[0] = state.reward
            d.sensordata[1:] = state.info['target_angles']

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    pass


if __name__ == '__main__':
    main()
