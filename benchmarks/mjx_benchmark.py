import jax
import timeit

import numpy as np

## for standard myosuite-MJX envs
from mujoco_playground import registry as pg_registry
from myosuite.envs.myo.mjx import make as mjx_make


def measure_num_env_simulation_steps(seed=0, loop_iterations=16, env_name="MjxElbowPoseRandom-v0", impl="jax"):
  """Measure how number of envs influences execution time (total number of steps)"""
  env = mjx_make(env_name, config_overrides={"impl": impl})  # Overwrite with your env's name
  u = env.action_size

  v_reset = jax.jit(jax.vmap(env.reset))
  v_step = jax.vmap(env.step)

  main_key = jax.random.PRNGKey(seed)
  reset_key, scan_key = jax.random.split(main_key)
  res = []
  for e in [64, 512, 1024, 2048, 4096, 8192]:
    reset_keys = jax.random.split(reset_key, e)
    v_state = v_reset(reset_keys)

    def physics_loop(carry, _):
      state, key = carry
      key, subkey = jax.random.split(key)
      actions = jax.random.uniform(subkey, shape=(e, u), minval=0.0, maxval=1.0)
      next_s = v_step(state, actions)
      return (next_s, key), None

    print(f"Testing env num: {e}")

    jit_loop = lambda s_init, k_init: jax.lax.scan(
      physics_loop, (s_init, k_init), None, length=loop_iterations
    )

    (end_state, _), _ = jit_loop(v_state, scan_key)  # Preheat the function
    jax.block_until_ready(end_state)

    def run_benchmark():
      (final_state, _), _ = jit_loop(v_state, scan_key)
      jax.block_until_ready(final_state)

    results = timeit.repeat(run_benchmark, number=8192 // e, repeat=3)
    print(f"Results for {e} envs: {8192*loop_iterations} total steps take {results} seconds.")
    res.append(np.mean(results))
  print("[" + ", ".join([str(r) for r in res]) + "]")

  return res


def main():
  results = {}
  for env_name in ["MjxElbowPoseRandom-v0", "MjxFingerPoseRandom-v0", "MjxHandReachRandom-v0"]:
    for impl in ["warp", "jax"]:
      print(f"MyoSuite env {env_name} -- Testing implementation: {impl}")
      results[f"{env_name}_{impl}"] = measure_num_env_simulation_steps(seed=0, env_name=env_name, impl=impl)
  print(results)

  np.save("mjx_benchmark_results.npy", results, allow_pickle=True)

if __name__ == "__main__":
  main()
