import timeit

import numpy as np

from myosuite.utils import gym


def measure_num_env_simulation_steps(num_steps=8192*16, env_name="ElbowPoseRandom-v0"):
  """Measure the execution time of base (non-MJX/Warp) MyoSuite environments (total number of steps)"""
  env = gym.make(env_name)

  env.reset()

  def single_step():
      a = np.random.uniform(low=0.0, high=1.0, size=env.action_space.shape)  # Sample a random action
      next_o, r, done, *_, ifo = env.step(a)  # take an action
  
  results = timeit.repeat(single_step, number=num_steps, repeat=3)
  print(f"Results for single CPU env: {num_steps} total steps take {results} seconds.")
  res = np.mean(results)
  print(res)

  env.close()

  return res


def main():
  env_names = ["myoElbowPose1D6MRandom-v0", "myoFingerPoseRandom-v0", "myoHandReachRandom-v0"]
  assert all([gym.spec(env_name) is not None for env_name in env_names]), "Please ensure the env names are correct and registered in MyoSuite."
  
  results = {}
  for env_name in env_names:
      print(f"MyoSuite env {env_name} -- Testing implementation: CPU")
      results[f"{env_name}_CPU"] = measure_num_env_simulation_steps(env_name=env_name)
  print(results)

  np.save("mjx_benchmark_results_baseline.npy", results, allow_pickle=True)

if __name__ == "__main__":
  main()
