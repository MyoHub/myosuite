import timeit

import numpy as np

## for standard myosuite-MJX envs
from myosuite.utils import gym
from stable_baselines3 import PPO

def measure_num_env_training_steps(env_name="MjxElbowPoseRandom-v0", num_timesteps=5_000_000):
  """Measure the training time of base (non-MJX/Warp) MyoSuite environments (total number of steps)"""

  def run_benchmark():
    env = gym.make(env_name)
    
    env.reset()

    model = PPO("MlpPolicy", env, verbose=0, device="cpu")

    print("========================================")
    print("Starting policy learning")
    print("========================================")

    model.learn(total_timesteps=num_timesteps)

    print("========================================")
    print("Job Finished.")
    print("========================================")
  
    env.close()

  results = timeit.repeat(run_benchmark, number=1, repeat=3)
  print(f"Results for single CPU env: {num_timesteps} total steps take {results} seconds.")
  res = np.mean(results)
  print(res)

  return res

def main():
  env_names = ["myoElbowPose1D6MRandom-v0", "myoFingerPoseRandom-v0", "myoHandReachRandom-v0"]
  assert all([gym.spec(env_name) is not None for env_name in env_names]), "Please ensure the env names are correct and registered in MyoSuite."
  
  results = {}
  for env_name in env_names:
      print(f"MyoSuite env {env_name} -- Testing implementation: CPU")
      results[f"{env_name}_CPU"] = measure_num_env_training_steps(env_name=env_name)
  print(results)
  
  np.save("mjx_benchmark_PPO_results_baseline.npy", results, allow_pickle=True)

if __name__ == "__main__":
  main()
