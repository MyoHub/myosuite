from stable_baselines3 import PPO
import myosuite
# from myosuite.utils.import_utils import import_gym; gym = import_gym()
from myosuite.utils import gym

from myosuite import myosuite_myobase_suite
from myosuite import myosuite_myochal_suite
for env_name in ['myoHandKeyTurnFixed-v0']:#myosuite_myobase_suite+myosuite_myochal_suite:
    print("\n\n"+env_name + " ---- Starting policy learning")
    env = gym.make(env_name)
    model = PPO("MlpPolicy", env, verbose=0)
    print("========================================")
    model.learn(total_timesteps=10)


