
import gymnasium as gym
import myosuite
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import myosuite.envs.myo.myoskeleton # Register envs
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np

class DetailedMetricsCallback(BaseCallback):
    """Custom callback to log detailed metrics to W&B"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Log per-step metrics
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    ep_reward = info['episode']['r']
                    ep_length = info['episode']['l']
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)

                    # Log to W&B
                    wandb.log({
                        'episode/reward': ep_reward,
                        'episode/length': ep_length,
                        'episode/reward_mean_100': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) > 0 else 0,
                    })

                # Log reward components if available
                if 'rwd_dict' in info:
                    rwd_dict = info['rwd_dict']
                    wandb.log({
                        f'reward_components/{k}': v for k, v in rwd_dict.items()
                    })

        return True

def main():
    # Initialize W&B
    run = wandb.init(
        project="myoskeleton-ppo",
        name="ppo-baseline",
        config={
            "env_name": "MyoSkeletonTrack-v0",
            "algorithm": "PPO",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "total_timesteps": 1_000_000,
        },
        sync_tensorboard=True,
        monitor_gym=True,
    )

    env_name = "MyoSkeletonTrack-v0"
    print(f"Creating environment: {env_name}")

    env = gym.make(env_name)

    print("Environment created.")

    # PPO Parameters
    policy_kwargs = dict(net_arch=[64, 64, 64])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"./runs/{run.id}"
    )

    print("Starting training...")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./myoskeleton_checkpoints/',
        name_prefix='myoskeleton_model'
    )

    wandb_callback = WandbCallback(
        model_save_path=f"./models/{run.id}",
        verbose=2,
    )

    metrics_callback = DetailedMetricsCallback()

    model.learn(
        total_timesteps=1_000_000,
        callback=[checkpoint_callback, wandb_callback, metrics_callback]
    )

    model.save("myoskeleton_sb3_final")
    print("Training finished.")

    # Save final model to W&B
    wandb.save("myoskeleton_sb3_final.zip")
    run.finish()

if __name__ == "__main__":
    main()
