import os
import datetime
import gymnasium as gym
import myosuite
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback

def main():
    # 1. Configuration
    env_name = "MyoSkeletonStand-v0"
    total_timesteps = 1_000_000
    seed = 42

    # Output directory
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"stand-baseline-{time_str}"
    log_dir = f"./runs/{run_id}"
    os.makedirs(log_dir, exist_ok=True)

    # 2. WandB Setup
    run = wandb.init(
        project="myoskeleton-ppo",
        name="ppo-standing",
        config={
            "env_name": env_name,
            "total_timesteps": total_timesteps,
            "algorithm": "PPO",
        },
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    # 3. Environment Setup
    def make_env():
        env = gym.make(env_name)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # 4. Model Definition
    # PPO with optimized hyperparameters for stability
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        seed=seed,
        learning_rate=1e-4,  # Lower learning rate for stability
        n_steps=4096,         # More steps for better gradient estimation
        batch_size=256,       # Larger batch size
        n_epochs=20,          # More epochs for better policy improvement
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,        # Encourage exploration
    )

    # 5. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=log_dir,
        name_prefix="ppo_stand_model",
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run_id}",
        verbose=2,
    )

    # 6. Training
    print(f"Starting training on {env_name}...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, wandb_callback],
        progress_bar=True
    )

    # 7. Save Final Model
    model.save(f"{log_dir}/ppo_stand_final")
    print(f"Training complete. Model saved to {log_dir}")

    run.finish()

if __name__ == "__main__":
    main()
