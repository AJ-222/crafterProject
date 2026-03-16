from sb3_contrib import RecurrentPPO as PPO
import os

def train_ppo(env, total_timesteps, log_dir):
    print(f"Starting PPO training for {total_timesteps} timesteps...")
    model = PPO(
        "CnnLstmPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        learning_rate=2.5e-4,
        tensorboard_log=os.path.join(log_dir, 'tensorboard_lstm')
    )
    model.learn(total_timesteps=total_timesteps)
    model_path = os.path.join(log_dir, 'ppo_model.zip')
    model.save(model_path)   
    print(f"PPO LSTM training finished. Model saved to {model_path}")
    return model_path