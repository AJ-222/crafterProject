from sb3_contrib import RecurrentPPO as PPO
import os

def train_ppo_tuned(env, total_timesteps, log_dir, n_steps, batch_size,
                    learning_rate, ent_coef, vf_coef, gamma):
    """
    Creates, trains (indefinitely until interrupted), and saves a PPO LSTM agent
    using specified hyperparameters.
    """
    print(f"Starting PPO LSTM + Shaping training with tuned parameters...")
    print(f"  n_steps: {n_steps}, batch_size: {batch_size}, learning_rate: {learning_rate}")
    print(f"  ent_coef: {ent_coef}, vf_coef: {vf_coef}, gamma: {gamma}")
    print(f"Training indefinitely until Ctrl+C is pressed...")

    model_path = os.path.join(log_dir, 'ppo_lstm_shaped_tuned_model.zip') 

    try:
        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path} to continue training...")
            model = PPO.load(model_path, env=env, custom_objects={'learning_rate': learning_rate})
            model.tensorboard_log = os.path.join(log_dir, 'tensorboard_lstm_shaped_tuned')
        else:
            print("No existing model found, starting training from scratch...")
            model = PPO(
                "CnnLstmPolicy",
                env,
                verbose=1,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=10,
                gamma=gamma,
                learning_rate=learning_rate,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                tensorboard_log=os.path.join(log_dir, 'tensorboard_lstm_shaped_tuned')
            )
        model.learn(total_timesteps=int(total_timesteps), 
                    reset_num_timesteps=not os.path.exists(model_path))

    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C). Saving model...")
        model.save(model_path)
        print(f"Model saved to {model_path}")
        return model_path
    model.save(model_path)
    print(f"PPO LSTM + Shaping training finished (reached max timesteps). Model saved to {model_path}")
    return model_path