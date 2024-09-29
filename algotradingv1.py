import os
import pandas as pd
from datetime import datetime
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import numpy as np
from torch.optim.lr_scheduler import StepLR
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
import torch
from gymnasium_env import CryptoTradingEnv
from sb3_util import LRSchedulerCallback, SaveVecNormalizeCallback, TensorboardRewardLoggingCallback, ValidationCallback, lr_schedule, EntropyDecayCallback
import cProfile
import pstats

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"Using device: {device}")

import os
print("Current Working Directory: ", os.getcwd())

PREDICT_ONLY = False
MODEL_PATH = "ppo_crypto_trader"
NUM_ENVS = 72
CHECKPOINT_SAVE_INTERVAL = 200_000
TOTAL_TIMESTEPS_INITIAL = 500_000_000
TOTAL_TIMESTEPS_CONTINUE = 500_000_000
MODEL_NAME = 'ppo_crypto_768lstm_512_256_128_64'
csv_file_path = "processed_coinbase_data_with_indicators.csv"

pr = cProfile.Profile()
pr.enable()
data = pd.read_csv(csv_file_path, skiprows=range(1, 2_000_000))
data = data.fillna(method='ffill').fillna(method='bfill')
train_data = data.iloc[:-100_000]
validation_data = data.iloc[-100_000:]

def make_env(data, rank):
    def _init():
        env = CryptoTradingEnv(
            data=data,
            initial_balance=5000,
            commission=0.0025,
            render_mode=None
        )
        return env
    return _init

if __name__ == '__main__':
    if not PREDICT_ONLY:
        env_fns = [make_env(train_data, i) for i in range(NUM_ENVS)]
        train_env = SubprocVecEnv(env_fns)
        validation_env = DummyVecEnv([lambda: CryptoTradingEnv(data=validation_data)])
                if os.path.exists(f'{MODEL_NAME}_vns.pkl'):
            print(f"Loading existing pkl from from {MODEL_NAME}_vns.pkl...")
            train_env = VecNormalize.load(f'{MODEL_NAME}_vns.pkl', train_env)
        else:
            train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

        train_env.training = True
        train_env.norm_reward = True
        
        initial_lr = 2e-5 

        if os.path.exists(MODEL_NAME + ".zip"):
            print(f"Loading existing model from {MODEL_NAME}...")
            model = RecurrentPPO.load(MODEL_NAME, env=train_env, device=device, learning_rate=lr_schedule, initial_lr=initial_lr)

            previous_timesteps = model.num_timesteps
            print(f"Model previously trained for: {previous_timesteps} timesteps")
            for param_group in model.policy.optimizer.param_groups:
                param_group['lr'] = initial_lr

            total_timesteps = TOTAL_TIMESTEPS_CONTINUE 

        else:
            previous_timesteps = 0
            initial_lr = 5e-4
            total_timesteps = TOTAL_TIMESTEPS_INITIAL

            policy_kwargs = dict(
                lstm_hidden_size=768,
                net_arch=[512, 256, 128, 64]
            )

            model = RecurrentPPO(
                'MlpLstmPolicy',
                train_env,
                learning_rate=lr_schedule,
                verbose=0, 
                tensorboard_log="./ppo_crypto_tensorboard/", 
                policy_kwargs=policy_kwargs,
                n_steps=8192,
                batch_size=8192,
                device=device,
                n_epochs=4,
                ent_coef=0.03
            )

        entropy_decay_callback = EntropyDecayCallback(
            initial_entropy_coef=0.03,
            final_entropy_coef=0.001,
            total_timesteps=TOTAL_TIMESTEPS_INITIAL,
            decay_rate=0.99
        )
        reward_logging_callback = TensorboardRewardLoggingCallback(log_freq=8192)
        checkpoint_callback = CheckpointCallback(
            save_freq=CHECKPOINT_SAVE_INTERVAL, 
            save_path='./models/', 
            name_prefix=MODEL_NAME
        )
        save_vec_normalize_callback = SaveVecNormalizeCallback(save_freq=CHECKPOINT_SAVE_INTERVAL, save_path='./models/', model_name=MODEL_NAME, verbose=1)
        validation_callback = ValidationCallback(validation_env=validation_env, n_steps=8192*64*8, model_name=MODEL_NAME)

        callback = CallbackList([reward_logging_callback, checkpoint_callback, save_vec_normalize_callback, validation_callback, entropy_decay_callback])
        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
        model.save(MODEL_NAME)
        train_env.save(f'{MODEL_NAME}_vns.pkl')
        
    pr.disable()
    ps = pstats.Stats(pr)
    ps.sort_stats('cumtime').print_stats(20) 
