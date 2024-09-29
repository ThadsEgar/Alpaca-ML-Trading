import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

class TensorboardRewardLoggingCallback(BaseCallback):
    def __init__(self, log_freq=1000, verbose=0):
        super(TensorboardRewardLoggingCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.episode_length = 0

    def _on_step(self) -> bool:
        self.episode_length += 1

        immediate_reward = self.locals["rewards"][0]

        if self.num_timesteps % self.log_freq == 0:
            self.logger.record('rollout/step_reward', immediate_reward)
            print(f"Step {self.num_timesteps}: Immediate Reward: {immediate_reward}")

        if self.locals["dones"][0]:
            self.logger.record('rollout/episode_length', self.episode_length)
            print(f"Episode Length: {self.episode_length}")
            
            self.episode_length = 0

        return True
    
def lr_schedule(progress_remaining):
    lr = max(1e-5, 5e-4 * progress_remaining)
    print(f'lr: {lr}')
    return lr
            
    
class LRSchedulerCallback(BaseCallback):
            def __init__(self, scheduler, verbose=1):
                super(LRSchedulerCallback, self).__init__(verbose)
                self.scheduler = scheduler
        
            def _on_rollout_end(self):
                self.scheduler.step()
                if self.verbose > 0:
                    print(f"Learning rate updated to: {self.scheduler.get_last_lr()[0]}")
        
            def _on_step(self):
                return True
            
class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_freq, save_path, model_name, verbose=0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.model_name = model_name

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            vec_normalize_file = os.path.join(self.save_path, f'{self.model_name}_vns_{self.n_calls}_STEPS.pkl')
            self.training_env.save(vec_normalize_file)
            if self.verbose > 0:
                print(f"VecNormalize stats saved to {vec_normalize_file} at step {self.n_calls}")
        return True
            

class EntropyDecayCallback(BaseCallback):
    def __init__(self, initial_entropy_coef, final_entropy_coef, total_timesteps, decay_rate=0.99, verbose=0):
        super(EntropyDecayCallback, self).__init__(verbose)
        self.initial_entropy_coef = initial_entropy_coef
        self.final_entropy_coef = final_entropy_coef
        self.total_timesteps = total_timesteps
        self.decay_rate = decay_rate
        self.current_step = 0

    def _on_step(self) -> bool:
        timesteps_left = max(self.total_timesteps - self.current_step, 1)
        progress = self.current_step / self.total_timesteps

        decayed_entropy_coef = self.initial_entropy_coef * (self.decay_rate ** progress) + \
                               self.final_entropy_coef * (1 - self.decay_rate ** progress)
        
        self.model.ent_coef = decayed_entropy_coef
        
        self.current_step += 1
        
        if self.verbose > 0:
            print(f"Step: {self.current_step}, Entropy Coefficient: {decayed_entropy_coef}")

        return True


class ValidationCallback(BaseCallback):
    def __init__(self, validation_env, n_steps, model_name, verbose=1):
        super(ValidationCallback, self).__init__(verbose)
        self.validation_env = validation_env
        self.n_steps = n_steps
        self.model_name = model_name
        self.validation_counter = 0
        self.lstm_states = None
        self.episode_starts = True
        self.total_reward = 0
    
    def _on_step(self):
        if self.num_timesteps - self.validation_counter >= self.n_steps:
            self.validation_counter = self.num_timesteps
            self.run_validation()

        return True
    
    def run_validation(self):
        print(f"Running validation at {self.num_timesteps} steps")

        vecnormalize_path = f'{self.model_name}_vns.pkl'
        if os.path.exists(vecnormalize_path):
            self.validation_env = VecNormalize.load(vecnormalize_path, self.validation_env)
            self.validation_env.training = False
            self.validation_env.norm_reward = False
        else:
            print(f"Warning: VecNormalize stats file not found at {vecnormalize_path}")

        obs = self.validation_env.reset()
        self.lstm_states = None
        self.episode_starts = True
        self.total_reward = 0

        done = False

        while not done:
            action, self.lstm_states = self.model.predict(
                obs,
                state=self.lstm_states,
                episode_start=np.array([self.episode_starts]),
                deterministic=True
            )

            obs, reward, done, info = self.validation_env.step(action)

            self.total_reward += reward

            self.episode_starts = done

        print(f"Validation completed. Total reward: {self.total_reward}")

        self.logger.record('validation/total_reward', self.total_reward)
        self.logger.dump(self.num_timesteps)