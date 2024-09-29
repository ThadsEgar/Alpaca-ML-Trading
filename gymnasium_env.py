import pygame
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class CryptoTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=5000, commission=0.0025, render_mode='rgb_array', debug=False):
        self.data = data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = None
        self.current_step = 0
        self.commission = commission
        self.render_mode = render_mode
        self.metadata = {"render_modes": ["human", 'rgb_array'], "render_fps": 30}
        self.debug = debug
        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption('Trading Agent')
            self.clock = pygame.time.Clock()
            self.fps = 30
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.balance = self.initial_balance
        self.position = None
        self.current_step = 0
        
        self.buy_price = 0
        self.trade_capital = 0

        obs = self._get_observation()
        info = {}
        return obs, info

    def _get_observation(self):
        entry_price = self.buy_price if self.position == 'buy' and self.buy_price != None else -1
        obs = self.data.iloc[self.current_step][['close', 'rsi', 'upper_band', 'lower_band', 'volume', 'ema_short', 'ema_long', 'macd']].values
        return np.append(obs, [entry_price, self.balance]).astype(np.float32)

    def step(self, action):
        current_price = self.data['close'].iloc[self.current_step]
        reward = 0
        stop_loss_pct = 0.005
        take_profit_pct = 0.04
        position_size = 0.1

        min_balance_threshold = 10
        min_sell_threshold = .01

        if action == 0:
            if self.position is None and self.balance > min_balance_threshold:
                self.position_size = (self.balance * position_size) / (current_price * (1 + self.commission))
                
                if self.position_size > 0:
                    self.buy_price = current_price
                    self.trade_capital = self.position_size * current_price
                    self.position = 'buy'
                    self.balance -= self.trade_capital * self.commission
                    reward = 1
                else:
                    reward = -2
                    if self.debug: 
                        print("Insufficient funds to buy.")
            else:
                reward = -2
                if self.debug: 
                    print("Already in position or insufficient balance, can't buy.")

        elif action == 1:
            if self.position == 'buy':
                profit = (current_price - self.buy_price) * self.position_size * (1 - self.commission)

                if abs(profit) > min_sell_threshold:
                    if profit > 0:
                        reward = profit * 1.25
                    else:
                        reward = profit * 1.25

                    self.balance += profit
                    self.position = None
                    self.position_size = 0
                    self.buy_price = None
                else:
                    reward = -2
                    if self.debug: 
                        print(f"Sell attempt with insignificant profit/loss: {profit:.5f}")
            else:
                reward = -2
                if self.debug: 
                    print("Attempted to sell, but no position is open to sell.")

        elif action == 2:
            if self.position == 'buy':
                unrealized_profit = (current_price - self.buy_price) * self.position_size

                if unrealized_profit < -self.trade_capital * stop_loss_pct:
                    loss = abs(unrealized_profit)
                    reward = -loss * 2
                    self.balance -= loss
                    self.position = None
                    self.position_size = 0  
                    self.buy_price = None
                    if self.debug:
                        print(f"Stopped out at: {current_price}, Unrealized Loss: {unrealized_profit:.2f}")

                elif unrealized_profit > self.trade_capital * take_profit_pct:
                    reward = unrealized_profit * (1 - self.commission) * 2
                    self.balance += unrealized_profit
                    self.position = None
                    self.position_size = 0  
                    self.buy_price = None
                    if self.debug:
                        print(f"TP out at: {current_price}, Unrealized Gain: {unrealized_profit:.2f}")

        self.current_step += 1

        if self.balance <= min_balance_threshold:
            print("Account balance is bust!")
            reward = -100
            done = True
        else:
            done = self.current_step >= len(self.data) - 1

        obs = self._get_observation()

        truncated = False

        info = {'balance': self.balance}

        return obs, reward, done, truncated, info


    def render(self):
        if self.render_mode == 'human':
            width, height = 800, 600
            if not hasattr(self, 'screen'):
                pygame.init()
                self.screen = pygame.display.set_mode((width, height))
                pygame.display.set_caption("Trading Environment")
                self.clock = pygame.time.Clock()

            self.screen.fill((30, 30, 30))

            window_size = 100
            if self.current_step < window_size:
                visible_prices = self.data['close'].iloc[:self.current_step]
            else:
                visible_prices = self.data['close'].iloc[self.current_step - window_size:self.current_step]

            min_price, max_price = np.min(visible_prices), np.max(visible_prices)

            if min_price == max_price:
                min_price -= 1e-5

            font = pygame.font.SysFont('Arial', 24)
            balance_text = font.render(f'Balance: {self.balance:.2f}', True, (255, 255, 255))
            self.screen.blit(balance_text, (50, 50))

            price_chart_height = height * 0.5
            price_chart_top = height * 0.3
            price_color = (0, 255, 0)

            for i in range(len(visible_prices) - 1):
                x1 = i * (width / window_size)
                x2 = (i + 1) * (width / window_size)
                y1 = price_chart_top + (1 - (visible_prices.iloc[i] - min_price) / (max_price - min_price)) * price_chart_height
                y2 = price_chart_top + (1 - (visible_prices.iloc[i + 1] - min_price) / (max_price - min_price)) * price_chart_height
                pygame.draw.line(self.screen, price_color, (x1, y1), (x2, y2), 2)

            for j in range(len(self.actions)):
                action = self.actions[j]

                if j >= self.current_step - window_size and j < self.current_step:
                    window_idx = j - (self.current_step - window_size)
                    x = window_idx * (width / window_size)

                    price_idx = j - (self.current_step - window_size)

                    if price_idx < 0 or price_idx >= len(visible_prices):
                        continue

                    y = price_chart_top + (1 - (visible_prices.iloc[price_idx] - min_price) / (max_price - min_price)) * price_chart_height

                    if np.isnan(y):
                        y = price_chart_top

                    if action == 1:
                        pygame.draw.circle(self.screen, (0, 255, 0), (int(x), int(y)), 5)
                    elif action == 2:
                        pygame.draw.circle(self.screen, (255, 0, 0), (int(x), int(y)), 5)

            if self.position == 'buy':
                current_price = visible_prices.iloc[-1]
                unrealized_gain = (current_price - self.buy_price) * self.position_size * (1 - self.commission)

                entry_text = font.render(f'Entry Price: {self.buy_price:.2f}', True, (255, 255, 255))
                gain_text = font.render(f'Unrealized Gain: {unrealized_gain:.2f}', True, (255, 255, 255))

                self.screen.blit(entry_text, (50, 100))
                self.screen.blit(gain_text, (50, 150))

            if self.position == 'buy':
                position_status_text = font.render("Holding Position", True, (255, 255, 255))
            else:
                position_status_text = font.render("Not Holding Position", True, (255, 255, 255))
            
            self.screen.blit(position_status_text, (50, 200))
            
            pygame.display.flip()

            self.clock.tick(30)

        elif self.render_mode == 'rgb_array':
            print(self.balance)

        return
    
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class ValidationCallback(BaseCallback):
    def __init__(self, validation_env, n_steps, verbose=1):
        super(ValidationCallback, self).__init__(verbose)
        self.validation_env = validation_env
        self.n_steps = n_steps
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

        obs, _ = self.validation_env.reset()
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



