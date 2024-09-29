import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium_env import CryptoTradingEnv
import torch

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"Using device: {device}")

# Paths
MODEL_NAME = 'ppo_crypto_512lstm_256_256_128_64'
VNS_PKL_NAME = 'ppo_crypto_512lstm_256_256_128_64_vns'
csv_file_path = "processed_coinbase_data_with_indicators.csv"

# Load only the last 400k rows from the CSV file
data = pd.read_csv(csv_file_path, skiprows=lambda x: x < (4_900_000 - 400_000) and x != 0)

# Fill missing values (if necessary)
data = data.fillna(method='ffill').fillna(method='bfill')

# Define the validation environment
validation_env = DummyVecEnv([lambda: CryptoTradingEnv(
    data=data,
    initial_balance=5000,
    commission=0.0025,
    render_mode=None
)])

# Load the VecNormalize stats (used during training)
if os.path.exists(f'{VNS_PKL_NAME}.pkl'):
    print('LOADING PKL')
    validation_env = VecNormalize.load(f'{VNS_PKL_NAME}.pkl', validation_env)
validation_env.training = False
validation_env.norm_reward = False

# Load the trained model
try:
    model = RecurrentPPO.load(MODEL_NAME, custom_objects={"torch.load": lambda f: torch.load(f, map_location=torch.device('cpu'))})
except Exception as e:
    print(f"Error loading model: {e}")

lstm_states = None
n_envs = validation_env.num_envs
episode_starts = np.ones((n_envs,), dtype=bool)  # Mask for episode start
obs = validation_env.reset()
balance_over_time = []
price_over_time = []
buy_markers = []  # Points where buy actions happen
sell_markers = []  # Points where sell actions happen
trade_count = 0
successful_trades = 0
failing_trades = 0
total_reward = 0

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))  # Two subplots

line, = ax1.plot([], [], label="Balance")
ax1.set_xlabel("Steps")
ax1.set_ylabel("Balance")
ax1.set_title("Balance Over Time")
ax1.legend()

# Price plot
price_line, = ax2.plot([], [], label="Price")
buy_scatter, = ax2.plot([], [], 'go', label="Buy", markersize=5) 
sell_scatter, = ax2.plot([], [], 'ro', label="Sell", markersize=5)
ax2.set_xlabel("Steps")
ax2.set_ylabel("Price")
ax2.set_title("Crypto Price Over Time")
ax2.legend()

balance_text = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, fontsize=12, verticalalignment='top')
win_rate_text = ax1.text(0.05, 0.90, '', transform=ax1.transAxes, fontsize=12, verticalalignment='top')

def update_plot(step, balance, price, win_percentage, action):
    balance_over_time.append(balance)
    price_over_time.append(price)

    line.set_data(range(len(balance_over_time)), balance_over_time)

    if action == 0:  # 0 = Buy
        buy_markers.append((step, price)) 
    elif action == 1:  # 1 = Sell
        sell_markers.append((step, price))

    if buy_markers:
        buy_steps, buy_prices = zip(*buy_markers)
        buy_scatter.set_data(buy_steps, buy_prices)
    
    if sell_markers:
        sell_steps, sell_prices = zip(*sell_markers)
        sell_scatter.set_data(sell_steps, sell_prices)

    price_line.set_data(range(len(price_over_time)), price_over_time)

    balance_text.set_text(f'Balance: ${balance:.2f}')
    win_rate_text.set_text(f'Win Rate: {win_percentage:.2f}%')

    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    plt.draw()
    plt.pause(0.00001)

done = False
step = 0
while not done:
    action, lstm_states = model.predict(
        obs,
        state=lstm_states, 
        episode_start=episode_starts,
        deterministic=True
    )
    
    results = validation_env.step(action)
    
    if len(results) == 5:
        obs, reward, terminated, truncated, info = results
        done = terminated or truncated
    else:
        obs, reward, done, info = results
    
    total_reward += reward

    current_balance = info[0].get('balance', 5000)
    current_price = data['close'].iloc[step]

    if action in [0, 1]: 
        trade_count += 1
        profit_loss = info[0].get('profit_loss', 0)
        if profit_loss > 0:
            successful_trades += 1
        else:
            failing_trades += 1

    # Calculate win percentage
    if trade_count > 0:
        win_percentage = (successful_trades / trade_count) * 100
    else:
        win_percentage = 0

    update_plot(step, current_balance, current_price, win_percentage, action)

    episode_starts = done 
    
    step += 1

    if done:
        print(f"Validation episode finished. Total reward: {total_reward}")
        break

plt.ioff()
plt.plot(balance_over_time)
plt.show()

print(f"Total Trades: {trade_count}")
print(f"Successful Trades: {successful_trades}")
print(f"Failing Trades: {failing_trades}")
print(f"Win Percentage: {win_percentage:.2f}%")
