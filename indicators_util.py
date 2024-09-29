import pandas as pd

def calculate_macd(data, short_window=8, long_window=21, signal_window=9):
    data['ema_short'] = data['close'].ewm(span=short_window, adjust=False).mean()
    data['ema_long'] = data['close'].ewm(span=long_window, adjust=False).mean()
    data['macd'] = data['ema_short'] - data['ema_long']
    data['signal'] = data['macd'].ewm(span=signal_window, adjust=False).mean()
    return data

def calculate_rsi(data, window=7):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))
    return data

def calculate_bollinger_bands(data, window=15, num_std=2):
    data['sma'] = data['close'].rolling(window=window).mean()  # 20-period moving average
    data['stddev'] = data['close'].rolling(window=window).std()  # Standard deviation

    data['upper_band'] = data['sma'] + (num_std * data['stddev'])  # Upper Bollinger Band
    data['lower_band'] = data['sma'] - (num_std * data['stddev'])  # Lower Bollinger Band

    return data

def apply_all_indicators(data):
    data = calculate_macd(data)
    data = calculate_rsi(data)
    data = calculate_bollinger_bands(data)
    return data

def find_buy_sell_signals(data):
    # Buy when MACD crosses above the signal line AND RSI < 30 (oversold) AND price is below the lower Bollinger Band
    data['buy_signal'] = (
        (data['macd'] > data['signal']) |
        (data['macd'].shift(1) <= data['signal'].shift(1)) |
        (data['rsi'] < 30) |
        (data['close'] < data['lower_band'])
    )
    
    # Sell when MACD crosses below the signal line AND RSI > 70 (overbought) AND price is above the upper Bollinger Band
    data['sell_signal'] = (
        (data['macd'] < data['signal']) |
        (data['macd'].shift(1) >= data['signal'].shift(1)) |
        (data['rsi'] > 70) |
        (data['close'] > data['upper_band'])
    )
    return data

# Calculate expected revenue based on buy and sell signals
def calculate_expected_revenue(data, trade_amount_usd=500, commission_fee=0.0):
    revenue = 0
    positions = []  

    for i in range(len(data)):
        if data['buy_signal'].iloc[i]:
            buy_price = data['close'].iloc[i]
            btc_bought = trade_amount_usd / buy_price
            positions.append((btc_bought, buy_price)) 

        # Sell signal
        if data['sell_signal'].iloc[i] and len(positions) > 0:
            sell_price = data['close'].iloc[i]
            btc_bought, buy_price = positions.pop(0)  # Pop the earliest buy position

            profit = btc_bought * (sell_price - buy_price)

            # Apply commission fees (taker fee) for both the buy and sell transactions
            buy_fee = trade_amount_usd * commission_fee
            sell_fee = (btc_bought * sell_price) * commission_fee 
            net_profit = profit - buy_fee - sell_fee 

            revenue += net_profit 

    return revenue