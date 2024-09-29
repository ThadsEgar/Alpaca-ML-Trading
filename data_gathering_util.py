import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import time

from indicators_util import apply_all_indicators

API_KEY = "REDACTED"
API_SECRET = "REDACTED"
client = CryptoHistoricalDataClient(API_KEY, API_SECRET)

def fetch_and_save_bars_with_indicators_to_csv(symbol, timeframe, start_date, end_date, filename, bar_limit=5000, overlap_bars=30):
    current_start = start_date
    data_dict = {
        'symbol': [],
        'timestamp': [],
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': [],
        'trade_count': [],
        'vwap': []
    }
    
    while current_start < end_date:
        bars_per_day = 12 * 24
        days_per_request = bar_limit / bars_per_day
        
        current_end = min(current_start + timedelta(days=days_per_request), end_date)
        overlap_start = max(current_start - timedelta(minutes=overlap_bars * 5), start_date)  # Adjust the overlap window
        print(current_start, current_end)
        
        request_params = CryptoBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=timeframe,
            start=overlap_start,
            end=current_end,
            limit=5000
        )

        bars = client.get_crypto_bars(request_params)

        bars = bars[symbol]
        print(len(bars))
        if bars:
            print("Data fetched for:", symbol)
            for bar in bars:
                data_dict['symbol'].append(bar.symbol)
                data_dict['timestamp'].append(bar.timestamp)
                data_dict['open'].append(bar.open)
                data_dict['high'].append(bar.high)
                data_dict['low'].append(bar.low)
                data_dict['close'].append(bar.close)
                data_dict['volume'].append(bar.volume)
                data_dict['trade_count'].append(bar.trade_count)
                data_dict['vwap'].append(bar.vwap)
            df = pd.DataFrame(data_dict)

            apply_all_indicators(df)

            df = df.iloc[overlap_bars:]

            df.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)
            
            print(f"Data from {current_start} to {current_end} with indicators saved to {filename}")
            
            for key in data_dict:
                data_dict[key] = []
        
        current_start = current_end
        time.sleep(1)

    print(f"Data fetching complete. Saved to {filename}")

def preproccess_coinbase_data():
    filename = "coinbaseUSD_1-min_data.csv"
    data = pd.read_csv(filename)
    data['timestamp'] = pd.to_datetime(data['Unix Timestamp'], unit='s')
    data = data[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
    data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    data = data.sort_values('timestamp').reset_index(drop=True)
    data_with_indicators = apply_all_indicators(data)
    data_with_indicators = data_with_indicators.fillna(method='ffill').fillna(method='bfill')
    data_with_indicators.to_csv('processed_coinbase_data_with_indicators.csv', index=False)
    print(data_with_indicators.head())
    
preproccess_coinbase_data()