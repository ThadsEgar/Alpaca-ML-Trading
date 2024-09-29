import matplotlib.pyplot as plt

def plot_buy_sell_signals(data):
    plt.figure(figsize=(14, 8))

    # Plot the closing prices
    plt.plot(data['timestamp'], data['close'], label='Close Price', color='black')
    
    # Mark Buy signals
    plt.plot(data['timestamp'][data['buy_signal']], data['close'][data['buy_signal']], '^', markersize=10, color='g', label='Buy Signal')
    
    # Mark Sell signals
    plt.plot(data['timestamp'][data['sell_signal']], data['close'][data['sell_signal']], 'v', markersize=10, color='r', label='Sell Signal')
    
    plt.title('Crypto Price and Buy/Sell Signals')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend()

    plt.tight_layout()
    plt.show()