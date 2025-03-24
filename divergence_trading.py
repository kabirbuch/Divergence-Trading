"""
Code and backtesting for a divergence trading strategy. Detailed strategy
explanation in the slides
"""

import pandas as pd
from matplotlib import pyplot as plt
plt.ion()

BEAR: int = -1
BULL: int = 1
NEITHER: int = 0
wordmap = {BEAR: 'covered', BULL: 'sold'}

def calculate_macd(stocks: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates MACD given our data frame, assuming there is a close and macd
    column. Returns the updated data frame.
    """

    #Can technically change slow and fast periods, but not recommended.
    fast_period:int = 12
    slow_period:int = 26

    #Calculate ema using a given function
    ema_fast = stocks['Close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = stocks['Close'].ewm(span=slow_period, adjust=False).mean()

    #Calc macd
    stocks['Macd'] = ema_fast - ema_slow

    #Calc macd signal line
    signal = [stocks['Macd'].iloc[0]]
    for i, macd in enumerate(stocks['Macd'].iloc[1:], 1):
        signal.append(signal[i - 1] * 0.8 + macd * 0.2)
    stocks['Signal'] = pd.Series(signal, index=stocks['Macd'].index)

    return stocks

def implement_strat(stocks: pd.DataFrame) -> None:
    """
    Conduct day-by-day analysis of the stock data and implement trades
    """
    # Load SnP 500
    snp = pd.read_csv('SnP.csv')
    snp = snp.iloc[::-1]
    snp['Date'] = pd.to_datetime(snp['Date'])
    snp.set_index('Date', inplace=True)
    snp.rename(columns={'Close/Last': 'Close'}, inplace=True)
    stocks['SnP'] = (100000 / snp['Close'].iloc[0]) * snp['Close']
    # Initialize starting conditions
    prev_prices = []
    prev_macd = []
    position = NEITHER
    warning_threshold = 0.1 * 100000
    portfolio = []
    snp_shares = 100000 / snp['Close'].iloc[0]
    results = []
    for iloc in range(0, 20):
        prev_prices.append(stocks['Close'].iloc[iloc])
        prev_macd.append(stocks['Macd'].iloc[iloc])
        results.append(snp_shares * snp['Close'].iloc[iloc])

    # Go day-by-day starting from when we have good MACD data
    for i, day in enumerate(stocks.index[20:], 20):
        curr_macd = stocks['Macd'].iloc[i]
        curr_signal = stocks['Signal'].iloc[i]
        curr_price = stocks['Close'].iloc[i]

        sorted_macd = prev_macd[:] + [curr_macd]
        sorted_macd.sort()
        macd_position = sorted_macd.index(curr_macd)

        sorted_price = prev_prices[:] + [curr_price]
        sorted_price.sort()
        price_position = sorted_price.index(curr_price)

        snp_price = snp['Close'].iloc[i]

        trade_cap = max(1, len(portfolio)) # Each subsequent trade should be smaller
        snp_holdings = snp_shares * snp_price
        if snp_holdings < warning_threshold:
            print(f"Warning: on {day}, S&P holdings became critical: ${snp_holdings}")
            warning_threshold = snp_holdings
        if snp_holdings < 0:
            raise ValueError("No money to trade")

        # Should we realize our investments?
        if ((position == BULL and curr_macd < curr_signal) or
         (position == BEAR and curr_macd > curr_signal)):
            share_count = 0
            gains = 0
            for shares, day in portfolio:
                share_count += shares
                gains += shares * curr_price
            snp_shares += gains / snp_price
            print(f"On {day.date()}, {wordmap[position]} {abs(round(share_count, 2))}"
            f" shares, for a total of ${round(gains, 2)}")
            position = NEITHER
            portfolio = []

        # Check for shorting opportunity
        if stocks['Close'].iloc[i] > max(prev_prices): # Are prices peaking?
            divergence = price_position - macd_position - 5
            if divergence > 0: # Is MACD and the signal lines crossed?
                if curr_macd < curr_signal: # Have MACD and signal lines crossed?
                    investment_size = (((0.3 / trade_cap) * snp_holdings) -
                        ((0.01 / trade_cap) * snp_holdings) * divergence)
                    snp_shares += investment_size / snp_price
                    portfolio.append((-1 * (investment_size / curr_price), day))
                    position = BEAR
                    print(f"On {day.date()}, shorted "
                    f"{round(investment_size / curr_price, 2)} shares, for a total of "
                    f"${round(investment_size, 2)}")

        # Check for buying opportunity
        elif stocks['Close'].iloc[i] < min(prev_prices): # Are prices bottoming?
            divergence = macd_position - price_position - 5
            if divergence > 0: # Is the divergence significant?
                if curr_macd > curr_signal: # Have MACD and signal lines crossed?
                    investment_size = (((0.3 / trade_cap) * snp_holdings) -
                        ((0.01 / trade_cap) * snp_holdings) * divergence)
                    snp_shares -= investment_size / snp_price
                    portfolio.append((investment_size / curr_price, day))
                    position = BULL
                    print(f"On {day.date()}, bought "
                    f"{round(investment_size / curr_price, 2)} shares, for a total of "
                    f"${round(investment_size, 2)}")

        networth = snp_shares * snp_price
        for shares, day in portfolio:
            networth += shares * curr_price
        results.append(networth)
        prev_prices.pop(0)
        prev_macd.pop(0)
        prev_prices.append(stocks['Close'].iloc[i])
        prev_macd.append(stocks['Macd'].iloc[i])

    print(f"Final portfolio value: ${round(networth, 2)}")
    stocks['Portfolio'] = pd.Series(results, index=stocks.index)
    return stocks

def main():
    """
    Execute strategy
    """
    # Initalize date

    trading = pd.read_csv('BA.csv') # Can change stock
    trading = trading.iloc[::-1]
    trading['Date'] = pd.to_datetime(trading['Date'])
    trading.set_index('Date', inplace=True)
    trading.rename(columns={'Close/Last': 'Close'}, inplace=True)
    trading['Close'] = trading['Close'].apply(lambda x: float(x[1:]))
    trading = calculate_macd(trading)
    pd.set_option('display.max_rows', None)

    strategy = implement_strat(trading)

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(12, 6)) # Adjust figure size as needed
    plt.rcParams['font.family'] = 'sans serif'

    # Plot Price data on the first axis
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color='black')
    ax1.plot(strategy.index, strategy['Close'], color=color, label='Price')
    ax1.tick_params(axis='y', labelcolor='black')

    # Create a second axis for MACD data
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('MACD/Signal', color='black')
    ax2.plot(strategy.index, strategy['Macd'],
             color=color, label='MACD')
    ax2.plot(strategy.index, strategy['Signal'],
             color='tab:green', label='Signal') # Plot signal line
    ax2.tick_params(axis='y', labelcolor='black')

    # Add legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    ax1.set_facecolor('#f8f8f3')
    fig.patch.set_facecolor('#f8f8f3')
    fig.tight_layout()    # Adjust layout for better spacing
    plt.title('Price, MACD, and Signal over Time')

    # --- Plotting portfolio value over time ---
    plt.rcParams['font.family'] = 'sans serif'
    fig, ax1 = plt.subplots(figsize=(12, 6))
    plt.plot(strategy.index, strategy['SnP'], label='Holding S&P 500', color='red')
    plt.plot(strategy.index, strategy['Portfolio'], label='Divergence Trading', color='green')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Value over Time')
    ax1.set_facecolor('#f8f8f3')
    fig.patch.set_facecolor('#f8f8f3')
    plt.legend()
    plt.grid(True)
    plt.show(block=True)

if __name__ == "__main__":
    main()
