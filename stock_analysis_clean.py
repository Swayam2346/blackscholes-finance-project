"""Black-Scholes Option Pricing & Market Analytics Platform"""

from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from scipy.stats import norm

sns.set(style="darkgrid", palette="muted")

TICKERS = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL"]
TRADING_DAYS = 252
RISK_FREE_RATE = 0.05


def download_data(tickers: list[str], years: int = 10) -> pd.DataFrame:
    """Pulls 10 years of adjusted closing prices for the provided tickers."""
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=years)
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
        threads=True,
        group_by="ticker",
    )
    prices = raw["Close"].copy()
    prices.dropna(how="all", inplace=True)
    return prices


def calculate_returns(price_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily returns show percent price changes, while log returns help in additive models."""
    simple_returns = price_df.pct_change().dropna()
    log_returns = np.log(price_df / price_df.shift(1)).dropna()
    return simple_returns, log_returns


def calculate_volatility(returns_df: pd.DataFrame, trading_days: int = TRADING_DAYS) -> pd.Series:
    """Annualized volatility quantifies the standard deviation of returns over a year."""
    return returns_df.std() * np.sqrt(trading_days)


def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float) -> dict[str, float]:
    """Returns the Black-Scholes Greeks for a call option, which explain sensitivity."""
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (
        -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2)
    )
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega / 100,
        "rho": rho / 100,
    }


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def portfolio_analysis(
    returns_df: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS,
) -> dict[str, float]:
    """Equally weighted portfolio metrics illustrate diversification benefits."""
    weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
    mean_daily = returns_df.mean()
    cov = returns_df.cov()
    daily_return = float(np.dot(weights, mean_daily))
    daily_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
    annual_return = daily_return * trading_days
    annual_vol = daily_vol * np.sqrt(trading_days)
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol else 0.0
    return {
        "return": annual_return,
        "volatility": annual_vol,
        "sharpe_ratio": sharpe,
    }


def visualize_data(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    log_returns: pd.DataFrame,
    volatility: pd.Series,
    correlation: pd.DataFrame,
) -> None:
    """Creates professional charts that highlight price trends and risk diagnostics."""
    fig, ax = plt.subplots(figsize=(12, 6))
    prices.plot(ax=ax)
    ax.set_title("Price Trend Across Assets")
    ax.set_ylabel("Adjusted Close Price")
    ax.legend(title="Ticker")
    plt.tight_layout()

    for ticker in prices.columns:
        ma50 = prices[ticker].rolling(window=50).mean()
        ma200 = prices[ticker].rolling(window=200).mean()
        plt.figure(figsize=(11, 4))
        plt.plot(prices[ticker], label=f"{ticker} Price", linewidth=1.4)
        plt.plot(ma50, label="50d MA", linestyle="--")
        plt.plot(ma200, label="200d MA", linestyle=":")
        plt.title(f"{ticker} Moving Averages")
        plt.legend()
        plt.tight_layout()

    plt.figure(figsize=(10, 5))
    sns.barplot(x=volatility.index, y=volatility.values, palette="crest")
    plt.title("Annualized Volatility Comparison")
    plt.ylabel("Volatility")
    plt.tight_layout()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Log Return Correlation")
    plt.tight_layout()

    plt.figure(figsize=(12, 5))
    sns.histplot(returns, kde=True, stat="density", element="step", palette="muted")
    plt.title("Daily Return Distribution")
    plt.tight_layout()

    plt.show()


def max_drawdown(price_series: pd.Series) -> float:
    """Maximum drawdown measures the largest peak-to-trough loss over time."""
    rolling_max = price_series.cummax()
    drawdowns = (price_series - rolling_max) / rolling_max
    return drawdowns.min()


def simulate_option_price_changes(S: float, r: float, sigma: float) -> tuple[pd.Series, pd.Series]:
    """Simulates how option prices react to strike and volatility shifts."""
    strikes = np.linspace(S * 0.8, S * 1.2, 30)
    strike_prices = pd.Series(
        [black_scholes_call(S, K, 0.5, r, sigma) for K in strikes], index=strikes
    )
    sigmas = np.linspace(0.1, 0.6, 30)
    vol_prices = pd.Series(
        [black_scholes_call(S, S, 0.5, r, vol) for vol in sigmas], index=sigmas
    )
    return strike_prices, vol_prices


def plot_option_simulations(strike_series: pd.Series, vol_series: pd.Series) -> None:
    """Displays how options price themselves across strike and volatility axes."""
    plt.figure(figsize=(10, 5))
    plt.plot(strike_series.index, strike_series.values, marker="o", label="Call vs Strike")
    plt.title("Option Price vs Strike")
    plt.xlabel("Strike Price")
    plt.ylabel("Call Option Price")
    plt.tight_layout()

    plt.figure(figsize=(10, 5))
    plt.plot(vol_series.index, vol_series.values, marker="o", color="tab:orange")
    plt.title("Option Price vs Volatility")
    plt.xlabel("Volatility")
    plt.ylabel("Call Option Price")
    plt.tight_layout()
    plt.show()


def main() -> None:
    prices = download_data(TICKERS)
    simple_returns, log_returns = calculate_returns(prices)
    volatility = calculate_volatility(simple_returns)
    correlation = log_returns.corr()

    sharpe_ratios = (
        simple_returns.mean() * TRADING_DAYS - RISK_FREE_RATE
    ) / (simple_returns.std() * np.sqrt(TRADING_DAYS))
    drawdowns = prices.apply(max_drawdown)

    portfolio_metrics = portfolio_analysis(simple_returns)
    strike_series, vol_series = simulate_option_price_changes(
        prices.iloc[-1, 0], RISK_FREE_RATE, volatility.iloc[0]
    )

    visualize_data(prices, simple_returns, log_returns, volatility, correlation)
    plot_option_simulations(strike_series, vol_series)

    print("\n--- Financial Insights ---")
    print(f"Sharpe Ratios (annualized):\n{sharpe_ratios.round(2)}")
    print(f"Annual Volatility:\n{volatility.round(2)}")
    print(f"Max Drawdowns:\n{drawdowns.round(2)}")
    print(
        "Portfolio (equally weighted) -> "
        f"Return: {portfolio_metrics['return']:.2%}, "
        f"Volatility: {portfolio_metrics['volatility']:.2%}, "
        f"Sharpe: {portfolio_metrics['sharpe_ratio']:.2f}"
    )

    sample_ticker = TICKERS[0]
    greeks = calculate_greeks(
        prices[sample_ticker].iloc[-1], prices[sample_ticker].iloc[-1] * 1.05, 0.5, RISK_FREE_RATE, volatility[sample_ticker]
    )
    print("\nOption Greeks for a near-the-money call on", sample_ticker)
    for greek, value in greeks.items():
        print(f"  {greek.title()}: {value:.4f}")


if __name__ == "__main__":
    main()
