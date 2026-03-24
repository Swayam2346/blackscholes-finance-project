import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(1)
    return data

def calculate_returns(prices):
    returns = prices.pct_change().dropna()
    return returns

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0):
    p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_returns - risk_free_rate) / p_std

def check_sum(weights):
    return np.sum(weights) - 1

def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficient_portfolios = []
    num_assets = len(mean_returns)
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = num_assets * [1. / num_assets,]
    for ret in returns_range:
        constraints = ({'type': 'eq', 'fun': check_sum},
                       {'type': 'eq', 'fun': lambda w: portfolio_performance(w, mean_returns, cov_matrix)[0] - ret})
        result = minimize(lambda w: portfolio_performance(w, mean_returns, cov_matrix)[1], initial_guess,
                          method='SLSQP', bounds=bounds, constraints=constraints)
        efficient_portfolios.append(result)
    return efficient_portfolios

def monte_carlo_simulation(prices, weights, num_simulations=3000, num_days=252):
    mean_returns = prices.pct_change().mean()
    cov_matrix = prices.pct_change().cov()
    portfolio_mean = np.dot(weights, mean_returns) * num_days
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * num_days, weights)))
    results = np.zeros(num_simulations)
    for i in range(num_simulations):
        daily_returns = np.random.normal(portfolio_mean / num_days, portfolio_std / np.sqrt(num_days), num_days)
        results[i] = np.prod(1 + daily_returns)
    return results

def plot_efficient_frontier(efficient_portfolios, mean_returns, cov_matrix):
    returns = [p['fun'] for p in efficient_portfolios]
    risks = []
    for p in efficient_portfolios:
        risks.append(p['fun'])
    plt.plot(risks, returns, 'b-o')
    plt.xlabel('Risk (Std. Deviation)')
    plt.ylabel('Return')
    plt.title('Efficient Frontier')
    plt.savefig('efficient_frontier.png')
    plt.close()

def plot_monte_carlo(sim_results, savepath="monte_carlo.png"):
    plt.hist(sim_results, bins=50, alpha=0.7)
    plt.title('Monte Carlo Simulation Results')
    plt.xlabel('Portfolio Value')
    plt.ylabel('Frequency')
    plt.savefig(savepath)
    plt.close()

def compute_var_from_simulation(sim_results, initial_value=10000, confidence_levels=(0.05, 0.01)):
    var_results = {}
    sorted_results = np.sort(sim_results)
    for cl in confidence_levels:
        index = int(cl * len(sorted_results))
        var = initial_value - initial_value * sorted_results[index]
        var_results[f'VaR_{int(cl*100)}%'] = var
        cvar = initial_value - initial_value * np.mean(sorted_results[:index])
        var_results[f'CVaR_{int(cl*100)}%'] = cvar
    return var_results

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    prices = download_data(tickers, '2020-01-01', '2023-01-01')
    returns = calculate_returns(prices)
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    num_assets = len(tickers)
    weights = np.array(num_assets * [1. / num_assets,])

    returns_range = np.linspace(mean_returns.min(), mean_returns.max(), 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, returns_range)
    plot_efficient_frontier(efficient_portfolios, mean_returns, cov_matrix)

    sim_results = monte_carlo_simulation(prices, weights, num_simulations=3000, num_days=252)
    plot_monte_carlo(sim_results, savepath="monte_carlo.png")

    var_res = compute_var_from_simulation(sim_results, initial_value=10000, confidence_levels=(0.05, 0.01))