import pandas as pd
import numpy as np
import yfinance as yf   # if you use yfinance inside download_data
def download_data(tickers, start=None, end=None, years=10):
    """Download adjusted/close prices robustly from yfinance.

    - If start/end provided they will be used; otherwise `years` back from today is used.
    - Handles MultiIndex columns (ticker, field) as well as flat DataFrames.
    - Prefers 'Adj Close' then 'Close'.
    - Prints column information when something unexpected happens to help debugging.
    """
    # determine date range
    if start is None or end is None:
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.DateOffset(years=years)
    else:
        start_date = start
        end_date = end

    print(f"Downloading tickers={tickers} start={start_date.date()} end={end_date.date()}")
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False,
        group_by='ticker',
        auto_adjust=False,
    )

    # quick sanity check
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data. Check tickers and your internet connection.")

    # Debug: show columns
    print("raw.columns:", raw.columns)

    # Case 1: MultiIndex columns like ("AAPL", "Adj Close") or ("AAPL","Close")
    if isinstance(raw.columns, pd.MultiIndex):
        # try nice selection using level values
        lev1 = list(raw.columns.get_level_values(1))
        if 'Adj Close' in lev1:
            prices = raw.xs('Adj Close', axis=1, level=1).copy()
            print("Using MultiIndex level 'Adj Close'")
            return prices
        if 'Close' in lev1:
            prices = raw.xs('Close', axis=1, level=1).copy()
            print("Using MultiIndex level 'Close'")
            return prices

        # fallback: flatten MultiIndex to simple column names and search for matching patterns
        flat = raw.copy()
        flat.columns = ["_".join(map(str, c)).strip() for c in raw.columns.values]
        print("Flattened columns sample:", flat.columns[:10])
        import re
        adj_cols = [c for c in flat.columns if re.search(r'Adj[\s_]?Close$', c, flags=re.IGNORECASE)]
        close_cols = [c for c in flat.columns if re.search(r'Close$', c, flags=re.IGNORECASE) and c not in adj_cols]
        if adj_cols:
            prices = flat[adj_cols].copy()
            prices.columns = [c.split("_")[0] for c in adj_cols]
            print("Using flattened 'Adj Close' columns")
            return prices
        if close_cols:
            prices = flat[close_cols].copy()
            prices.columns = [c.split("_")[0] for c in close_cols]
            print("Using flattened 'Close' columns")
            return prices

        raise RuntimeError("Could not find 'Adj Close' or 'Close' in MultiIndex columns: " + str(raw.columns))

    # Case 2: flat columns
    # prefer Adj Close
    if 'Adj Close' in raw.columns:
        prices = raw['Adj Close'].copy()
        print("Using 'Adj Close' from flat columns")
        return prices
    if 'Close' in raw.columns:
        prices = raw['Close'].copy()
        print("Using 'Close' from flat columns")
        return prices

    # fallback: maybe the DataFrame already contains only prices
    # attempt to detect by dtype (numeric) and number of columns
    numeric_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
    if numeric_cols:
        prices = raw[numeric_cols].copy()
        print("Using numeric columns as prices (fallback):", numeric_cols[:10])
        return prices

    # final fallback: raise helpful error with column info
    raise RuntimeError("Unable to extract price columns from yfinance output. Columns:\n" + str(list(raw.columns)))
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]

# ---------------------------
# Portfolio analysis section
# ---------------------------
import numpy as np

def analyze_portfolio(prices, weights, risk_free_rate=0.02):
    """
    Given a price DataFrame (columns = tickers) and a weights array,
    return annualized expected return, annualized volatility, and Sharpe.
    """
    # ensure DataFrame columns order matches weights length
    if prices.shape[1] != len(weights):
        raise ValueError(f"Number of weights ({len(weights)}) must match number of price columns ({prices.shape[1]}).")

    # daily returns
    returns = prices.pct_change().dropna()

    # mean daily return vector
    mean_daily = returns.mean()

    # annualize expected portfolio return (simple arithmetic)
    annualized_return = float(np.dot(mean_daily.values, weights) * 252)

    # annualized covariance matrix
    cov_annual = returns.cov() * 252

    # portfolio volatility (std)
    portfolio_vol = float(np.sqrt(weights.T @ cov_annual.values @ weights))

    # Sharpe ratio (using provided risk_free_rate, annual)
    sharpe = (annualized_return - risk_free_rate) / portfolio_vol if portfolio_vol != 0 else np.nan

    return {
        "annualized_return": annualized_return,
        "annualized_volatility": portfolio_vol,
        "sharpe": sharpe,
        "returns_df_head": returns.head()
    }

if __name__ == "__main__":
    import traceback, sys, time
    print(">>> Running stock_analysis.py (debug mode) —", time.asctime())
    try:
        # quick sanity: show file saved timestamp
        import os
        print("File modified:", time.ctime(os.path.getmtime(__file__)))
        # example args (edit if you want)
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        prices = download_data(tickers)   # must be defined above
        print("download_data() returned. Columns:", list(prices.columns))
        # Example: equal weights (ensure length matches)
        import numpy as np
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        results = analyze_portfolio(prices, weights, risk_free_rate=0.02)
        print("Date range:", prices.index.min().date(), "to", prices.index.max().date())
        print(f"Annualized expected return: {results['annualized_return']:.2%}")
        print(f"Annualized volatility (std): {results['annualized_volatility']:.2%}")
        print(f"Sharpe ratio (rf=2%): {results['sharpe']:.3f}")
        print("\nSample daily returns (head):")
        print(results['returns_df_head'].to_string())
    except Exception as e:
        print("!!! Exception occurred — printing traceback:")
        traceback.print_exc(file=sys.stdout)
        # also save the traceback to run_debug.log for inspection
        with open("run_debug.log", "w") as f:
            traceback.print_exc(file=f)
        print("Traceback also saved to run_debug.log")
        # ---------------------------
# Efficient frontier + plotting
# ---------------------------
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def portfolio_stats(weights, mean_daily, cov_annual):
    """
    Return (annual_return, annual_volatility) for a given weights vector.
    mean_daily: pd.Series of mean daily returns
    cov_annual: annualized covariance DataFrame or np.array
    """
    ann_ret = float(np.dot(mean_daily.values, weights) * 252)
    ann_vol = float(np.sqrt(weights.T @ cov_annual.values @ weights))
    return ann_ret, ann_vol

def min_variance_weights_for_target(target_return, mean_daily, cov_annual):
    n = len(mean_daily)
    # objective: minimize portfolio variance
    def objective(w):
        return float(w.T @ cov_annual.values @ w)
    # constraints: sum(w)=1 and portfolio return = target_return
    cons = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'eq', 'fun': lambda w: float(np.dot(mean_daily.values, w) * 252) - target_return}
    )
    bounds = tuple((0.0, 1.0) for _ in range(n))  # long-only; change if you want leverage/shorts
    w0 = np.repeat(1.0 / n, n)
    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol':1e-9})
    if not result.success:
        raise RuntimeError("Optimization failed: " + str(result.message))
    return result.x

def efficient_frontier(mean_daily, cov_annual, returns_count=50):
    # compute target returns range
    # min return = min single-asset annualized mean, max = max single-asset annualized mean (useable bounds)
    single_ann = mean_daily * 252
    min_r = float(single_ann.min())
    max_r = float(single_ann.max())
    target_returns = np.linspace(min_r * 0.8, max_r * 1.2, returns_count)  # a wider grid
    frontier = []
    weights_list = []
    for r in target_returns:
        try:
            w = min_variance_weights_for_target(r, mean_daily, cov_annual)
            ann_r, ann_vol = portfolio_stats(w, mean_daily, cov_annual)
            frontier.append((ann_vol, ann_r))
            weights_list.append(w)
        except Exception:
            # skip infeasible targets
            continue
    frontier = np.array(frontier)
    return frontier, weights_list

def plot_efficient_frontier(prices, weights_init=None, risk_free_rate=0.02, n_random=5000, savepath="efficient_frontier.png"):
    """
    prices: DataFrame of close prices
    weights_init: initial weights (for example equal-weight) or None
    """
    returns = prices.pct_change().dropna()
    mean_daily = returns.mean()
    cov_annual = returns.cov() * 252

    n = prices.shape[1]
    # random portfolios (for background)
    rand_rets = []
    rand_vols = []
    rand_sharpes = []
    rng = np.random.default_rng(42)
    for _ in range(n_random):
        w = rng.random(n)
        w = w / w.sum()
        r, v = portfolio_stats(w, mean_daily, cov_annual)
        rand_rets.append(r)
        rand_vols.append(v)
        rand_sharpes.append((r - risk_free_rate) / v if v != 0 else np.nan)

    # efficient frontier
    frontier, weights_list = efficient_frontier(mean_daily, cov_annual, returns_count=80)

    # compute special portfolios
    # equal-weight or provided
    if weights_init is None:
        w_eq = np.repeat(1.0 / n, n)
    else:
        w_eq = np.array(weights_init)
    eq_r, eq_v = portfolio_stats(w_eq, mean_daily, cov_annual)

    # max Sharpe among random + also compute true max Sharpe via optimization (maximize (r-rf)/vol)
    # quick approximate: take best among random
    idx_best = int(np.nanargmax(rand_sharpes))
    w_best_approx = None
    if idx_best >= 0:
        # reconstruct weight from RNG? we didn't store them — instead compute max Sharpe by a constrained optimizer
        def neg_sharpe(w):
            r, v = portfolio_stats(w, mean_daily, cov_annual)
            return - (r - risk_free_rate) / v if v != 0 else 1e6
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
        bounds = tuple((0.0, 1.0) for _ in range(n))
        w0 = w_eq
        res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol':1e-9})
        if res.success:
            w_best = res.x
            best_r, best_v = portfolio_stats(w_best, mean_daily, cov_annual)
            best_sh = (best_r - risk_free_rate) / best_v
        else:
            w_best = None
            best_r = best_v = best_sh = np.nan
    else:
        w_best = None
        best_r = best_v = best_sh = np.nan

    # plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(rand_vols, rand_rets, c=rand_sharpes, cmap="viridis", alpha=0.3, label="Random portfolios")
    if len(frontier) > 0:
        # frontier is (vol, ret)
        plt.plot(frontier[:,0], frontier[:,1], color="red", linewidth=2, label="Efficient frontier")
    plt.scatter(eq_v, eq_r, marker='*', s=200, color='black', label="Equal weight" )
    if w_best is not None:
        plt.scatter(best_v, best_r, marker='X', s=150, color='orange', label=f"Max Sharpe (Sharpe={best_sh:.2f})")
    plt.xlabel("Annualized volatility")
    plt.ylabel("Annualized return")
    plt.title("Efficient Frontier and Random Portfolios")
    plt.colorbar(label="Sharpe (random portfolios)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    print(f"Efficient frontier plot saved to: {savepath}")
    plt.close()

# If you want to run automatically when executing the file, add a small trigger:
if __name__ == "__main__":
    try:
        # make sure the debug main ran already and prices variable exists; otherwise download again
        if 'prices' not in globals():
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
            prices = download_data(tickers)
        # example weights (equal)
        eq_weights = np.repeat(1.0 / prices.shape[1], prices.shape[1])
        plot_efficient_frontier(prices, weights_init=eq_weights, risk_free_rate=0.02, n_random=3000, savepath="efficient_frontier.png")
    except Exception as e:
        print("Efficient frontier plotting failed:", str(e))
        # ---------------------------
# Monte Carlo Simulation
# ---------------------------
def monte_carlo_simulation(prices, weights, num_simulations=1000, num_days=252):
    """
    Simulates future portfolio value paths using Monte Carlo.
    """
    returns = prices.pct_change().dropna()

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # initial portfolio value
    initial_value = 10000

    # store simulation results
    simulation_results = np.zeros((num_days, num_simulations))

    for sim in range(num_simulations):
        # generate random daily returns
        simulated_returns = np.random.multivariate_normal(
            mean_returns, cov_matrix, num_days
        )

        # portfolio returns
        portfolio_returns = np.dot(simulated_returns, weights)

        # cumulative returns
        portfolio_path = initial_value * np.cumprod(1 + portfolio_returns)

        simulation_results[:, sim] = portfolio_path

    return simulation_results
def monte_carlo_simulation(prices, weights, num_simulations=1000, num_days=252):
    """
    Simulates future portfolio value paths using Monte Carlo.
    Returns an array shape (num_days, num_simulations) of portfolio values.
    """
    returns = prices.pct_change().dropna()

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # initial portfolio value
    initial_value = 10000.0

    # store simulation results
    simulation_results = np.zeros((num_days, num_simulations))

    for sim in range(num_simulations):
        # generate random daily returns (multivariate)
        simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, num_days)

        # portfolio returns each day
        portfolio_returns = np.dot(simulated_returns, weights)

        # cumulative portfolio value path
        portfolio_path = initial_value * np.cumprod(1 + portfolio_returns)

        simulation_results[:, sim] = portfolio_path

    return simulation_results


def plot_monte_carlo(simulation_results, savepath="monte_carlo.png"):
    """
    Plot many simulated portfolio value paths and save the figure.
    """
    import matplotlib.pyplot as plt  # <- MUST be indented inside the function

    plt.figure(figsize=(10, 6))
    plt.plot(simulation_results, alpha=0.08)
    plt.title("Monte Carlo Simulation of Portfolio Value")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value")
    plt.grid(True)

    plt.savefig(savepath, dpi=150)
    print(f"Monte Carlo plot saved to: {savepath}")
    plt.close()
    
    # ---------------------------
# Value at Risk (VaR) from Monte Carlo
# ---------------------------
import json
import matplotlib.pyplot as plt

def compute_var_from_simulation(simulation_results, initial_value=10000, confidence_levels=(0.05, 0.01)):
    """
    Compute VaR and CVaR (ES) for provided confidence levels from simulation results.
    simulation_results: np.array shape (days, sims) representing portfolio value path
    We'll use the terminal values (last row) to compute VaR.
    Returns a dict keyed by confidence level with VaR and CVaR.
    """
    terminal = simulation_results[-1, :]  # terminal portfolio values for each simulation
    # compute returns relative to initial value
    terminal_returns = (terminal / initial_value) - 1.0

    results = {}
    for alpha in confidence_levels:
        # VaR at level alpha (e.g., alpha=0.05 -> 5% VaR) = loss that will not be exceeded with prob 1-alpha
        # We interpret VaR as a positive loss number. For example: VaR(5%) = 0.12 -> expect a 12% loss at 5% worst-case.
        threshold = np.quantile(terminal_returns, alpha)  # lower tail (negative numbers for losses)
        var = -threshold  # convert to positive loss
        # CVaR (Expected Shortfall): mean loss beyond (<=) the VaR threshold
        tail_losses = terminal_returns[terminal_returns <= threshold]
        cvar = -float(tail_losses.mean()) if tail_losses.size > 0 else None

        results[alpha] = {
            "VaR": float(var),
            "CVaR": cvar,
            "threshold_return": float(threshold),
            "num_tail_sims": int(tail_losses.size)
        }

    # Also provide some summary stats on terminal distribution
    results["summary"] = {
        "initial_value": float(initial_value),
        "terminal_mean": float(np.mean(terminal)),
        "terminal_median": float(np.median(terminal)),
        "terminal_std": float(np.std(terminal)),
        "num_simulations": int(terminal.shape[0])
    }
    return results

def plot_terminal_distribution(simulation_results, initial_value=10000, var_results=None, savepath="terminal_dist.png"):
    """
    Plot histogram of terminal portfolio values and mark VaR lines (if provided).
    var_results: dict as returned by compute_var_from_simulation
    """
    terminal = simulation_results[-1, :]
    plt.figure(figsize=(10,6))
    plt.hist(terminal, bins=80, alpha=0.8)
    plt.xlabel("Terminal portfolio value")
    plt.ylabel("Frequency")
    plt.title("Distribution of terminal portfolio values (Monte Carlo)")

    if var_results:
        for alpha, stats in var_results.items():
            if alpha == "summary":
                continue
            var_loss = stats["VaR"]
            threshold_return = stats["threshold_return"]
            var_value = initial_value * (1 + threshold_return)
            plt.axvline(var_value, color='red', linestyle='--', linewidth=2,
                        label=f"VaR {int(alpha*100)}% = ${var_value:,.0f}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    print(f"Terminal distribution plot saved to: {savepath}")
    plt.close()

def save_var_summary(results_dict, filepath="var_summary.json"):
    with open(filepath, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"VaR summary saved to: {filepath}")

# Hook into main execution: compute VaR after Monte Carlo run & plotting
if __name__ == "__main__":
    try:
        # ensure simulation exists (run it if not)
        if 'sim_results' not in globals():
            # require 'prices' and 'weights' variables exist (they are created earlier in main)
            sim_results = monte_carlo_simulation(prices, weights, num_simulations=3000, num_days=252)

        # compute VaR / CVaR using terminal values
        var_res = compute_var_from_simulation(sim_results, initial_value=10000, confidence_levels=(0.05, 0.01))
        # pretty print
        print("\nValue-at-Risk (VaR) and CVaR (Expected Shortfall) from Monte Carlo (initial_value=10000):")
        for alpha, stats in var_res.items():
            if alpha == "summary":
                continue
            pct = int(alpha*100)
            print(f"  {pct}% VaR: {stats['VaR']:.2%}  |  {pct}% CVaR: {stats['CVaR']:.2%}  | tail sims: {stats['num_tail_sims']}")

        print("\nTerminal distribution summary:")
        s = var_res["summary"]
        print(f"  mean: ${s['terminal_mean']:.2f}, median: ${s['terminal_median']:.2f}, std: ${s['terminal_std']:.2f}, sims: {s['num_simulations']}")

        # save summary JSON and distribution plot
        save_var_summary(var_res, filepath="var_summary.json")
        plot_terminal_distribution(sim_results, initial_value=10000, var_results=var_res, savepath="terminal_dist.png")

    except Exception as e:
        print("VaR computation failed:", str(e))
    