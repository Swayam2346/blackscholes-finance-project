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
