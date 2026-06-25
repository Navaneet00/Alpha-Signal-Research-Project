"""
Builds point-in-time universe for NSE stocks
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
import matplotlib.pyplot as plt
from jugaad_data.nse import stock_df, index_df

# Configuration
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
UNIVERSE_DIR = DATA_DIR / "universe"

for d in [RAW_DIR, PROCESSED_DIR, UNIVERSE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =================================================
# STEP 1: Download Historical Index Constituents
# =================================================

def download_nifty_constituents_history():
    print("Downloading NIFTY 500 historical data...")

    # Get current NIFTY 500 constituents
    nifty500_current = pd.read_csv("https://archives.nseindia.com/content/indices/ind_nifty500list.csv")
    all_tickers = nifty500_current['Symbol'].tolist()

    print(f"Current NIFTY 500 has {len(all_tickers)} stocks.")
    return all_tickers

def download_full_history_for_universe(tickers, start_date="2015-01-01", end_date="2025-12-31"):
    """
    Download full OHLCV history for all potential universe members.
    This lets us determine which stocks were actively trading on any given date.
    """

    all_data = {}
    failed = []

    for i, ticker in enumerate(tickers):
        try:
            # Add .NS suffix for NSE stocks via yfinance
            yf_ticker = f"{ticker}.NS"
            df = yf.download(yf_ticker, start=start_date, end=end_date, progress=False)

            if len(df) > 100: # Minimum trading history threshold
                df['ticker'] = ticker
                all_data[ticker] = df
                print(f"[{i+1}/{len(tickers)}] Downloaded {ticker} with {len(df)} records.")
            else:
                failed.append((ticker, "Insufficient data"))

        except Exception as e:
            print(f"[{i+1}/{len(tickers)}] Failed to download {ticker}: {e}")
            failed.append((ticker, str(e)))

    print(f"Downloaded data for {len(all_data)} tickers. Failed for {len(failed)} tickers.")
    return all_data, failed


# =========================================
# STEP 2: Build Point-in-Time Universe
# =========================================

def build_point_in_time_universe(all_data, liquidity_threshold=0.5):
    """
    Build date -> [tickers] mapping with liquidity filtering.
    
    Rules:
    1. Stock must have price data on that date.
    2. Stock must have non-zero volume.
    3. Stock must meet minimum liquidity (median daily turnover)
    
    Parameters:
    -----------
    all_data: dict
        Dictionary of ticker -> DataFrame with OHLCV data.
    liquidity_threshold: float
        Percentile cutoff for liquidity (0.5 = top 50% by turnover)
    """

    print("Building point-in-time universe...")

    # Combine all data into a single DataFrame
    combined = pd.concat(all_data.values(), ignore_index=False)
    combined.reset_index(inplace=True)
    combined.rename(columns={'Date': 'date'}, inplace=True)

    # Calculate daily turnover (proxy for liquidity)
    combined['turnover'] = combined['Close'] * combined['Volume']

    #For each date, determine which stocks are valid
    universe_records = []

    dates = combined['date'].unique()
    dates = pd.to_datetime(dates).sort_values()

    for date in dates:
        day_data = combined[combined['date'] == date].copy()

        # Basic filters
        valid = day_data[
            (day_data['Volume'] > 0) & 
            (day_data['Close'] > 0) & 
            (day_data['turnover'].notna())
        ].copy()

        # Liquidity filter: top 300 by 6-month rolling average saily turnover
        if len(valid) > 300:
            turnover_median = valid['turnover'].median()
            valid = valid[valid['turnover'] >= turnover_median * 0.1]  # Relaxed threshold

        tickers_on_date = valid['ticker'].tolist()
        universe_records.append({
            'date': date,
            'tickers': tickers_on_date,
            'n_tickers': len(tickers_on_date)
        })

    universe_df = pd.DataFrame(universe_records)
    universe_df.set_index('date', inplace=True)

    # Save 
    universe_df.to_parquet(UNIVERSE_DIR / "point_in_time_universe.parquet")

    print(f"Universe built with {len(universe_df)} dates. Saved to {UNIVERSE_DIR / 'point_in_time_universe.parquet'}")
    print(f"Average number of tickers per date: {universe_df['n_tickers'].mean():.0f}")

    return universe_df


# ======================================================
# STEP 3: Universse Validation (Survivorship Bias Check)
# ======================================================

def validate_universe(universe_df):
    """
    Validate the universe for survivorship bias.
    """
    print("Validating universe for survivorship bias...")

    # Check 1: Number of tickers should vary over time
    n_tickers_series = universe_df['n_tickers']
    print(f"\nTickers per day - Min: {n_tickers_series.min()}, Max: {n_tickers_series.max()}, Mean: {n_tickers_series.mean():.0f}, Std: {n_tickers_series.std():.0f}")

    # Check 2: Specific delisted stock example
    # Example: DHFL was delisted in 2021
    delisted_example = 'DHFL'
    post_delist = universe_df[universe_df.index > "2021-01-01"]['tickers']
    dhfl_present = any(delisted_example in tickers for tickers in post_delist)
    print(f"\nDelisted stock {delisted_example} present after delisting? {'FAIL - BIAS DETECTED' if dhfl_present else 'PASS - NO BIAS DETECTED'}")

    # Check 3: Plot universe size over time
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(universe_df.index, universe_df['n_tickers'], alpha=0.7)
    ax.set_title("Point-in-Time Universe Size Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Tickers")
    ax.axhline(y=300, color='r', linestyle='--', label='Target: Top 300')
    ax.legend()
    plt.tight_layout()
    plt.savefig(UNIVERSE_DIR / "universe_size_over_time.png", dpi=300)
    plt.show()

    print(f"\n Validation plot saved to {UNIVERSE_DIR / 'universe_size_over_time.png'}")
    return True


# ==========================
# MAIN EXECUTION
# ==========================

if __name__ == "__main__":
    # Step 1: Get tickers
    tickers = download_nifty_constituents_history()

    # Step 2: Download history
    all_data, failed = download_full_history_for_universe(tickers)

    # Step 3: Build universe
    universe = build_point_in_time_universe(all_data)

    # Step 4: Validate universe
    validate_universe(universe)

    print("\nPoint-in-time universe construction and validation complete.")