"""
data_acquisition.py
Downloads and adjusts OHLCV & fundamental data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import yfinance as yf
from jugaad_data.nse import stock_df
import requests
from io import StringIO

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
ADJUSTED_DIR = DATA_DIR / "adjusted"
CORP_ACTIONS_DIR = DATA_DIR / "corp_actions"

for d in [RAW_DIR, ADJUSTED_DIR, CORP_ACTIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================
# STEP 1: Download Raw OHLCV
# ============================

def download_ohlcv_batch(tickers, start_date="2015-01-01", end_date="2025-12-31"):
    """
    Download raw OHLCV for all tickers using yfinance.
    """
    print(f"Downloading OHLCV for {len(tickers)} tickers...")

    all_data = {}

    for i, ticker in enumerate(tickers):
        try:
            yf_ticker = f"{ticker}.NS"
            df = yf.download(
                yf_ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False  # We want raw first, then adjust manually
            )

            if len(df) > 100:
                df.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
                df['ticker'] = ticker
                df.reset_index(inplace=True)
                df.rename(columns={'Date': 'date'}, inplace=True)
                all_data[ticker] = df
                print(f"[{i+1}/{len(tickers)}] {ticker}")
            else:
                print(f"[{i+1}/{len(tickers)}] {ticker}: insufficient data")

        except Exception as e:
            print(f"[{i+1}/{len(tickers)}] {ticker}: {e}")
    
    return all_data


# ====================================
# STEP 2: Download Corporate Actions
# ====================================

def download_corporate_actions_nse(ticker):
    """
    Download corporate actions from NSE.
    Returns: DataFrame with date, action_type, ratio/value
    """

    # NSE corporate actions API
    # url = f"https://www.nseindia.com/api/corporates-corporateActions?symbol={ticker}"

    # Note: In practice, you may need to handle NSE's session cookies
    # Alternative: Manual download from https://www.nseindia.com/companies-listing/corporate-filings-actions

    # For this implementation, we'll use yfinance's actions as fallback
    try:
        stock = yf.Ticker(f"{ticker}.NS")
        actions = stock.actions   # dividends and splits

        if len(actions) == 0:
            return pd.DataFrame(columns=['date', 'action_type', 'value'])
        
        actions.reset_index(inplace=True)
        actions.rename(columns={'Date': 'date'}, inplace=True)

        # Separate splits and dividends
        corp_actions = []

        for _, row in actions.iterrows():
            if 'Stock Splits' in actions.columns and pd.notna(row.get('Stock Splits')):
                corp_actions.append({
                    'date': row['date'],
                    'action_type': 'SPLIT',
                    'value': row['Stock Splits']    # e.g: 2.0 means 2:1 split
                })

            if 'Dividends' in actions.columns and pd.notna(row.get('Dividends')):
                corp_actions.append({
                    'date': row['date'],
                    'action_type': 'DIVIDEND',
                    'value': row['DIVIDENDS']
                })

        return pd.DataFrame(corp_actions)
    
    except Exception as e:
        print(f"Failed to get corp actions for {ticker}: {e}")
        return pd.DataFrame(columns=['date', 'action_type', 'value'])
    

# =============================================
# STEP 3: Adjust Prices for Corporate Actions
# =============================================

def adjust_prices_for_corporate_actions(df, corp_actions):
    """
    Backward-adjust prices for splits and dividends

    Logic:
    - For splits: multiply all pre-split prices by split ratio
    - For dividends: multiply all pre-dividend prices by (1 - dividend/price)

    We work backwards from most recent to oldest.
    """

    if len(corp_actions) == 0:
        return df
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Sort corp actions by date descending (most recent first)
    corp_actions = corp_actions.copy()
    corp_actions['date'] = pd.to_datetime(corp_actions['date'])
    corp_actions.sort_values('date', ascending=False, inplace=True)

    # Initialize adjusted columns
    df['adj_open'] = df['open'].astype(float)
    df['adj_high'] = df['high'].astype(float)
    df['adj_low'] = df['low'].astype(float)
    df['adj_close'] = df['close'].astype(float)

    cumulative_factor = 1.0

    for _, action in corp_actions.iterrows():
        action_date = action['date']
        action_type = action['action_type']
        value = action['value']

        # Find all rows before this action date
        mask = df['date'] < action_date

        if action_type == 'SPLIT':
            # Split ratio: e.g: 2.0 means 2 shares for 1
            # Pre-split prices should be divided by ratio
            split_factor = 1.0 / value
            cumulative_factor *= split_factor

            df.loc[mask, 'adj_open'] *= split_factor
            df.loc[mask, 'adj_high'] *= split_factor
            df.loc[mask, 'adj_low'] *= split_factor
            df.loc[mask, 'adj_close'] *= split_factor

        elif action_type == 'DIVIDEND':
            # Find close price on dividend date for ratio calculation
            div_day = df[df['date'] == action_date]
            if len(div_day) > 0:
                close_price = div_day['close'].iloc[0]
                if close_price > 0:
                    div_factor = (close_price - value) / close_price
                    cumulative_factor *= div_factor

                    df.loc[mask, 'adj_open'] *= div_factor
                    df.loc[mask, 'adj_high'] *= div_factor
                    df.loc[mask, 'adj_low'] *= div_factor
                    df.loc[mask, 'adj_close'] *= div_factor
    
    # Volume adjustment: opposite direction for splits
    df['adj_volume'] = df['volume'].astype(float)
    for _, action in corp_actions.iterrows():
        action_date = action['date']
        if action['action_type'] == 'SPLIT':
            split_ratio = action['value']
            mask = df['date'] < action_date
            df.loc[mask, 'adj_volume'] *= split_ratio

    return df


# ==================================
# STEP 4: Build Adjusted Database
# ==================================

def build_adjusted_database(tickers, start_date="2015-01-01", end_date="2025-12-31"):
    """
    Full pipeline: download raw, get corp actions, adjust, save to parquet
    """

    print("Building adjusted database...")

    all_adjusted = []
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Processing {ticker}...")

        try:
            # Download raw
            yf_ticker = f"{ticker}.NS"
            raw = yf.download(yf_ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)

            if len(raw) < 100:
                print(f"Insufficient data")
                continue

            raw.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
            raw.reset_index(inplace=True)
            raw.rename(columns={'Date': 'date'}, inplace=True)
            raw['ticker'] = ticker

            # Get corporate actions
            corp_actions = download_corporate_actions_nse(ticker)
            print(f"Corp actions found: {len(corp_actions)}")

            # Adjust
            adjusted = adjust_prices_for_corporate_actions(raw, corp_actions)

            # Save individual file
            adjusted.to_parquet(ADJUSTED_DIR / f"{ticker}.parquet", index=False)

            all_adjusted.append(adjusted)
            print(f"Saved: {len(adjusted)} rows")

        except Exception as e:
            print(f"Error: {e}")

    # Combine into master database
    if all_adjusted:
        master = pd.concat(all_adjusted, ignore_index=True)
        master['date'] = pd.to_datetime(master['date'])
        master.sort_values(['ticker', 'date'], inplace=True)

        # Save as partitioned parquet for fast loading
        master.to_parquet(
            DATA_DIR / "master_adjusted_ohlcv.parquet",
            index=False,
            partition_cols=['ticker']  # Partition by ticker for fast queries
        )

        print(f"\n{'='*60}")
        print("Master Database Built")
        print(f"Total rows: {len(master):,}")
        print(f"Tickers: {master['ticker'].nunique()}")
        print(f"Date range: {master['date'].min()} to {master['date'].max()}")
        print(f"Saved to: {DATA_DIR / 'master_adjusted_ohlcv.parquet'}")

        return master
    
    return None


# ==========================
# STEP 5: Fast Data Loader
# ==========================

class DataStore:
    """
    Fast data access layer. Load any ticker in under 100ms.
    """

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = Path(data_dir)
        self.master_path = self.data_dir / "master_adjusted_ohlcv.parquet"
        self.universe_path = self.data_dir / "universe" / "point_in_time_universe.parquet"

        # Cache
        self._master = None
        self._universe = None

    def load_universe(self):
        """Load point-in-time universe."""
        if self._universe is None:
            self._universe = pd.read_parquet(self.universe_path)
            return self._universe