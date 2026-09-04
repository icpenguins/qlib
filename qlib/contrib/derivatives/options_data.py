# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Options Data Loader & Downloader
================================
Provides robust ingestion, local caching, and fallback generation of equity option chains.
Supports:
1. Local CSV/Parquet option chain files (e.g. <data_dir>/options/<SYMBOL>_options.csv).
2. Automated download via yfinance (if available) with local disk caching.
3. Deterministic Synthetic Option Surface Generator calibrated to spot price,
   BOCD 21-day realized volatility, and empirical put-call skew when external data is unavailable.
"""

import os
import sys
import json
import math
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("OptionsDataLoader")


class SyntheticOptionSurfaceGenerator:
    """
    Generates a realistic, deterministic synthetic option chain calibrated to:
    - Current spot price S
    - Realized or implied volatility sigma
    - Empirical volatility skew (higher IV for OTM puts, lower for OTM calls)
    - Standard expiration cycles (Front-month ~15-30 DTE, Next-month ~45-60 DTE)
    - Log-normal open interest distribution clustered around ATM strikes and round-number psychological pins.
    """

    @staticmethod
    def generate_synthetic_chain(
        spot_price: float,
        annual_vol: float = 0.25,
        dte_days: int = 30,
        num_strikes: int = 25,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate synthetic option chain DataFrame with columns:
        ['strike', 'expiration', 'dte', 'option_type', 'bid', 'ask', 'lastPrice', 'impliedVolatility', 'openInterest', 'volume']
        """
        np.random.seed(seed)
        spot = float(spot_price)
        vol = max(0.10, min(0.80, float(annual_vol)))
        t_years = max(1.0 / 365.0, dte_days / 365.0)

        # Strike grid spanning +/- 20% around spot
        step = round(max(1.0, spot * 0.015), 1)
        # Round spot to nearest step for center
        center_strike = round(spot / step) * step
        strikes = np.linspace(center_strike - 12 * step, center_strike + 12 * step, num_strikes)
        strikes = np.unique(np.round(strikes, 2))

        records = []
        exp_date = (datetime.date.today() + datetime.timedelta(days=dte_days)).strftime("%Y-%m-%d")

        for K in strikes:
            # Moneyness m = ln(K / S)
            m = math.log(K / spot)
            
            # Volatility Skew: Puts (m < 0) trade at higher IV; Calls (m > 0) trade at lower IV
            # Skew function: sigma(m) = vol - 0.15 * m + 0.30 * m^2
            iv_call = max(0.08, vol - 0.10 * m + 0.20 * (m ** 2))
            iv_put = max(0.08, vol - 0.18 * m + 0.25 * (m ** 2))

            # Open Interest model: Gaussian bell centered slightly below spot (retail put bias)
            # Round numbers (e.g. divisible by 5 or 10) have heightened open interest (gamma pins)
            pin_multiplier = 2.5 if (K % 10 == 0) else (1.6 if (K % 5 == 0) else 1.0)
            
            base_call_oi = 500.0 * math.exp(-0.5 * ((m - 0.02) / 0.06) ** 2) * pin_multiplier
            base_put_oi = 650.0 * math.exp(-0.5 * ((m + 0.03) / 0.07) ** 2) * pin_multiplier

            call_oi = int(max(20, round(base_call_oi + np.random.normal(0, 30))))
            put_oi = int(max(20, round(base_put_oi + np.random.normal(0, 40))))

            # Call record
            records.append({
                "strike": float(K),
                "expiration": exp_date,
                "dte": dte_days,
                "option_type": "call",
                "bid": 0.0,
                "ask": 0.0,
                "lastPrice": 0.0,
                "impliedVolatility": round(iv_call, 4),
                "openInterest": call_oi,
                "volume": int(round(call_oi * 0.35)),
            })

            # Put record
            records.append({
                "strike": float(K),
                "expiration": exp_date,
                "dte": dte_days,
                "option_type": "put",
                "bid": 0.0,
                "ask": 0.0,
                "lastPrice": 0.0,
                "impliedVolatility": round(iv_put, 4),
                "openInterest": put_oi,
                "volume": int(round(put_oi * 0.40)),
            })

        df = pd.DataFrame(records)
        return df


class OptionsDataLoader:
    """
    Manages loading and caching of equity option chains from local files or automated downloaders.
    """

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        self.data_dir = Path(data_dir).expanduser().resolve() if data_dir else None

    def get_options_chain(
        self,
        symbol: str,
        spot_price: float,
        annual_vol: float = 0.25,
        force_synthetic: bool = False,
        auto_download: bool = True,
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Retrieve option chain DataFrame for symbol.

        Returns
        -------
        Tuple[pd.DataFrame, bool]
            (df_chain, is_synthetic)
        """
        sym = symbol.upper()

        if not force_synthetic and self.data_dir is not None:
            # 1. Search local options directory
            cand_paths = [
                self.data_dir / "options" / f"{sym}_options.csv",
                self.data_dir / "options" / f"{sym}.csv",
                self.data_dir / f"{sym}_options.csv",
            ]
            for p in cand_paths:
                if p.exists() and p.is_file():
                    try:
                        df = pd.read_csv(p)
                        if self._validate_chain_df(df):
                            logger.info(f"Loaded options chain for {sym} from {p} ({len(df)} contracts).")
                            return df, False
                    except Exception as e:
                        logger.warning(f"Error reading options file {p}: {e}")

            # 2. Attempt download if auto_download is enabled
            if auto_download:
                try:
                    df = self.download_and_cache(sym, target_dir=self.data_dir / "options")
                    if df is not None and not df.empty and self._validate_chain_df(df):
                        return df, False
                except Exception as e:
                    logger.debug(f"Live option chain download failed for {sym}: {e}")

        # 3. Fallback to calibrated synthetic option surface
        logger.info(f"Generating calibrated synthetic option chain for {sym} (Spot: ${spot_price:.2f}, Vol: {annual_vol*100:.1f}%).")
        df_syn = SyntheticOptionSurfaceGenerator.generate_synthetic_chain(
            spot_price=spot_price,
            annual_vol=annual_vol,
            dte_days=30,
        )
        return df_syn, True

    def load_or_generate_chain(
        self,
        symbol: str,
        spot: float,
        realized_vol_21d: float = 0.25,
        r: float = 0.045,
        auto_download: bool = True,
    ) -> pd.DataFrame:
        """
        Convenience method to retrieve option chain DataFrame directly.
        """
        df, _ = self.get_options_chain(
            symbol=symbol,
            spot_price=spot,
            annual_vol=realized_vol_21d,
            auto_download=auto_download,
        )
        return df

    def download_and_cache(

        self,
        symbol: str,
        target_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Download live option chain using yfinance and save to CSV.
        """
        sym = symbol.upper()
        out_dir = Path(target_dir).expanduser().resolve() if target_dir else (self.data_dir / "options" if self.data_dir else Path("options"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{sym}_options.csv"

        try:
            import yfinance as yf
            ticker = yf.Ticker(sym)
            expirations = ticker.options
            if not expirations:
                logger.debug(f"No option expirations found via yfinance for {sym}.")
                return None

            # Fetch front-month and second-month (up to 45 days)
            today = datetime.date.today()
            records = []

            for exp in expirations[:3]:  # Top 3 closest expirations
                try:
                    exp_dt = datetime.datetime.strptime(exp, "%Y-%m-%d").date()
                    dte = max(1, (exp_dt - today).days)
                    chain = ticker.option_chain(exp)

                    # Calls
                    calls = chain.calls
                    if not calls.empty:
                        for _, row in calls.iterrows():
                            records.append({
                                "strike": float(row["strike"]),
                                "expiration": exp,
                                "dte": dte,
                                "option_type": "call",
                                "bid": float(row.get("bid", 0.0) or 0.0),
                                "ask": float(row.get("ask", 0.0) or 0.0),
                                "lastPrice": float(row.get("lastPrice", 0.0) or 0.0),
                                "impliedVolatility": float(row.get("impliedVolatility", 0.25) or 0.25),
                                "openInterest": int(row.get("openInterest", 0) or 0),
                                "volume": int(row.get("volume", 0) or 0),
                            })

                    # Puts
                    puts = chain.puts
                    if not puts.empty:
                        for _, row in puts.iterrows():
                            records.append({
                                "strike": float(row["strike"]),
                                "expiration": exp,
                                "dte": dte,
                                "option_type": "put",
                                "bid": float(row.get("bid", 0.0) or 0.0),
                                "ask": float(row.get("ask", 0.0) or 0.0),
                                "lastPrice": float(row.get("lastPrice", 0.0) or 0.0),
                                "impliedVolatility": float(row.get("impliedVolatility", 0.25) or 0.25),
                                "openInterest": int(row.get("openInterest", 0) or 0),
                                "volume": int(row.get("volume", 0) or 0),
                            })
                except Exception as exp_err:
                    logger.debug(f"Error fetching expiration {exp} for {sym}: {exp_err}")

            if records:
                df = pd.DataFrame(records)
                df.to_csv(out_file, index=False)
                logger.info(f"Downloaded and cached {len(df)} live option contracts for {sym} to {out_file}")
                return df
        except ImportError:
            logger.debug("yfinance not installed; cannot download live options.")
        except Exception as e:
            logger.debug(f"yfinance options download failed for {sym}: {e}")

        return None

    def _validate_chain_df(self, df: pd.DataFrame) -> bool:
        """Check that DataFrame has required columns."""
        required = {"strike", "option_type", "openInterest"}
        return required.issubset(df.columns) and len(df) > 0
