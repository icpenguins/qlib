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
from typing import Any, Dict, List, Optional, Tuple, Union

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
        symbol: Optional[str] = None,
        adtv: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic option chain DataFrame with columns:
        ['strike', 'expiration', 'dte', 'option_type', 'bid', 'ask', 'lastPrice', 'impliedVolatility', 'openInterest', 'volume']
        
        Calibrated to exchange strike increments, ADTV/mega-cap liquidity scaling,
        and asymmetric open interest distributions (OTM calls above spot, OTM puts below spot).
        """
        np.random.seed(seed)
        spot = float(spot_price)
        vol = max(0.10, min(0.80, float(annual_vol)))
        t_years = max(1.0 / 365.0, dte_days / 365.0)

        # Standard discrete exchange strike increments
        if spot >= 250.0:
            step = 5.0
        elif spot >= 100.0:
            step = 2.5
        elif spot >= 25.0:
            step = 1.0
        else:
            step = 0.5

        center_strike = round(spot / step) * step
        half = num_strikes // 2
        raw_strikes = [center_strike + (i - half) * step for i in range(num_strikes)]
        strikes = np.unique([round(k, 2) for k in raw_strikes if k > 0])

        # Dynamic liquidity scaling by ADTV or Mega-Cap universe
        mega_caps = {"MSFT", "NVDA", "AAPL", "AMZN", "GOOGL", "GOOG", "META", "TSLA"}
        sym_clean = str(symbol).upper() if symbol else ""
        is_mega = (sym_clean in mega_caps) or (adtv is not None and adtv >= 10_000_000.0)

        if adtv is not None and adtv > 0:
            eff_adtv = float(adtv)
        elif is_mega:
            eff_adtv = 20_000_000.0
        else:
            eff_adtv = 1_500_000.0

        base_oi_scale = max(2500.0, (eff_adtv * 0.08) / max(1, len(strikes)))

        records = []
        exp_date = (datetime.date.today() + datetime.timedelta(days=dte_days)).strftime("%Y-%m-%d")

        for K in strikes:
            # Moneyness m = ln(K / S)
            m = math.log(K / spot)
            
            # Volatility Skew: Puts (m < 0) trade at higher IV; Calls (m > 0) trade at lower IV
            iv_call = max(0.08, vol - 0.10 * m + 0.20 * (m ** 2))
            iv_put = max(0.08, vol - 0.18 * m + 0.25 * (m ** 2))

            # Asymmetric pinning:
            # Calls have heightened open interest at round strikes ABOVE spot (covered call resistance & speculative calls)
            # Puts have heightened open interest at round strikes BELOW spot (downside portfolio hedging)
            call_pin = 2.5 if (K % 10 == 0 and K > spot) else (1.8 if (K % 5 == 0 and K > spot) else (1.2 if (K % 10 == 0) else 1.0))
            put_pin = 2.5 if (K % 10 == 0 and K < spot) else (1.8 if (K % 5 == 0 and K < spot) else (1.2 if (K % 10 == 0) else 1.0))

            # Call OI centered OTM above spot (m ~ +0.035 to +0.045)
            # Put OI centered OTM below spot (m ~ -0.045 to -0.065)
            base_call_oi = base_oi_scale * 0.85 * math.exp(-0.5 * ((m - 0.035) / 0.055) ** 2) * call_pin
            base_put_oi = base_oi_scale * 1.15 * math.exp(-0.5 * ((m + 0.045) / 0.065) ** 2) * put_pin

            call_oi = int(max(50, round(base_call_oi + np.random.normal(0, base_oi_scale * 0.04))))
            put_oi = int(max(50, round(base_put_oi + np.random.normal(0, base_oi_scale * 0.04))))

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
                "volume": int(round(call_oi * 0.25)),
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
                "volume": int(round(put_oi * 0.25)),
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
        adtv: Optional[float] = None,
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
            symbol=symbol,
            adtv=adtv,
        )
        return df_syn, True

    def load_or_generate_chain(
        self,
        symbol: str,
        spot: float,
        realized_vol_21d: float = 0.25,
        r: float = 0.045,
        auto_download: bool = True,
        adtv: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Convenience method to retrieve option chain DataFrame directly.
        """
        df, _ = self.get_options_chain(
            symbol=symbol,
            spot_price=spot,
            annual_vol=realized_vol_21d,
            auto_download=auto_download,
            adtv=adtv,
        )
        return df

    def _download_yahoo_native(self, sym: str, max_expirations: int = 4) -> List[Dict[str, Any]]:
        """
        Download option chains directly from Yahoo Finance v7 API using session cookies and crumb,
        aggregating multiple expirations (front 45 days) to capture true institutional open interest.
        """
        try:
            import requests
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })
            session.get("https://fc.yahoo.com", timeout=6)
            r_crumb = session.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=6)
            if r_crumb.status_code != 200:
                return []
            crumb = r_crumb.text.strip()
            if not crumb:
                return []

            url = f"https://query2.finance.yahoo.com/v7/finance/options/{sym}?crumb={crumb}"
            resp = session.get(url, timeout=6)
            if resp.status_code != 200:
                return []
            res_list = resp.json().get("optionChain", {}).get("result", [])
            if not res_list:
                return []
            res = res_list[0]
            exp_dates = res.get("expirationDates", [])
            if not exp_dates:
                return []

            today = datetime.date.today()
            records = []
            for exp_ts in exp_dates[:max_expirations]:
                exp_str = datetime.date.fromtimestamp(exp_ts).strftime("%Y-%m-%d")
                dte = max(1, (datetime.date.fromtimestamp(exp_ts) - today).days)
                url_exp = f"https://query2.finance.yahoo.com/v7/finance/options/{sym}?date={exp_ts}&crumb={crumb}"
                r_exp = session.get(url_exp, timeout=6)
                if r_exp.status_code != 200:
                    continue
                opts_list = r_exp.json().get("optionChain", {}).get("result", [])[0].get("options", [])
                if not opts_list:
                    continue
                opt_data = opts_list[0]
                calls = opt_data.get("calls", [])
                puts = opt_data.get("puts", [])
                for c in calls:
                    records.append({
                        "strike": float(c["strike"]),
                        "expiration": exp_str,
                        "dte": dte,
                        "option_type": "call",
                        "bid": float(c.get("bid", 0.0) or 0.0),
                        "ask": float(c.get("ask", 0.0) or 0.0),
                        "lastPrice": float(c.get("lastPrice", 0.0) or 0.0),
                        "impliedVolatility": float(c.get("impliedVolatility", 0.25) or 0.25),
                        "openInterest": int(c.get("openInterest", 0) or 0),
                        "volume": int(c.get("volume", 0) or 0),
                    })
                for p in puts:
                    records.append({
                        "strike": float(p["strike"]),
                        "expiration": exp_str,
                        "dte": dte,
                        "option_type": "put",
                        "bid": float(p.get("bid", 0.0) or 0.0),
                        "ask": float(p.get("ask", 0.0) or 0.0),
                        "lastPrice": float(p.get("lastPrice", 0.0) or 0.0),
                        "impliedVolatility": float(p.get("impliedVolatility", 0.25) or 0.25),
                        "openInterest": int(p.get("openInterest", 0) or 0),
                        "volume": int(p.get("volume", 0) or 0),
                    })
            return records
        except Exception as e:
            logger.debug(f"Direct Yahoo options download encountered error: {e}")
            return []

    def download_and_cache(
        self,
        symbol: str,
        target_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Download live option chain using yfinance or native Yahoo HTTP endpoint and save to CSV.
        """
        sym = symbol.upper()
        out_dir = Path(target_dir).expanduser().resolve() if target_dir else (self.data_dir / "options" if self.data_dir else Path("options"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{sym}_options.csv"

        records: List[Dict[str, Any]] = []

        # 1. Attempt using yfinance if available
        try:
            import yfinance as yf
            ticker = yf.Ticker(sym)
            expirations = ticker.options
            if expirations:
                today = datetime.date.today()
                for exp in expirations[:4]:  # Front 4 expirations
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
        except ImportError:
            logger.debug("yfinance not installed; trying native Yahoo HTTP downloader.")
        except Exception as e:
            logger.debug(f"yfinance options download failed for {sym}: {e}")

        # 2. Fallback to native Yahoo API if yfinance produced no records
        if not records:
            records = self._download_yahoo_native(sym, max_expirations=4)

        if records:
            df = pd.DataFrame(records)
            df.to_csv(out_file, index=False)
            logger.info(f"Downloaded and cached {len(df)} live option contracts for {sym} to {out_file}")
            return df

        return None

    def _validate_chain_df(self, df: pd.DataFrame) -> bool:
        """Check that DataFrame has required columns."""
        required = {"strike", "option_type", "openInterest"}
        return required.issubset(df.columns) and len(df) > 0
