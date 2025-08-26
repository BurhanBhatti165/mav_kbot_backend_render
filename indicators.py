# indicators.py
from __future__ import annotations
from typing import Iterable, List, Dict, Optional
import pandas as pd

ALLOWED_RSI_PERIODS = {6, 8, 12, 14, 24}

def parse_periods(arg: Optional[str], default: List[int] = [12]) -> List[int]:
    """
    Parse CSV string like "6,12,24" into a deduped, validated list within ALLOWED_RSI_PERIODS.
    Falls back to default if none valid.
    """
    if arg is None or str(arg).strip() == "":
        return default
    out: List[int] = []
    for part in str(arg).split(","):
        p = part.strip()
        if p.isdigit():
            n = int(p)
            if n in ALLOWED_RSI_PERIODS and n not in out:
                out.append(n)
    return out or default

def rsi_wilder(close: pd.Series, period: int) -> pd.Series:
    """
    Wilder's RSI (EMA with alpha=1/period). Returns a pandas Series aligned with 'close'.
    """
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_rsi_dataframe(df: pd.DataFrame, periods: Iterable[int]) -> pd.DataFrame:
    """
    Given a DataFrame with at least a 'close' column, return a DataFrame of RSI columns:
    rsi_<period> for each requested period.
    """
    out = pd.DataFrame(index=df.index)
    for p in periods:
        out[f"rsi_{p}"] = rsi_wilder(df["close"], p)
    return out

def rsi_series_values(df: pd.DataFrame, periods: Iterable[int]) -> Dict[str, List[Optional[float]]]:
    """
    Return dict of period -> list of RSI values (None for warmup indices).
    """
    rsi_df = compute_rsi_dataframe(df, periods)
    result: Dict[str, List[Optional[float]]] = {}
    for p in periods:
        s = rsi_df[f"rsi_{p}"]
        result[str(p)] = [None if pd.isna(v) else float(v) for v in s.tolist()]
    return result

def latest_rsi_values(df: pd.DataFrame, periods: Iterable[int]) -> Dict[str, Optional[float]]:
    """
    Return dict of period -> single latest RSI value (None if not enough data).
    """
    rsi_df = compute_rsi_dataframe(df, periods)
    last = rsi_df.iloc[-1]
    out: Dict[str, Optional[float]] = {}
    for p in periods:
        v = last.get(f"rsi_{p}")
        out[str(p)] = None if pd.isna(v) else float(v)
    return out

def attach_rsi_to_candles(df: pd.DataFrame, periods: Iterable[int]) -> List[Dict]:
    """
    Build an array of candle dicts (time/open/high/low/close/volume) and attach
    nested 'rsi' dict per item, computed on the SAME candles.
    Keeps response shape: array of dictionaries.
    """
    rsi_df = compute_rsi_dataframe(df, periods)
    # epoch ms from tz-aware timestamp
    times_ms = (df["open_time"].astype("int64") // 10**6).astype("int64")
    candles: List[Dict] = []
    for idx, row in df.iterrows():
        item = {
            "time": int(times_ms.loc[idx]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low":  float(row["low"]),
            "close":float(row["close"]),
            "volume": float(row["volume"]),
            "rsi": {str(p): (None if pd.isna(rsi_df.loc[idx, f"rsi_{p}"]) else float(rsi_df.loc[idx, f"rsi_{p}"]))
                    for p in periods}
        }
        candles.append(item)
    return candles
