# indicators.py
from __future__ import annotations
from typing import Iterable, List, Dict, Optional, Tuple
import pandas as pd

# -------------------- Allowed / Defaults --------------------

ALLOWED_RSI_PERIODS = {6, 8, 12, 14, 24}
ALLOWED_SMA_PERIODS = {5, 9, 14, 20, 21, 50, 100, 200, 233}
ALLOWED_EMA_PERIODS = {5, 9, 12, 14, 20, 21, 50, 100, 200}

# sensible defaults for production
DEFAULT_RSI_PERIODS = [12]
DEFAULT_SMA_PERIODS = [20, 50]
DEFAULT_EMA_PERIODS = [20, 50]
DEFAULT_MACD = (12, 26, 9)   # (fast, slow, signal)
DEFAULT_BB = (20, 2.0)       # (period, stddev)
DEFAULT_STOCH = (14, 3, 3)   # (k, d, smooth_k)

# -------------------- Parsers --------------------

def parse_periods(arg: Optional[str], allowed: set[int], default: List[int]) -> List[int]:
    """
    Parse CSV like "6,12,24" into a deduped list filtered by 'allowed'.
    Falls back to default if nothing valid.
    """
    if arg is None or str(arg).strip() == "":
        return default
    out: List[int] = []
    for part in str(arg).split(","):
        p = part.strip()
        if p.isdigit():
            n = int(p)
            if n in allowed and n not in out:
                out.append(n)
    return out or default

def parse_macd(arg: Optional[str], default: Tuple[int,int,int]=DEFAULT_MACD) -> Tuple[int,int,int]:
    """
    Parse 'fast,slow,signal' e.g. '12,26,9'. Ensures fast<slow and all >=1.
    """
    if not arg:
        return default
    try:
        fast, slow, signal = [int(x.strip()) for x in arg.split(",")]
        if fast < 1 or slow < 1 or signal < 1 or not (fast < slow):
            return default
        return (fast, slow, signal)
    except Exception:
        return default

def parse_bb(arg: Optional[str], default: Tuple[int,float]=DEFAULT_BB) -> Tuple[int,float]:
    """
    Parse 'period,std' e.g. '20,2' with period>=1 and 0.5<=std<=4.0.
    """
    if not arg:
        return default
    try:
        p_str, s_str = [x.strip() for x in arg.split(",")]
        period = int(p_str)
        std = float(s_str)
        if period < 1 or std < 0.5 or std > 4.0:
            return default
        return (period, std)
    except Exception:
        return default

def parse_stoch(arg: Optional[str], default: Tuple[int,int,int]=DEFAULT_STOCH) -> Tuple[int,int,int]:
    """
    Parse 'k,d,smooth_k' e.g. '14,3,3' with all >=1.
    """
    if not arg:
        return default
    try:
        k, d, smooth_k = [int(x.strip()) for x in arg.split(",")]
        if k < 1 or d < 1 or smooth_k < 1:
            return default
        return (k, d, smooth_k)
    except Exception:
        return default

# -------------------- RSI --------------------

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
    out = pd.DataFrame(index=df.index)
    for p in periods:
        out[f"rsi_{p}"] = rsi_wilder(df["close"], p)
    return out

def rsi_series_values(df: pd.DataFrame, periods: Iterable[int]) -> Dict[str, List[Optional[float]]]:
    rsi_df = compute_rsi_dataframe(df, periods)
    result: Dict[str, List[Optional[float]]] = {}
    for p in periods:
        s = rsi_df[f"rsi_{p}"]
        result[str(p)] = [None if pd.isna(v) else float(v) for v in s.tolist()]
    return result

def latest_rsi_values(df: pd.DataFrame, periods: Iterable[int]) -> Dict[str, Optional[float]]:
    rsi_df = compute_rsi_dataframe(df, periods)
    last = rsi_df.iloc[-1]
    out: Dict[str, Optional[float]] = {}
    for p in periods:
        v = last.get(f"rsi_{p}")
        out[str(p)] = None if pd.isna(v) else float(v)
    return out

# -------------------- SMA / EMA --------------------

def compute_sma_dataframe(df: pd.DataFrame, periods: Iterable[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for p in periods:
        out[f"sma_{p}"] = df["close"].rolling(window=p, min_periods=p).mean()
    return out

def compute_ema_dataframe(df: pd.DataFrame, periods: Iterable[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for p in periods:
        out[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False, min_periods=p).mean()
    return out

# -------------------- MACD --------------------

def macd_dataframe(close: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": hist
    }, index=close.index)

# -------------------- Bollinger Bands --------------------

def bollinger_dataframe(close: pd.Series, period: int, std: float) -> pd.DataFrame:
    ma = close.rolling(window=period, min_periods=period).mean()
    dev = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = ma + std * dev
    lower = ma - std * dev
    return pd.DataFrame({
        "bb_middle": ma,
        "bb_upper": upper,
        "bb_lower": lower
    }, index=close.index)

# -------------------- Stochastic Oscillator --------------------

def stochastic_dataframe(df: pd.DataFrame, k: int, d: int, smooth_k: int) -> pd.DataFrame:
    # Highest high / lowest low over k
    hh = df["high"].rolling(window=k, min_periods=k).max()
    ll = df["low"].rolling(window=k, min_periods=k).min()
    # Raw %K
    raw_k = (df["close"] - ll) / (hh - ll) * 100.0
    # Smooth %K (moving average over smooth_k)
    k_smoothed = raw_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
    # %D is MA of %K over d
    d_line = k_smoothed.rolling(window=d, min_periods=d).mean()
    return pd.DataFrame({
        "stoch_k": k_smoothed,
        "stoch_d": d_line
    }, index=df.index)

# -------------------- Attachment Helpers --------------------

def attach_rsi_to_candles(df: pd.DataFrame, periods: Iterable[int]) -> List[Dict]:
    rsi_df = compute_rsi_dataframe(df, periods)
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

def assemble_candles_with_indicators(
    df: pd.DataFrame,
    *,
    rsi_periods: Optional[List[int]] = None,
    sma_periods: Optional[List[int]] = None,
    ema_periods: Optional[List[int]] = None,
    macd_cfg: Optional[Tuple[int,int,int]] = None,      # (fast, slow, signal)
    bb_cfg: Optional[Tuple[int,float]] = None,          # (period, std)
    stoch_cfg: Optional[Tuple[int,int,int]] = None      # (k, d, smooth_k)
) -> List[Dict]:
    """
    Build array of candle dicts; attach requested indicators.
    Each indicator is included ONLY if its corresponding argument is not None.
    All values are floats or None (JSON-friendly).
    """
    times_ms = (df["open_time"].astype("int64") // 10**6).astype("int64")

    # precompute frames for efficiency
    rsi_df = compute_rsi_dataframe(df, rsi_periods) if rsi_periods else None
    sma_df = compute_sma_dataframe(df, sma_periods) if sma_periods else None
    ema_df = compute_ema_dataframe(df, ema_periods) if ema_periods else None
    macd_df = macd_dataframe(df["close"], *macd_cfg) if macd_cfg else None
    bb_df = bollinger_dataframe(df["close"], *bb_cfg) if bb_cfg else None
    stoch_df = stochastic_dataframe(df, *stoch_cfg) if stoch_cfg else None

    candles: List[Dict] = []
    for idx, row in df.iterrows():
        c: Dict = {
            "time": int(times_ms.loc[idx]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low":  float(row["low"]),
            "close":float(row["close"]),
            "volume": float(row["volume"])
        }

        if rsi_df is not None:
            c["rsi"] = {
                str(p): (None if pd.isna(rsi_df.loc[idx, f"rsi_{p}"]) else float(rsi_df.loc[idx, f"rsi_{p}"]))
                for p in rsi_periods  # type: ignore
            }

        if sma_df is not None:
            c["sma"] = {
                str(p): (None if pd.isna(sma_df.loc[idx, f"sma_{p}"]) else float(sma_df.loc[idx, f"sma_{p}"]))
                for p in sma_periods  # type: ignore
            }

        if ema_df is not None:
            c["ema"] = {
                str(p): (None if pd.isna(ema_df.loc[idx, f"ema_{p}"]) else float(ema_df.loc[idx, f"ema_{p}"]))
                for p in ema_periods  # type: ignore
            }

        if macd_df is not None:
            c["macd"] = {
                "macd":   (None if pd.isna(macd_df.loc[idx, "macd"]) else float(macd_df.loc[idx, "macd"])),
                "signal": (None if pd.isna(macd_df.loc[idx, "macd_signal"]) else float(macd_df.loc[idx, "macd_signal"])),
                "hist":   (None if pd.isna(macd_df.loc[idx, "macd_hist"]) else float(macd_df.loc[idx, "macd_hist"]))
            }

        if bb_df is not None:
            c["bb"] = {
                "middle": (None if pd.isna(bb_df.loc[idx, "bb_middle"]) else float(bb_df.loc[idx, "bb_middle"])),
                "upper":  (None if pd.isna(bb_df.loc[idx, "bb_upper"]) else float(bb_df.loc[idx, "bb_upper"])),
                "lower":  (None if pd.isna(bb_df.loc[idx, "bb_lower"]) else float(bb_df.loc[idx, "bb_lower"]))
            }

        if stoch_df is not None:
            c["stoch"] = {
                "k": (None if pd.isna(stoch_df.loc[idx, "stoch_k"]) else float(stoch_df.loc[idx, "stoch_k"])),
                "d": (None if pd.isna(stoch_df.loc[idx, "stoch_d"]) else float(stoch_df.loc[idx, "stoch_d"]))
            }

        candles.append(c)

    return candles
