# indicators.py
from __future__ import annotations
from typing import Iterable, List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

# -------------------- Allowed / Defaults --------------------

ALLOWED_RSI_PERIODS = {6, 8, 12, 14, 24}
ALLOWED_SMA_PERIODS = {5, 9, 14, 20, 21, 50, 100, 200, 233}
ALLOWED_EMA_PERIODS = {5, 9, 12, 14, 20, 21, 50, 100, 200}

# NEW: extra indicators allowed/defaults
ALLOWED_ADX_PERIODS = {7, 10, 14, 20, 24, 30}
DEFAULT_ADX = 14

DEFAULT_RSI_PERIODS = [12]
DEFAULT_SMA_PERIODS = [20, 50]
DEFAULT_EMA_PERIODS = [20, 50]
DEFAULT_MACD = (12, 26, 9)   # (fast, slow, signal)
DEFAULT_BB = (20, 2.0)       # (period, stddev)
DEFAULT_STOCH = (14, 3, 3)   # (k, d, smooth_k)

# NEW defaults
DEFAULT_SUPERTREND = (10, 3.0)     # (period, multiplier)
DEFAULT_PSAR = (0.02, 0.2)         # (step, max_step)
DEFAULT_KC = (20, 2.0)             # (period, multiplier)
DEFAULT_CCI = 20                   # (period)

# -------------------- Parsers --------------------

def parse_periods(arg: Optional[str], allowed: set[int], default: List[int]) -> List[int]:
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
    if not arg:
        return default
    try:
        k, d, smooth_k = [int(x.strip()) for x in arg.split(",")]
        if k < 1 or d < 1 or smooth_k < 1:
            return default
        return (k, d, smooth_k)
    except Exception:
        return default

# NEW: extra parsers
def parse_supertrend(arg: Optional[str], default: Tuple[int, float]=DEFAULT_SUPERTREND) -> Tuple[int, float]:
    """
    'period,multiplier' e.g. '10,3'
    constraints: period>=1, 1.0 <= multiplier <= 5.0
    """
    if not arg:
        return default
    try:
        p_str, m_str = [x.strip() for x in arg.split(",")]
        period = int(p_str)
        mult = float(m_str)
        if period < 1 or mult < 1.0 or mult > 5.0:
            return default
        return (period, mult)
    except Exception:
        return default

def parse_psar(arg: Optional[str], default: Tuple[float, float]=DEFAULT_PSAR) -> Tuple[float, float]:
    """
    'step,max' e.g. '0.02,0.2'
    constraints: 0.001 <= step <= 0.2, step <= max <= 0.5
    """
    if not arg:
        return default
    try:
        step, max_step = [float(x.strip()) for x in arg.split(",")]
        if step < 0.001 or step > 0.2 or max_step < step or max_step > 0.5:
            return default
        return (step, max_step)
    except Exception:
        return default

def parse_adx(arg: Optional[str], default: int=DEFAULT_ADX) -> int:
    """
    'period' e.g. '14' from ALLOWED_ADX_PERIODS
    """
    if not arg:
        return default
    try:
        p = int(arg.strip())
        return p if p in ALLOWED_ADX_PERIODS else default
    except Exception:
        return default

def parse_kc(arg: Optional[str], default: Tuple[int, float]=DEFAULT_KC) -> Tuple[int, float]:
    """
    'period,mult' e.g. '20,2'
    constraints: period>=1, 1.0 <= mult <= 3.0
    """
    if not arg:
        return default
    try:
        p_str, m_str = [x.strip() for x in arg.split(",")]
        period = int(p_str)
        mult = float(m_str)
        if period < 1 or mult < 1.0 or mult > 3.0:
            return default
        return (period, mult)
    except Exception:
        return default

def parse_cci(arg: Optional[str], default: int=DEFAULT_CCI) -> int:
    """
    'period' e.g. '20' (>= 5 and <= 200)
    """
    if not arg:
        return default
    try:
        p = int(arg.strip())
        if p < 5 or p > 200:
            return default
        return p
    except Exception:
        return default

# -------------------- Shared helpers --------------------

def true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    prev_close_filled = prev_close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close_filled).abs(),
        (low  - prev_close_filled).abs()
    ], axis=1).max(axis=1)
    return tr

def atr_wilder(df: pd.DataFrame, period: int) -> pd.Series:
    tr = true_range(df["high"], df["low"], df["close"])
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return atr

# -------------------- RSI --------------------

def rsi_wilder(close: pd.Series, period: int) -> pd.Series:
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
    hh = df["high"].rolling(window=k, min_periods=k).max()
    ll = df["low"].rolling(window=k, min_periods=k).min()
    # avoid division by zero
    range_ = (hh - ll).replace(0, np.nan)
    raw_k = (df["close"] - ll) / range_ * 100.0
    k_smoothed = raw_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
    d_line = k_smoothed.rolling(window=d, min_periods=d).mean()
    return pd.DataFrame({
        "stoch_k": k_smoothed,
        "stoch_d": d_line
    }, index=df.index)

# -------------------- NEW: Supertrend --------------------

def supertrend_dataframe(df: pd.DataFrame, period: int, mult: float) -> pd.DataFrame:
    atr = atr_wilder(df, period)
    hl2 = (df["high"] + df["low"]) / 2.0
    upper_basic = hl2 + mult * atr
    lower_basic = hl2 - mult * atr

    upper = upper_basic.copy()
    lower = lower_basic.copy()

    # Final bands (carry forward logic)
    for i in range(1, len(df)):
        if not pd.isna(upper.iloc[i-1]):
            upper.iloc[i] = min(upper_basic.iloc[i], upper.iloc[i-1]) if df["close"].iloc[i-1] > upper.iloc[i-1] else upper_basic.iloc[i]
        if not pd.isna(lower.iloc[i-1]):
            lower.iloc[i] = max(lower_basic.iloc[i], lower.iloc[i-1]) if df["close"].iloc[i-1] < lower.iloc[i-1] else lower_basic.iloc[i]

    # Trend direction and supertrend line
    trend = pd.Series(index=df.index, dtype="int64")
    st = pd.Series(index=df.index, dtype="float64")

    # initialize
    trend.iloc[0] = 1  # 1 up, -1 down (arbitrary start)
    st.iloc[0] = lower.iloc[0]

    for i in range(1, len(df)):
        prev_trend = trend.iloc[i-1]
        if st.iloc[i-1] == upper.iloc[i-1]:
            curr_trend = -1 if df["close"].iloc[i] > upper.iloc[i] else -1
        else:
            curr_trend = 1 if df["close"].iloc[i] >= lower.iloc[i] else -1

        # Switch logic
        if curr_trend == 1:
            st.iloc[i] = lower.iloc[i]
        else:
            st.iloc[i] = upper.iloc[i]

        # Flip on cross
        if prev_trend == -1 and df["close"].iloc[i] > upper.iloc[i-1]:
            curr_trend = 1
            st.iloc[i] = lower.iloc[i]
        elif prev_trend == 1 and df["close"].iloc[i] < lower.iloc[i-1]:
            curr_trend = -1
            st.iloc[i] = upper.iloc[i]

        trend.iloc[i] = curr_trend

    return pd.DataFrame({"supertrend": st, "supertrend_trend": trend}, index=df.index)

# -------------------- NEW: Parabolic SAR --------------------

def psar_series(high: pd.Series, low: pd.Series, step: float=0.02, max_step: float=0.2) -> pd.Series:
    """
    Classic PSAR (iterative). Returns a Series aligned to index.
    """
    length = len(high)
    psar = np.zeros(length)
    bull = True  # initial direction
    af = step
    ep = low.iloc[0]  # extreme point

    psar[0] = low.iloc[0]  # seed

    for i in range(1, length):
        prior_psar = psar[i-1]

        # Update PSAR
        if bull:
            psar[i] = prior_psar + af * (ep - prior_psar)
        else:
            psar[i] = prior_psar + af * (ep - prior_psar)

        # Ensure PSAR stays on correct side of last two candles
        if bull:
            psar[i] = min(psar[i], low.iloc[i-1], low.iloc[i-2] if i >= 2 else low.iloc[i-1])
        else:
            psar[i] = max(psar[i], high.iloc[i-1], high.iloc[i-2] if i >= 2 else high.iloc[i-1])

        # Check reversal
        if bull:
            if low.iloc[i] < psar[i]:
                bull = False
                psar[i] = ep
                af = step
                ep = high.iloc[i]
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + step, max_step)
        else:
            if high.iloc[i] > psar[i]:
                bull = True
                psar[i] = ep
                af = step
                ep = low.iloc[i]
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + step, max_step)

    return pd.Series(psar, index=high.index)

# -------------------- NEW: ADX (+DI / -DI) --------------------

def adx_dataframe(df: pd.DataFrame, period: int) -> pd.DataFrame:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    tr = true_range(df["high"], df["low"], df["close"])
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr)

    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) ) * 100
    adx = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    return pd.DataFrame({
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di
    }, index=df.index)

# -------------------- NEW: Keltner Channels --------------------

def keltner_dataframe(df: pd.DataFrame, period: int, mult: float) -> pd.DataFrame:
    ema_mid = df["close"].ewm(span=period, adjust=False, min_periods=period).mean()
    atr = atr_wilder(df, period)
    upper = ema_mid + mult * atr
    lower = ema_mid - mult * atr
    return pd.DataFrame({
        "kc_middle": ema_mid,
        "kc_upper": upper,
        "kc_lower": lower
    }, index=df.index)

# -------------------- NEW: CCI --------------------

def cci_dataframe(df: pd.DataFrame, period: int) -> pd.DataFrame:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma_tp = tp.rolling(window=period, min_periods=period).mean()
    mean_dev = (tp - sma_tp).abs().rolling(window=period, min_periods=period).mean()
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    return pd.DataFrame({"cci": cci}, index=df.index)

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
    macd_cfg: Optional[Tuple[int,int,int]] = None,
    bb_cfg: Optional[Tuple[int,float]] = None,
    stoch_cfg: Optional[Tuple[int,int,int]] = None,
    # NEW:
    supertrend_cfg: Optional[Tuple[int, float]] = None,
    psar_cfg: Optional[Tuple[float, float]] = None,
    adx_period: Optional[int] = None,
    kc_cfg: Optional[Tuple[int, float]] = None,
    cci_period: Optional[int] = None,
) -> List[Dict]:
    """
    Build array of candle dicts; attach requested indicators.
    All values are floats or None. Keys are included only when requested.
    """
    times_ms = (df["open_time"].astype("int64") // 10**6).astype("int64")

    rsi_df = compute_rsi_dataframe(df, rsi_periods) if rsi_periods else None
    sma_df = compute_sma_dataframe(df, sma_periods) if sma_periods else None
    ema_df = compute_ema_dataframe(df, ema_periods) if ema_periods else None
    macd_df = macd_dataframe(df["close"], *macd_cfg) if macd_cfg else None
    bb_df = bollinger_dataframe(df["close"], *bb_cfg) if bb_cfg else None
    stoch_df = stochastic_dataframe(df, *stoch_cfg) if stoch_cfg else None

    # NEW computes
    st_df = supertrend_dataframe(df, *supertrend_cfg) if supertrend_cfg else None
    psar_s = psar_series(df["high"], df["low"], *psar_cfg) if psar_cfg else None
    adx_df = adx_dataframe(df, adx_period) if adx_period else None
    kc_df = keltner_dataframe(df, *kc_cfg) if kc_cfg else None
    cci_df = cci_dataframe(df, cci_period) if cci_period else None

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

        # NEW attachments
        if st_df is not None:
            c["supertrend"] = {
                "value": (None if pd.isna(st_df.loc[idx, "supertrend"]) else float(st_df.loc[idx, "supertrend"])),
                "trend": (None if pd.isna(st_df.loc[idx, "supertrend_trend"]) else int(st_df.loc[idx, "supertrend_trend"]))
            }

        if psar_s is not None:
            v = psar_s.loc[idx]
            c["psar"] = None if pd.isna(v) else float(v)

        if adx_df is not None:
            c["adx"] = {
                "adx":     (None if pd.isna(adx_df.loc[idx, "adx"]) else float(adx_df.loc[idx, "adx"])),
                "plus_di": (None if pd.isna(adx_df.loc[idx, "plus_di"]) else float(adx_df.loc[idx, "plus_di"])),
                "minus_di":(None if pd.isna(adx_df.loc[idx, "minus_di"]) else float(adx_df.loc[idx, "minus_di"]))
            }

        if kc_df is not None:
            c["kc"] = {
                "middle": (None if pd.isna(kc_df.loc[idx, "kc_middle"]) else float(kc_df.loc[idx, "kc_middle"])),
                "upper":  (None if pd.isna(kc_df.loc[idx, "kc_upper"]) else float(kc_df.loc[idx, "kc_upper"])),
                "lower":  (None if pd.isna(kc_df.loc[idx, "kc_lower"]) else float(kc_df.loc[idx, "kc_lower"]))
            }

        if cci_df is not None:
            v = cci_df.loc[idx, "cci"]
            c["cci"] = None if pd.isna(v) else float(v)

        candles.append(c)

    return candles
