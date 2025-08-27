from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import pandas as pd
from typing import Optional, List, Dict, Tuple
import uvicorn
import time
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx  # async HTTP client

# indicators
from indicators import (
    # allowed/defaults
    ALLOWED_RSI_PERIODS, ALLOWED_SMA_PERIODS, ALLOWED_EMA_PERIODS,
    DEFAULT_RSI_PERIODS, DEFAULT_SMA_PERIODS, DEFAULT_EMA_PERIODS,
    DEFAULT_MACD, DEFAULT_BB, DEFAULT_STOCH,
    # parsers
    parse_periods, parse_macd, parse_bb, parse_stoch,
    # builders
    assemble_candles_with_indicators
)

# -------------------- Config / Globals --------------------

# in-memory cache for /exchangeInfo
_EXINFO_CACHE = {"data": None, "ts": 0}
_EXINFO_TTL = 300  # seconds

ALLOWED_INTERVALS = {
    "1m","3m","5m","15m","30m",
    "1h","2h","4h","6h","8h","12h",
    "1d","3d","1w","1M"
}

# polling cadence per interval (seconds) for WebSocket loop
POLL_SECONDS = {
    "1m": 2, "3m": 3, "5m": 5, "15m": 10, "30m": 15,
    "1h": 20, "2h": 30, "4h": 60, "6h": 90, "8h": 120,
    "12h": 180, "1d": 300, "3d": 600, "1w": 900, "1M": 1800,
}

# FastAPI app instance
app = FastAPI(title="Crypto Chart API", description="Real-time cryptocurrency charting application")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# -------------------- Models --------------------

class CandleOut(BaseModel):
    time: int = Field(..., description="Epoch milliseconds (UTC)")
    open: float
    high: float
    low: float
    close: float
    volume: float
    # optional indicator attachments
    rsi: Optional[Dict[str, Optional[float]]] = None
    sma: Optional[Dict[str, Optional[float]]] = None
    ema: Optional[Dict[str, Optional[float]]] = None
    macd: Optional[Dict[str, Optional[float]]] = None
    bb: Optional[Dict[str, Optional[float]]] = None
    stoch: Optional[Dict[str, Optional[float]]] = None

# -------------------- Service --------------------

class CryptoDataService:
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol.upper()

    async def validate_symbol(self) -> tuple[bool, dict]:
        url = "https://api3.binance.com/api/v3/ticker/24hr"
        params = {"symbol": self.symbol}
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(url, params=params)
                if resp.status_code == 200:
                    return True, resp.json()
                return False, {}
        except httpx.RequestError:
            return False, {}

    async def fetch_binance_data(self, interval: str = "15m", limit: int = 100) -> Optional[List]:
        url = "https://api3.binance.com/api/v3/uiKlines"
        params = {"symbol": self.symbol, "interval": interval, "limit": limit}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()
        except httpx.RequestError as e:
            print(f"Error fetching data for {self.symbol} {interval}: {e}")
            return None

    def process_data(self, raw_data: List) -> Optional[pd.DataFrame]:
        if not raw_data:
            return None
        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'count', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ]
        df = pd.DataFrame(raw_data, columns=columns)
        df['open_time']  = pd.to_datetime(df['open_time'],  unit='ms', utc=True)
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df

    @staticmethod
    def _to_epoch_ms(ts: pd.Timestamp) -> int:
        return int(ts.value // 10**6)

# -------------------- REST: API ROUTES --------------------

@app.get("/api/data", response_model=List[CandleOut])
async def get_chart_data(
    symbol: str = Query("BTCUSDT", description="Trading pair symbol"),
    interval: str = Query("15m", description="Chart timeframe"),
    limit: int = Query(150, ge=100, le=1000, description="Number of candles (100–1000)"),
    # Indicators toggles (all optional, default OFF except using defaults when turned on)
    include_rsi: bool = Query(False),
    rsi_periods: str = Query(",".join(str(p) for p in DEFAULT_RSI_PERIODS), description="e.g. 6,8,12,14,24"),
    include_sma: bool = Query(False),
    sma_periods: str = Query(",".join(str(p) for p in DEFAULT_SMA_PERIODS), description="e.g. 20,50"),
    include_ema: bool = Query(False),
    ema_periods: str = Query(",".join(str(p) for p in DEFAULT_EMA_PERIODS), description="e.g. 20,50"),
    include_macd: bool = Query(False),
    macd: str = Query(f"{DEFAULT_MACD[0]},{DEFAULT_MACD[1]},{DEFAULT_MACD[2]}", description="fast,slow,signal e.g. 12,26,9"),
    include_bb: bool = Query(False),
    bb: str = Query(f"{DEFAULT_BB[0]},{DEFAULT_BB[1]}", description="period,std e.g. 20,2"),
    include_stoch: bool = Query(False),
    stoch: str = Query(f"{DEFAULT_STOCH[0]},{DEFAULT_STOCH[1]},{DEFAULT_STOCH[2]}", description="k,d,smooth_k e.g. 14,3,3")
):
    """
    Returns an array of candle dictionaries with epoch-ms timestamps.
    If indicator flags are true, each candle includes the corresponding nested dict(s):
      - rsi:  {"12": 56.2, ...}
      - sma:  {"20": 68000.1, "50": 67555.5}
      - ema:  {"20": 67990.4, "50": 67610.7}
      - macd: {"macd": 12.3, "signal": 10.1, "hist": 2.2}
      - bb:   {"middle": 68010.2, "upper": 69000.0, "lower": 67020.4}
      - stoch:{"k": 62.5, "d": 58.1}
    """
    if interval not in ALLOWED_INTERVALS:
        raise HTTPException(400, f"Invalid interval: {interval}. Allowed: {sorted(ALLOWED_INTERVALS)}")

    # parse indicator params
    rsi_p = parse_periods(rsi_periods, ALLOWED_RSI_PERIODS, DEFAULT_RSI_PERIODS) if include_rsi else None
    sma_p = parse_periods(sma_periods, ALLOWED_SMA_PERIODS, DEFAULT_SMA_PERIODS) if include_sma else None
    ema_p = parse_periods(ema_periods, ALLOWED_EMA_PERIODS, DEFAULT_EMA_PERIODS) if include_ema else None
    macd_cfg = parse_macd(macd) if include_macd else None
    bb_cfg = parse_bb(bb) if include_bb else None
    stoch_cfg = parse_stoch(stoch) if include_stoch else None

    svc = CryptoDataService(symbol.upper())
    ok, _ = await svc.validate_symbol()
    if not ok:
        raise HTTPException(400, f'Invalid symbol: {symbol}. Please check if the symbol exists on Binance.')

    raw = await svc.fetch_binance_data(interval, limit)
    df = svc.process_data(raw) if raw else None
    if df is None or df.empty:
        raise HTTPException(500, f'Failed to fetch data for {symbol}')

    candles = assemble_candles_with_indicators(
        df,
        rsi_periods=rsi_p,
        sma_periods=sma_p,
        ema_periods=ema_p,
        macd_cfg=macd_cfg,
        bb_cfg=bb_cfg,
        stoch_cfg=stoch_cfg
    )
    return candles

@app.get("/")
async def root():
    """Main endpoint"""
    return {
        "message": "Crypto Chart API",
        "endpoints": {
            "chart_data": "/api/data?symbol=BTCUSDT&interval=15m&limit=150",
            "chart_data_with_indicators": (
                "/api/data?symbol=BTCUSDT&interval=15m&limit=150"
                "&include_rsi=true&rsi_periods=12"
                "&include_sma=true&sma_periods=20,50"
                "&include_ema=true&ema_periods=20,50"
                "&include_macd=true&macd=12,26,9"
                "&include_bb=true&bb=20,2"
                "&include_stoch=true&stoch=14,3,3"
            ),
            "search": "/api/search?q=BTC",
            "popular": "/api/popular",
            "timeframes": "/api/timeframes",
            "ws_data": (
                "/ws/data?symbol=BTCUSDT&interval=15m&limit=150"
                "&include_rsi=false&include_sma=false&include_ema=false"
                "&include_macd=false&include_bb=false&include_stoch=false"
            )
        }
    }

# ---- keep your /api/search, /api/popular, /api/timeframes from your last version unchanged ----
# (omitted here for brevity)

# -------------------- WEBSOCKET: /ws/data --------------------

def _safe_int(value: Optional[str], default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default



@app.get("/api/search")
async def search_symbols(
    q: Optional[str] = Query(None, description="Search query for symbols (optional)"),
):
    """
    Search only USDT pairs.
    - If q is empty -> return ALL TRADING USDT pairs
    - If q is a coin name (e.g. 'BTC') -> auto-append 'USDT'
    """
    global _EXINFO_CACHE, _EXINFO_TTL

    now = time.time()
    if not _EXINFO_CACHE["data"] or (now - _EXINFO_CACHE["ts"] > _EXINFO_TTL):
        try:
            url = "https://api3.binance.com/api/v3/exchangeInfo"
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                _EXINFO_CACHE["data"] = resp.json()
                _EXINFO_CACHE["ts"] = now
        except httpx.RequestError:
            raise HTTPException(status_code=500, detail="Failed to fetch symbols from Binance")

    data = _EXINFO_CACHE["data"]

    # full TRADING list with only USDT quote
    all_trading = []
    for s in data.get("symbols", []):
        if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT":
            all_trading.append(s.get("symbol"))  # e.g. BTCUSDT

    if not q or not q.strip():
        return {"symbols": all_trading}

    query = q.strip().upper()
    if not query.endswith("USDT"):
        query = query + "USDT"

    if query in all_trading:
        return {"symbols": [query]}

    base = query.replace("USDT", "")
    hits = [sym for sym in all_trading if base in sym]
    if hits:
        return {"symbols": hits[:10]}

    return {"message": f"{query} not available against USDT"}

@app.get("/api/popular")
async def get_popular_symbols():
    """
    Return exactly top 10 USDT pairs from last 24h by quote volume.
    Response JSON: [{"symbol": "BTC", "price": 67000.5}, ...]
    """
    url = "https://api3.binance.com/api/v3/ticker/24hr"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            r.raise_for_status()
            tickers = r.json()
    except httpx.RequestError:
        raise HTTPException(status_code=500, detail="Failed to fetch 24h tickers")

    rows = []
    for t in tickers:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT"):  # only USDT pairs
            continue
        try:
            rows.append({
                "symbol": sym.replace("USDT", ""),  # base only
                "price": float(t.get("lastPrice", 0) or 0),
                "quoteVolume": float(t.get("quoteVolume", 0) or 0),
            })
        except (TypeError, ValueError):
            continue

    rows.sort(key=lambda x: x["quoteVolume"], reverse=True)
    top10 = [{"symbol": r["symbol"], "price": r["price"]} for r in rows[:10]]
    return top10

@app.get("/api/timeframes")
async def get_timeframes():
    """Return timeframes as a simple JSON list (interval + label)."""
    return [
        {"interval": "1m",  "label": "1 Minute"},
        {"interval": "3m",  "label": "3 Minutes"},
        {"interval": "5m",  "label": "5 Minutes"},
        {"interval": "15m", "label": "15 Minutes"},
        {"interval": "30m", "label": "30 Minutes"},
        {"interval": "1h",  "label": "1 Hour"},
        {"interval": "2h",  "label": "2 Hours"},
        {"interval": "4h",  "label": "4 Hours"},
        {"interval": "6h",  "label": "6 Hours"},
        {"interval": "8h",  "label": "8 Hours"},
        {"interval": "12h", "label": "12 Hours"},
        {"interval": "1d",  "label": "1 Day"},
        {"interval": "3d",  "label": "3 Days"},
        {"interval": "1w",  "label": "1 Week"},
        {"interval": "1M",  "label": "1 Month"},
    ]


@app.websocket("/ws/data")
async def ws_data(websocket: WebSocket):
    """
    WebSocket for streaming candles with optional indicators.
    Query params:
      - symbol (default BTCUSDT)
      - interval (default 15m)
      - limit (default 150; 100–1000)
      - include_* flags (default false)
      - rsi_periods (CSV; allowed 6,8,12,14,24; default 12)
      - sma_periods (CSV; allowed 5,9,14,20,21,50,100,200,233; default 20,50)
      - ema_periods (CSV; allowed 5,9,12,14,20,21,50,100,200; default 20,50)
      - macd (fast,slow,signal; default 12,26,9)
      - bb (period,std; default 20,2)
      - stoch (k,d,smooth_k; default 14,3,3)

    Control message (optional, JSON) to change indicators live:
      - {"type":"set_indicators",
         "include_rsi":true,"rsi_periods":[12],
         "include_sma":true,"sma_periods":[20,50],
         "include_ema":false,
         "include_macd":true,"macd":[12,26,9],
         "include_bb":false,
         "include_stoch":true,"stoch":[14,3,3]}
    """
    await websocket.accept()

    qp = websocket.query_params
    symbol = (qp.get("symbol") or "BTCUSDT").upper()
    interval = qp.get("interval") or "15m"
    limit = _safe_int(qp.get("limit"), 150)

    include_rsi = (str(qp.get("include_rsi", "false")).lower() == "true")
    include_sma = (str(qp.get("include_sma", "false")).lower() == "true")
    include_ema = (str(qp.get("include_ema", "false")).lower() == "true")
    include_macd = (str(qp.get("include_macd", "false")).lower() == "true")
    include_bb = (str(qp.get("include_bb", "false")).lower() == "true")
    include_stoch = (str(qp.get("include_stoch", "false")).lower() == "true")

    rsi_periods = parse_periods(qp.get("rsi_periods"), ALLOWED_RSI_PERIODS, DEFAULT_RSI_PERIODS) if include_rsi else None
    sma_periods = parse_periods(qp.get("sma_periods"), ALLOWED_SMA_PERIODS, DEFAULT_SMA_PERIODS) if include_sma else None
    ema_periods = parse_periods(qp.get("ema_periods"), ALLOWED_EMA_PERIODS, DEFAULT_EMA_PERIODS) if include_ema else None
    macd_cfg = parse_macd(qp.get("macd")) if include_macd else None
    bb_cfg = parse_bb(qp.get("bb")) if include_bb else None
    stoch_cfg = parse_stoch(qp.get("stoch")) if include_stoch else None

    if interval not in ALLOWED_INTERVALS:
        await websocket.send_json({"type": "error", "detail": f"Invalid interval: {interval}. Allowed: {sorted(ALLOWED_INTERVALS)}"})
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    if limit < 100 or limit > 1000:
        await websocket.send_json({"type": "error", "detail": "Limit must be between 100 and 1000"})
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    svc = CryptoDataService(symbol)
    ok, _ = await svc.validate_symbol()
    if not ok:
        await websocket.send_json({"type": "error", "detail": f"Invalid symbol: {symbol}. Please check if the symbol exists on Binance."})
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    poll = POLL_SECONDS.get(interval, 10)

    # snapshot
    raw = await svc.fetch_binance_data(interval, limit)
    df = svc.process_data(raw) if raw else None
    if df is None or df.empty:
        await websocket.send_json({"type": "error", "detail": f"Failed to fetch data for {symbol}"})
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    candles = assemble_candles_with_indicators(
        df,
        rsi_periods=rsi_periods,
        sma_periods=sma_periods,
        ema_periods=ema_periods,
        macd_cfg=macd_cfg,
        bb_cfg=bb_cfg,
        stoch_cfg=stoch_cfg
    )

    await websocket.send_json({
        "type": "snapshot",
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "candles": candles
    })

    last_bar_time_ms = candles[-1]["time"] if candles else None

    # control message handler
    async def maybe_receive_control():
        nonlocal include_rsi, include_sma, include_ema, include_macd, include_bb, include_stoch
        nonlocal rsi_periods, sma_periods, ema_periods, macd_cfg, bb_cfg, stoch_cfg
        try:
            msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.01)
        except asyncio.TimeoutError:
            return
        except Exception:
            return

        if isinstance(msg, dict) and msg.get("type") == "set_indicators":
            if "include_rsi" in msg: include_rsi = bool(msg["include_rsi"])
            if "include_sma" in msg: include_sma = bool(msg["include_sma"])
            if "include_ema" in msg: include_ema = bool(msg["include_ema"])
            if "include_macd" in msg: include_macd = bool(msg["include_macd"])
            if "include_bb" in msg: include_bb = bool(msg["include_bb"])
            if "include_stoch" in msg: include_stoch = bool(msg["include_stoch"])

            rsi_periods = parse_periods(",".join(map(str, msg.get("rsi_periods", rsi_periods or DEFAULT_RSI_PERIODS))),
                                        ALLOWED_RSI_PERIODS, DEFAULT_RSI_PERIODS) if include_rsi else None
            sma_periods = parse_periods(",".join(map(str, msg.get("sma_periods", sma_periods or DEFAULT_SMA_PERIODS))),
                                        ALLOWED_SMA_PERIODS, DEFAULT_SMA_PERIODS) if include_sma else None
            ema_periods = parse_periods(",".join(map(str, msg.get("ema_periods", ema_periods or DEFAULT_EMA_PERIODS))),
                                        ALLOWED_EMA_PERIODS, DEFAULT_EMA_PERIODS) if include_ema else None

            if include_macd:
                macd_list = msg.get("macd", list(DEFAULT_MACD))
                macd_str = ",".join(map(str, macd_list))
                macd_cfg = parse_macd(macd_str)
            else:
                macd_cfg = None

            if include_bb:
                bb_list = msg.get("bb", list(DEFAULT_BB))
                bb_str = ",".join(map(str, bb_list))
                bb_cfg = parse_bb(bb_str)
            else:
                bb_cfg = None

            if include_stoch:
                stoch_list = msg.get("stoch", list(DEFAULT_STOCH))
                stoch_str = ",".join(map(str, stoch_list))
                stoch_cfg = parse_stoch(stoch_str)
            else:
                stoch_cfg = None

            await websocket.send_json({
                "type": "ack",
                "message": "Indicator settings updated",
                "include": {
                    "rsi": include_rsi, "sma": include_sma, "ema": include_ema,
                    "macd": include_macd, "bb": include_bb, "stoch": include_stoch
                }
            })

    try:
        while True:
            await asyncio.sleep(poll)
            await maybe_receive_control()

            raw2 = await svc.fetch_binance_data(interval, max(2, min(limit, 200)))
            df2 = svc.process_data(raw2) if raw2 else None
            if df2 is None or df2.empty:
                continue

            # build indicators on the same small window; take only last candle for update
            latest_candles = assemble_candles_with_indicators(
                df2,
                rsi_periods=rsi_periods if include_rsi else None,
                sma_periods=sma_periods if include_sma else None,
                ema_periods=ema_periods if include_ema else None,
                macd_cfg=macd_cfg if include_macd else None,
                bb_cfg=bb_cfg if include_bb else None,
                stoch_cfg=stoch_cfg if include_stoch else None
            )
            latest_candle = latest_candles[-1]
            latest_time_ms = latest_candle["time"]
            is_new_bar = (last_bar_time_ms is not None and latest_time_ms > last_bar_time_ms)

            await websocket.send_json({
                "type": "update",
                "symbol": symbol,
                "interval": interval,
                "candle": latest_candle,
                "is_new_bar": bool(is_new_bar)
            })

            if is_new_bar:
                last_bar_time_ms = latest_time_ms

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "detail": f"Server error: {str(e)}"})
        finally:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)

# -------------------- ENTRYPOINT --------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
