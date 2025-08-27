# # main.py
# #
# #
# # from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, status
# # from fastapi.responses import HTMLResponse
# # from fastapi.staticfiles import StaticFiles
# # from fastapi.templating import Jinja2Templates
# # from starlette.requests import Request
# # import pandas as pd
# # from typing import Optional, List, Dict
# # import uvicorn
# # import time
# # import asyncio
# # from fastapi.middleware.cors import CORSMiddleware
# # from pydantic import BaseModel, Field
# # import httpx  # async HTTP client
# #
# # # indicators
# # from indicators import (
# #     ALLOWED_RSI_PERIODS,
# #     parse_periods,
# #     rsi_series_values,
# #     latest_rsi_values,
# #     attach_rsi_to_candles,
# # )
# #
# # # -------------------- Config / Globals --------------------
# #
# # # in-memory cache for /exchangeInfo
# # _EXINFO_CACHE = {"data": None, "ts": 0}
# # _EXINFO_TTL = 300  # seconds
# #
# # ALLOWED_INTERVALS = {
# #     "1m","3m","5m","15m","30m",
# #     "1h","2h","4h","6h","8h","12h",
# #     "1d","3d","1w","1M"
# # }
# #
# # # polling cadence per interval (seconds) for WebSocket loop
# # POLL_SECONDS = {
# #     "1m": 2, "3m": 3, "5m": 5, "15m": 10, "30m": 15,
# #     "1h": 20, "2h": 30, "4h": 60, "6h": 90, "8h": 120,
# #     "12h": 180, "1d": 300, "3d": 600, "1w": 900, "1M": 1800,
# # }
# #
# # # FastAPI app instance
# # app = FastAPI(title="Crypto Chart API", description="Real-time cryptocurrency charting application")
# #
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["GET", "POST", "OPTIONS"],
# #     allow_headers=["*"],
# # )
# #
# # # -------------------- Models --------------------
# #
# # class CandleOut(BaseModel):
# #     time: int = Field(..., description="Epoch milliseconds (UTC)")
# #     open: float
# #     high: float
# #     low: float
# #     close: float
# #     volume: float
# #     # optional RSI attachment per candle (only present if include_rsi=true)
# #     rsi: Optional[Dict[str, Optional[float]]] = None
# #
# # # -------------------- Service --------------------
# #
# # class CryptoDataService:
# #     def __init__(self, symbol: str = "BTCUSDT"):
# #         self.symbol = symbol.upper()
# #
# #     async def validate_symbol(self) -> tuple[bool, dict]:
# #         """Validate if symbol exists on Binance (returns 24hr ticker json if OK)."""
# #         url = "https://api3.binance.com/api/v3/ticker/24hr"
# #         params = {"symbol": self.symbol}
# #         try:
# #             async with httpx.AsyncClient(timeout=5) as client:
# #                 resp = await client.get(url, params=params)
# #                 if resp.status_code == 200:
# #                     return True, resp.json()
# #                 return False, {}
# #         except httpx.RequestError:
# #             return False, {}
# #
# #     async def fetch_binance_data(self, interval: str = "15m", limit: int = 100) -> Optional[List]:
# #         """Fetch kline data from Binance API."""
# #         url = "https://api3.binance.com/api/v3/uiKlines"
# #         params = {"symbol": self.symbol, "interval": interval, "limit": limit}
# #         try:
# #             async with httpx.AsyncClient(timeout=10) as client:
# #                 resp = await client.get(url, params=params)
# #                 resp.raise_for_status()
# #                 return resp.json()
# #         except httpx.RequestError as e:
# #             print(f"Error fetching data for {self.symbol} {interval}: {e}")
# #             return None
# #
# #     def process_data(self, raw_data: List) -> Optional[pd.DataFrame]:
# #         """Convert raw Binance data to pandas DataFrame (UTC-aware timestamps)."""
# #         if not raw_data:
# #             return None
# #
# #         columns = [
# #             'open_time', 'open', 'high', 'low', 'close', 'volume',
# #             'close_time', 'quote_volume', 'count', 'taker_buy_volume',
# #             'taker_buy_quote_volume', 'ignore'
# #         ]
# #         df = pd.DataFrame(raw_data, columns=columns)
# #
# #         # Binance timestamps are UTC; keep timezone-aware
# #         df['open_time']  = pd.to_datetime(df['open_time'],  unit='ms', utc=True)
# #         df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
# #
# #         for col in ['open', 'high', 'low', 'close', 'volume']:
# #             df[col] = df[col].astype(float)
# #
# #         return df
# #
# #     @staticmethod
# #     def _to_epoch_ms(ts: pd.Timestamp) -> int:
# #         """pandas Timestamp (ns) -> epoch milliseconds (int)."""
# #         return int(ts.value // 10**6)
# #
# #     def create_rows_array(self, df: pd.DataFrame) -> List[Dict]:
# #         """Return ONLY an array of dictionaries (candles), with time in epoch ms."""
# #         if df is None or df.empty:
# #             return []
# #         return [{
# #             "time": self._to_epoch_ms(row["open_time"]),
# #             "open": float(row["open"]),
# #             "high": float(row["high"]),
# #             "low":  float(row["low"]),
# #             "close":float(row["close"]),
# #             "volume":float(row["volume"])
# #         } for _, row in df.iterrows()]
# #
# # # -------------------- REST: API ROUTES --------------------
# #
# # @app.get("/api/data", response_model=List[CandleOut])
# # async def get_chart_data(
# #     symbol: str = Query("BTCUSDT", description="Trading pair symbol"),
# #     interval: str = Query("15m", description="Chart timeframe"),
# #     limit: int = Query(150, ge=100, le=1000, description="Number of candles (100–1000)"),
# #     include_rsi: bool = Query(False, description="Attach RSI values per candle"),
# #     rsi_periods: str = Query("12", description="CSV of RSI periods; allowed: 6,8,12,14,24 (default 12)")
# # ):
# #     """
# #     Returns an array of candle dictionaries with epoch-ms timestamps.
# #     If include_rsi=true, each candle includes: "rsi": {"12": <val>, "...": ...}
# #     """
# #     if interval not in ALLOWED_INTERVALS:
# #         raise HTTPException(400, f"Invalid interval: {interval}. Allowed: {sorted(ALLOWED_INTERVALS)}")
# #
# #     periods = parse_periods(rsi_periods, default=[12])
# #
# #     svc = CryptoDataService(symbol.upper())
# #     ok, _ = await svc.validate_symbol()
# #     if not ok:
# #         raise HTTPException(400, f'Invalid symbol: {symbol}. Please check if the symbol exists on Binance.')
# #
# #     raw = await svc.fetch_binance_data(interval, limit)
# #     df = svc.process_data(raw) if raw else None
# #     if df is None or df.empty:
# #         raise HTTPException(500, f'Failed to fetch data for {symbol}')
# #
# #     if include_rsi:
# #         # keep array-of-dicts shape, but add rsi per candle
# #         candles = attach_rsi_to_candles(df, periods)
# #     else:
# #         candles = svc.create_rows_array(df)
# #
# #     return candles
# #
# # @app.get("/")
# # async def root():
# #     """Main endpoint"""
# #     return {
# #         "message": "Crypto Chart API",
# #         "endpoints": {
# #             "chart_data": "/api/data?symbol=BTCUSDT&interval=15m&limit=150",
# #             "chart_data_with_rsi": "/api/data?symbol=BTCUSDT&interval=15m&limit=150&include_rsi=true&rsi_periods=12",
# #             "search": "/api/search?q=BTC",
# #             "popular": "/api/popular",
# #             "timeframes": "/api/timeframes",
# #             "ws_data": "/ws/data?symbol=BTCUSDT&interval=15m&limit=150&include_rsi=false&rsi_periods=12"
# #         }
# #     }
# #
# # @app.get("/api/search")
# # async def search_symbols(
# #     q: Optional[str] = Query(None, description="Search query for symbols (optional)"),
# # ):
# #     """
# #     Search only USDT pairs.
# #     - If q is empty -> return ALL TRADING USDT pairs
# #     - If q is a coin name (e.g. 'BTC') -> auto-append 'USDT'
# #     """
# #     global _EXINFO_CACHE, _EXINFO_TTL
# #
# #     now = time.time()
# #     if not _EXINFO_CACHE["data"] or (now - _EXINFO_CACHE["ts"] > _EXINFO_TTL):
# #         try:
# #             url = "https://api3.binance.com/api/v3/exchangeInfo"
# #             async with httpx.AsyncClient(timeout=10) as client:
# #                 resp = await client.get(url)
# #                 resp.raise_for_status()
# #                 _EXINFO_CACHE["data"] = resp.json()
# #                 _EXINFO_CACHE["ts"] = now
# #         except httpx.RequestError:
# #             raise HTTPException(status_code=500, detail="Failed to fetch symbols from Binance")
# #
# #     data = _EXINFO_CACHE["data"]
# #
# #     # full TRADING list with only USDT quote
# #     all_trading = []
# #     for s in data.get("symbols", []):
# #         if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT":
# #             all_trading.append(s.get("symbol"))  # e.g. BTCUSDT
# #
# #     if not q or not q.strip():
# #         return {"symbols": all_trading}
# #
# #     query = q.strip().upper()
# #     if not query.endswith("USDT"):
# #         query = query + "USDT"
# #
# #     if query in all_trading:
# #         return {"symbols": [query]}
# #
# #     base = query.replace("USDT", "")
# #     hits = [sym for sym in all_trading if base in sym]
# #     if hits:
# #         return {"symbols": hits[:10]}
# #
# #     return {"message": f"{query} not available against USDT"}
# #
# # @app.get("/api/popular")
# # async def get_popular_symbols():
# #     """
# #     Return exactly top 10 USDT pairs from last 24h by quote volume.
# #     Response JSON: [{"symbol": "BTC", "price": 67000.5}, ...]
# #     """
# #     url = "https://api3.binance.com/api/v3/ticker/24hr"
# #     try:
# #         async with httpx.AsyncClient(timeout=10) as client:
# #             r = await client.get(url)
# #             r.raise_for_status()
# #             tickers = r.json()
# #     except httpx.RequestError:
# #         raise HTTPException(status_code=500, detail="Failed to fetch 24h tickers")
# #
# #     rows = []
# #     for t in tickers:
# #         sym = t.get("symbol", "")
# #         if not sym.endswith("USDT"):  # only USDT pairs
# #             continue
# #         try:
# #             rows.append({
# #                 "symbol": sym.replace("USDT", ""),  # base only
# #                 "price": float(t.get("lastPrice", 0) or 0),
# #                 "quoteVolume": float(t.get("quoteVolume", 0) or 0),
# #             })
# #         except (TypeError, ValueError):
# #             continue
# #
# #     rows.sort(key=lambda x: x["quoteVolume"], reverse=True)
# #     top10 = [{"symbol": r["symbol"], "price": r["price"]} for r in rows[:10]]
# #     return top10
# #
# # @app.get("/api/timeframes")
# # async def get_timeframes():
# #     """Return timeframes as a simple JSON list (interval + label)."""
# #     return [
# #         {"interval": "1m",  "label": "1 Minute"},
# #         {"interval": "3m",  "label": "3 Minutes"},
# #         {"interval": "5m",  "label": "5 Minutes"},
# #         {"interval": "15m", "label": "15 Minutes"},
# #         {"interval": "30m", "label": "30 Minutes"},
# #         {"interval": "1h",  "label": "1 Hour"},
# #         {"interval": "2h",  "label": "2 Hours"},
# #         {"interval": "4h",  "label": "4 Hours"},
# #         {"interval": "6h",  "label": "6 Hours"},
# #         {"interval": "8h",  "label": "8 Hours"},
# #         {"interval": "12h", "label": "12 Hours"},
# #         {"interval": "1d",  "label": "1 Day"},
# #         {"interval": "3d",  "label": "3 Days"},
# #         {"interval": "1w",  "label": "1 Week"},
# #         {"interval": "1M",  "label": "1 Month"},
# #     ]
# #
# # # -------------------- WEBSOCKET: /ws/data --------------------
# #
# # def _safe_int(value: Optional[str], default: int) -> int:
# #     try:
# #         return int(value)
# #     except Exception:
# #         return default
# #
# # @app.websocket("/ws/data")
# # async def ws_data(websocket: WebSocket):
# #     """
# #     WebSocket for streaming candles, with optional RSI attached.
# #     Query params:
# #       - symbol (default BTCUSDT)
# #       - interval (default 15m)
# #       - limit (default 150; 100–1000)
# #       - include_rsi (default false)
# #       - rsi_periods (CSV, default "12", allowed 6,8,12,14,24)
# #
# #     Control message (optional, JSON):
# #       - {"type":"set_rsi","include":true|false,"periods":[6,12,24]}
# #     """
# #     await websocket.accept()
# #
# #     # parse query params
# #     qp = websocket.query_params
# #     symbol = (qp.get("symbol") or "BTCUSDT").upper()
# #     interval = qp.get("interval") or "15m"
# #     limit = _safe_int(qp.get("limit"), 150)
# #     include_rsi = (str(qp.get("include_rsi", "false")).lower() == "true")
# #     rsi_periods = parse_periods(qp.get("rsi_periods"), default=[12])
# #
# #     # validate basic params
# #     if interval not in ALLOWED_INTERVALS:
# #         await websocket.send_json({"type": "error", "detail": f"Invalid interval: {interval}. Allowed: {sorted(ALLOWED_INTERVALS)}"})
# #         await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
# #         return
# #
# #     if limit < 100 or limit > 1000:
# #         await websocket.send_json({"type": "error", "detail": "Limit must be between 100 and 1000"})
# #         await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
# #         return
# #
# #     svc = CryptoDataService(symbol)
# #
# #     # validate symbol
# #     ok, _ = await svc.validate_symbol()
# #     if not ok:
# #         await websocket.send_json({"type": "error", "detail": f"Invalid symbol: {symbol}. Please check if the symbol exists on Binance."})
# #         await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
# #         return
# #
# #     poll = POLL_SECONDS.get(interval, 10)
# #
# #     # initial snapshot
# #     raw = await svc.fetch_binance_data(interval, limit)
# #     df = svc.process_data(raw) if raw else None
# #     if df is None or df.empty:
# #         await websocket.send_json({"type": "error", "detail": f"Failed to fetch data for {symbol}"})
# #         await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
# #         return
# #
# #     # build candles (with or without RSI)
# #     if include_rsi:
# #         candles = attach_rsi_to_candles(df, rsi_periods)
# #     else:
# #         candles = svc.create_rows_array(df)
# #
# #     await websocket.send_json({
# #         "type": "snapshot",
# #         "symbol": symbol,
# #         "interval": interval,
# #         "limit": limit,
# #         "candles": candles
# #     })
# #
# #     last_bar_time_ms = candles[-1]["time"] if candles else None
# #
# #     async def maybe_receive_control():
# #         nonlocal include_rsi, rsi_periods
# #         try:
# #             # non-blocking-ish: short timeout to check for control messages
# #             msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.01)
# #         except asyncio.TimeoutError:
# #             return
# #         except Exception:
# #             return
# #
# #         if isinstance(msg, dict) and msg.get("type") == "set_rsi":
# #             new_include = msg.get("include", include_rsi)
# #             new_periods = msg.get("periods", None)
# #             # validate periods if provided
# #             if new_periods is not None:
# #                 valid = [int(p) for p in new_periods if isinstance(p, (int, float)) or (isinstance(p, str) and str(p).isdigit())]
# #                 valid = [p for p in valid if p in ALLOWED_RSI_PERIODS]
# #                 if valid:
# #                     rsi_periods = sorted(set(valid))
# #             include_rsi = bool(new_include)
# #             await websocket.send_json({
# #                 "type": "ack",
# #                 "message": "RSI settings updated",
# #                 "include_rsi": include_rsi,
# #                 "periods": rsi_periods
# #             })
# #
# #     try:
# #         while True:
# #             await asyncio.sleep(poll)
# #
# #             # allow runtime control (toggle RSI / change periods) without reconnect
# #             await maybe_receive_control()
# #
# #             # fetch fresh small window
# #             raw2 = await svc.fetch_binance_data(interval, max(2, min(limit, 200)))
# #             df2 = svc.process_data(raw2) if raw2 else None
# #             if df2 is None or df2.empty:
# #                 continue
# #
# #             latest_arr = svc.create_rows_array(df2)
# #             latest_candle = latest_arr[-1]
# #             latest_time_ms = latest_candle["time"]
# #             is_new_bar = (last_bar_time_ms is not None and latest_time_ms > last_bar_time_ms)
# #
# #             if include_rsi:
# #                 # compute latest RSI on same df2 and attach to candle
# #                 rsi_latest = latest_rsi_values(df2, rsi_periods)
# #                 latest_candle["rsi"] = rsi_latest
# #
# #             await websocket.send_json({
# #                 "type": "update",
# #                 "symbol": symbol,
# #                 "interval": interval,
# #                 "candle": latest_candle,
# #                 "is_new_bar": bool(is_new_bar)
# #             })
# #
# #             if is_new_bar:
# #                 last_bar_time_ms = latest_time_ms
# #
# #     except WebSocketDisconnect:
# #         return
# #     except Exception as e:
# #         try:
# #             await websocket.send_json({"type": "error", "detail": f"Server error: {str(e)}"})
# #         finally:
# #             await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
# #
# # # -------------------- ENTRYPOINT --------------------
# #
# # if __name__ == "__main__":
# #     uvicorn.run(
# #         "main:app",
# #         host="127.0.0.1",
# #         port=8000,
# #         reload=True,
# #         log_level="info"
# #     )
#
#
#
#
# indicators.py
#
#
# # indicators.py
# from __future__ import annotations
# from typing import Iterable, List, Dict, Optional
# import pandas as pd
#
# ALLOWED_RSI_PERIODS = {6, 8, 12, 14, 24}
#
# def parse_periods(arg: Optional[str], default: List[int] = [12]) -> List[int]:
#     """
#     Parse CSV string like "6,12,24" into a deduped, validated list within ALLOWED_RSI_PERIODS.
#     Falls back to default if none valid.
#     """
#     if arg is None or str(arg).strip() == "":
#         return default
#     out: List[int] = []
#     for part in str(arg).split(","):
#         p = part.strip()
#         if p.isdigit():
#             n = int(p)
#             if n in ALLOWED_RSI_PERIODS and n not in out:
#                 out.append(n)
#     return out or default
#
# def rsi_wilder(close: pd.Series, period: int) -> pd.Series:
#     """
#     Wilder's RSI (EMA with alpha=1/period). Returns a pandas Series aligned with 'close'.
#     """
#     delta = close.diff()
#     up = delta.clip(lower=0)
#     down = -delta.clip(upper=0)
#     roll_up = up.ewm(alpha=1/period, adjust=False).mean()
#     roll_down = down.ewm(alpha=1/period, adjust=False).mean()
#     rs = roll_up / roll_down
#     rsi = 100 - (100 / (1 + rs))
#     return rsi
#
# def compute_rsi_dataframe(df: pd.DataFrame, periods: Iterable[int]) -> pd.DataFrame:
#     """
#     Given a DataFrame with at least a 'close' column, return a DataFrame of RSI columns:
#     rsi_<period> for each requested period.
#     """
#     out = pd.DataFrame(index=df.index)
#     for p in periods:
#         out[f"rsi_{p}"] = rsi_wilder(df["close"], p)
#     return out
#
# def rsi_series_values(df: pd.DataFrame, periods: Iterable[int]) -> Dict[str, List[Optional[float]]]:
#     """
#     Return dict of period -> list of RSI values (None for warmup indices).
#     """
#     rsi_df = compute_rsi_dataframe(df, periods)
#     result: Dict[str, List[Optional[float]]] = {}
#     for p in periods:
#         s = rsi_df[f"rsi_{p}"]
#         result[str(p)] = [None if pd.isna(v) else float(v) for v in s.tolist()]
#     return result
#
# def latest_rsi_values(df: pd.DataFrame, periods: Iterable[int]) -> Dict[str, Optional[float]]:
#     """
#     Return dict of period -> single latest RSI value (None if not enough data).
#     """
#     rsi_df = compute_rsi_dataframe(df, periods)
#     last = rsi_df.iloc[-1]
#     out: Dict[str, Optional[float]] = {}
#     for p in periods:
#         v = last.get(f"rsi_{p}")
#         out[str(p)] = None if pd.isna(v) else float(v)
#     return out
#
# def attach_rsi_to_candles(df: pd.DataFrame, periods: Iterable[int]) -> List[Dict]:
#     """
#     Build an array of candle dicts (time/open/high/low/close/volume) and attach
#     nested 'rsi' dict per item, computed on the SAME candles.
#     Keeps response shape: array of dictionaries.
#     """
#     rsi_df = compute_rsi_dataframe(df, periods)
#     # epoch ms from tz-aware timestamp
#     times_ms = (df["open_time"].astype("int64") // 10**6).astype("int64")
#     candles: List[Dict] = []
#     for idx, row in df.iterrows():
#         item = {
#             "time": int(times_ms.loc[idx]),
#             "open": float(row["open"]),
#             "high": float(row["high"]),
#             "low":  float(row["low"]),
#             "close":float(row["close"]),
#             "volume": float(row["volume"]),
#             "rsi": {str(p): (None if pd.isna(rsi_df.loc[idx, f"rsi_{p}"]) else float(rsi_df.loc[idx, f"rsi_{p}"]))
#                     for p in periods}
#         }
#         candles.append(item)
#     return candles
