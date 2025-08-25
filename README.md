# Crypto Chart API (FastAPI) — Backend

Real-time cryptocurrency charting backend using **FastAPI**, **httpx (async)**, and **pandas**.  
It provides REST endpoints for historical candles and metadata, plus a WebSocket stream that pushes live updates.

---

## ✨ Features

- `/api/data` — Clean JSON candles (epoch-ms timestamps) for any Binance USDT pair.
- `/api/search` — Search/filter available **TRADING** USDT symbols from Binance exchangeInfo (cached).
- `/api/popular` — Top 10 USDT pairs by 24h quote volume.
- `/api/timeframes` — Supported intervals with labels.
- `/ws/data` — Live updates over WebSocket with initial snapshot + periodic updates.
- CORS enabled for all origins (for easy local dev with any frontend).

---

## 🧱 Tech

- Python 3.11+ recommended
- FastAPI, Starlette
- httpx (async)
- pandas
- uvicorn

---

## 🚀 Quick Start

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install -U pip
pip install fastapi uvicorn[standard] httpx pandas pydantic python-dateutil

# (Optional) If you want to pin versions, create requirements.txt from your freeze later.
# pip freeze > requirements.txt

# 3) Run the server (reload for development)
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
