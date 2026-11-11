"""
AI Trading Cloud V2 — Angel One paper scanner
Educational starter. This script NEVER places live orders.
Run in Google Colab or a private Python environment.
"""

import getpass
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyotp
import requests
import streamlit as st
from SmartApi import SmartConnect

st.set_page_config(page_title="AI Trading Cloud V2", layout="wide")
st.title("📈 AI Trading Cloud V2 — Angel One")
st.caption("Paper trading only • No live orders • Verify all signals independently")

# Editable starter watchlist. Check the current Nifty 50 / Bank Nifty constituents
# before expanding this list. Angel One scrip master supplies current symbol tokens.
STARTER_SYMBOLS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK",
    "KOTAKBANK", "INDUSINDBK", "BANKBARODA", "FEDERALBNK",
    "TCS", "INFY", "HCLTECH", "WIPRO", "BHARTIARTL",
    "ITC", "LT", "MARUTI", "TATAMOTORS", "SUNPHARMA", "TATASTEEL"
]
NIFTY_TOKEN = "99926000"
BANKNIFTY_TOKEN = "99926009"
SCRIP_MASTER_URL = (
    "https://margincalculator.angelone.in/OpenAPI_File/"
    "files/OpenAPIScripMaster.json"
)

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Angel One API Key", type="password")
    client_code = st.text_input("Angel One Client ID")
    pin = st.text_input("Angel One PIN", type="password")
    totp_secret = st.text_input("TOTP secret (never share)", type="password")
    interval = st.selectbox("Intraday candle interval", ["FIVE_MINUTE", "FIFTEEN_MINUTE"])
    paper_capital = st.number_input("Paper capital (₹)", min_value=1000, value=10000, step=1000)
    risk_pct = st.slider("Risk per trade (%)", 0.25, 1.0, 0.5, 0.25)
    run_scan = st.button("Login & run paper scan", type="primary")

st.info(
    "This starter scans a small editable liquid-stock watchlist, not a guaranteed "
    "complete live Nifty 50 constituent list. Add/remove symbols after checking "
    "the current exchange constituents. Bank Nifty is shown as an index context."
)

def load_scrip_master():
    response = requests.get(SCRIP_MASTER_URL, timeout=30)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data)
    required = {"symbol", "token", "exch_seg"}
    if not required.issubset(df.columns):
        raise ValueError("Unexpected Angel One scrip-master format")
    df = df[(df["exch_seg"] == "NSE") & (df["symbol"].astype(str).str.endswith("-EQ"))]
    df["base_symbol"] = df["symbol"].str.replace("-EQ", "", regex=False)
    return dict(zip(df["base_symbol"], df["token"].astype(str)))

def indicators(raw):
    df = pd.DataFrame(raw, columns=["time", "open", "high", "low", "close", "volume"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("time").reset_index(drop=True)
    if len(df) < 30:
        return None

    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)
    macd = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    df["vol_avg20"] = df["volume"].rolling(20).mean()
    return df.dropna().reset_index(drop=True)

def classify(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    bullish_cross = prev["ema9"] <= prev["ema21"] and last["ema9"] > last["ema21"]
    bearish_cross = prev["ema9"] >= prev["ema21"] and last["ema9"] < last["ema21"]
    vol_ok = last["volume"] >= last["vol_avg20"] if last["vol_avg20"] > 0 else False

    if bullish_cross and last["rsi"] > 50 and last["macd"] > last["macd_signal"] and vol_ok:
        bias = "Bullish setup"
    elif bearish_cross and last["rsi"] < 50 and last["macd"] < last["macd_signal"]:
        bias = "Bearish setup"
    elif last["ema9"] > last["ema21"] and last["rsi"] >= 50:
        bias = "Bullish watch"
    elif last["ema9"] < last["ema21"] and last["rsi"] < 50:
        bias = "Bearish watch"
    else:
        bias = "No trade / mixed"

    entry = float(last["close"])
    atr = float(last["atr"])
    sl = entry - 1.5 * atr if bias.startswith("Bullish") else entry + 1.5 * atr
    target = entry + 3 * atr if bias.startswith("Bullish") else entry - 3 * atr
    return {
        "Time": str(last["time"]),
        "LTP (candle close)": round(entry, 2),
        "RSI": round(float(last["rsi"]), 1),
        "EMA9": round(float(last["ema9"]), 2),
        "EMA21": round(float(last["ema21"]), 2),
        "Volume vs avg": "High" if vol_ok else "Normal/low",
        "Bias": bias,
        "Indicative SL": round(sl, 2),
        "Indicative target": round(target, 2),
        "ATR": round(atr, 2)
    }

def telegram_send(token, chat_id, message):
    if not token or not chat_id:
        return "Telegram not configured"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=20)
    r.raise_for_status()
    return "Telegram alert sent"

if run_scan:
    if not all([api_key, client_code, pin, totp_secret]):
        st.error("Fill all Angel One credentials in the sidebar. Do not share them with anyone.")
        st.stop()

    try:
        with st.spinner("Logging in and downloading instrument tokens..."):
            smart = SmartConnect(api_key=api_key)
            otp = pyotp.TOTP(totp_secret).now()
            login = smart.generateSession(client_code, pin, otp)
            if not login or not login.get("status"):
                raise RuntimeError(f"Angel One login failed: {login}")
            tokens = load_scrip_master()

        # Use a modest first-run universe to respect API limits.
        symbols = [s for s in STARTER_SYMBOLS if s in tokens]
        missing = sorted(set(STARTER_SYMBOLS) - set(symbols))
        if not symbols:
            raise RuntimeError("No starter symbols matched Angel One scrip master.")

        now = datetime.now()
        from_dt = now - timedelta(days=5 if interval == "FIVE_MINUTE" else 20)
        rows = []
        progress = st.progress(0)

        for idx, symbol in enumerate(symbols):
            try:
                params = {
                    "exchange": "NSE",
                    "symboltoken": tokens[symbol],
                    "interval": interval,
                    "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"),
                    "todate": now.strftime("%Y-%m-%d %H:%M")
                }
                response = smart.getCandleData(params)
                if response and response.get("status") and response.get("data"):
                    calc = indicators(response["data"])
                    if calc is not None:
                        row = classify(calc)
                        row["Symbol"] = symbol
                        rows.append(row)
                time.sleep(1.1)
            except Exception as e:
                st.warning(f"{symbol}: data unavailable ({e})")
            progress.progress((idx + 1) / len(symbols))

        smart.terminateSession(client_code)

        if rows:
            out = pd.DataFrame(rows)
            out = out[["Symbol", "Time", "LTP (candle close)", "RSI", "EMA9", "EMA21",
                       "Volume vs avg", "Bias", "Indicative SL", "Indicative target", "ATR"]]
            st.subheader("Scanner results")
            st.dataframe(out, use_container_width=True)
            st.download_button(
                "Download scan CSV",
                out.to_csv(index=False).encode("utf-8"),
                file_name=f"paper_scan_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv"
            )

            bullish = out[out["Bias"].str.startswith("Bullish")]
            bearish = out[out["Bias"].str.startswith("Bearish")]
            no_trade = out[out["Bias"].str.startswith("No trade")]
            c1, c2, c3 = st.columns(3)
            c1.metric("Bullish watch", len(bullish))
            c2.metric("Bearish watch", len(bearish))
            c3.metric("No-trade / mixed", len(no_trade))

            st.caption(
                f"Paper capital configured: ₹{paper_capital:,.0f}. "
                f"Risk budget per trade at {risk_pct:.2f}%: "
                f"₹{paper_capital*risk_pct/100:,.2f}. "
                "This version scans and reports; it does not simulate fills or place orders."
            )
        else:
            st.warning("No usable candles returned. Check market hours, API access and symbol tokens.")

        if missing:
            st.caption("Symbols not found in current scrip master: " + ", ".join(missing))

    except Exception as exc:
        st.error(f"Scan failed: {exc}")
        st.info("Check API access, TOTP/PIN, internet connection and Angel One service status.")

st.divider()
st.subheader("Telegram alerts (optional)")
st.write("Create a Telegram bot with BotFather, then add the bot token and your chat ID in a private deployment. Never publish these credentials.")
st.caption("Telegram sending is not enabled in this starter UI. Add it only after testing and protecting secrets.")

st.warning(
    "Educational paper scanner only. No live order function exists in this code. "
    "Signals are rule-based indicators, not an AI prediction or investment advice. "
    "Historical candles can be delayed/incomplete; include brokerage, taxes, slippage, "
    "liquidity and gaps in any proper backtest. Verify Angel One's current API, "
    "static-IP and SEBI/NSE algo requirements before any live deployment."
)
