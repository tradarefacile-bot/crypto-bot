import os
import json
from datetime import datetime
import asyncio
import pandas as pd
import numpy as np
import requests
import hashlib
import hmac
import base64
import time
import urllib.parse
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CallbackQueryHandler, ContextTypes
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
KRAKEN_API_KEY    = os.environ["KRAKEN_API_KEY"]
KRAKEN_API_SECRET = os.environ["KRAKEN_API_SECRET"]
ACCOUNT_BALANCE  = float(os.environ.get("ACCOUNT_BALANCE", "500"))

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "AVAXUSDT", "MATICUSDT", "LINKUSDT", "ADAUSDT"
]

# Mappa simboli → Kraken
KRAKEN_SYMBOL_MAP = {
    "BTCUSDT": "XBTUSD", "ETHUSDT": "ETHUSD", "SOLUSDT": "SOLUSD",
    "BNBUSDT": "BNBUSD", "XRPUSDT": "XRPUSD", "DOGEUSDT": "DOGEUSD",
    "AVAXUSDT": "AVAXUSD", "MATICUSDT": "POLUSD", "LINKUSDT": "LINKUSD",
    "ADAUSDT": "ADAUSD"
}

# Mappa simboli → coppia Kraken per ordini
KRAKEN_PAIR_MAP = {
    "BTCUSDT": "XBTUSD", "ETHUSDT": "ETHUSD", "SOLUSDT": "SOLUSD",
    "BNBUSDT": "BNBUSD", "XRPUSDT": "XRPUSD", "DOGEUSDT": "DOGEUSD",
    "AVAXUSDT": "AVAXUSD", "MATICUSDT": "POLUSD", "LINKUSDT": "LINKUSD",
    "ADAUSDT": "ADAUSD"
}

INTERVAL         = "15"
EMA_FAST         = 9
EMA_SLOW         = 21
VOLUME_MULT      = 2.0
RISK_PCT         = 0.02
SCAN_INTERVAL    = 60
ATR_PERIOD       = 14
ATR_SL_MULT      = 1.5
ATR_TP_MULT      = 3.0
MONITOR_INTERVAL = 30

TRADES_FILE = os.environ.get("TRADES_FILE", "/data/trades.json")

pending_orders: dict[str, dict] = {}
open_positions: dict[str, dict] = {}


# ── Kraken API helpers ────────────────────────────────────────────────────────

def kraken_signature(urlpath, data, secret):
    postdata = urllib.parse.urlencode(data)
    encoded = (str(data['nonce']) + postdata).encode()
    message = urlpath.encode() + hashlib.sha256(encoded).digest()
    mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()


def kraken_request(uri_path, data):
    headers = {
        'API-Key': KRAKEN_API_KEY,
        'API-Sign': kraken_signature(uri_path, data, KRAKEN_API_SECRET)
    }
    resp = requests.post(
        'https://api.kraken.com' + uri_path,
        headers=headers, data=data, timeout=10
    )
    return resp.json()


def load_trades():
    try:
        with open(TRADES_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def save_trade(symbol, signal, entry, exit_price, qty, reason):
    try:
        os.makedirs(os.path.dirname(TRADES_FILE), exist_ok=True)
        trades = load_trades()
        pnl = (exit_price - entry) * qty if signal == "BUY" else (entry - exit_price) * qty
        trades.append({
            "symbol": symbol, "signal": signal,
            "entry_price": entry, "exit_price": exit_price,
            "qty": qty, "pnl": round(pnl, 6),
            "reason": reason,
            "date": datetime.now().strftime("%d/%m %H:%M")
        })
        with open(TRADES_FILE, "w") as f:
            json.dump(trades, f)
    except Exception as e:
        logger.error(f"Errore save_trade: {e}")


def get_balance() -> float:
    try:
        data = {"nonce": str(int(time.time() * 1000))}
        resp = kraken_request("/0/private/Balance", data)
        if resp.get("error"):
            logger.error(f"Kraken balance error: {resp['error']}")
            return ACCOUNT_BALANCE
        balances = resp.get("result", {})
        usdt = float(balances.get("USDT", balances.get("USD", 0)))
        logger.info(f"Saldo Kraken: ${usdt:.2f}")
        return usdt if usdt > 0 else ACCOUNT_BALANCE
    except Exception as e:
        logger.error(f"Errore get_balance: {e}")
        return ACCOUNT_BALANCE


def get_klines(symbol: str) -> pd.DataFrame | None:
    try:
        kraken_symbol = KRAKEN_SYMBOL_MAP.get(symbol)
        if not kraken_symbol:
            return None
        interval_map = {"1": 1, "3": 3, "5": 5, "15": 15, "30": 30, "60": 60, "120": 120, "240": 240}
        kraken_interval = interval_map.get(INTERVAL, 15)
        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": kraken_symbol, "interval": kraken_interval}
        resp = requests.get(url, params=params, timeout=10)
        result = resp.json()
        if result.get("error"):
            logger.error(f"Kraken klines error {symbol}: {result['error']}")
            return None
        pair_key = list(result["result"].keys())[0]
        data = result["result"][pair_key]
        if not data:
            return None
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        logger.error(f"Errore get_klines {symbol}: {e}")
        return None


def calc_atr(df: pd.DataFrame) -> float:
    high = df["high"]
    low  = df["low"]
    close_prev = df["close"].shift(1)
    tr = pd.concat([(high - low), (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
    return tr.rolling(ATR_PERIOD).mean().iloc[-1]


def check_signal(df: pd.DataFrame) -> str | None:
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["vol_ma"]   = df["volume"].rolling(20).mean()
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    volume_spike = curr["volume"] > df["vol_ma"].iloc[-1] * VOLUME_MULT
    if prev["ema_fast"] < prev["ema_slow"] and curr["ema_fast"] > curr["ema_slow"] and volume_spike:
        return "BUY"
    if prev["ema_fast"] > prev["ema_slow"] and curr["ema_fast"] < curr["ema_slow"] and volume_spike:
        return "SELL"
    return None


def get_current_price(symbol: str) -> float | None:
    try:
        kraken_symbol = KRAKEN_SYMBOL_MAP.get(symbol)
        if not kraken_symbol:
            return None
        url = "https://api.kraken.com/0/public/Ticker"
        resp = requests.get(url, params={"pair": kraken_symbol}, timeout=10)
        result = resp.json()
        pair_key = list(result["result"].keys())[0]
        return float(result["result"][pair_key]["c"][0])
    except Exception as e:
        logger.error(f"Errore get_current_price {symbol}: {e}")
        return None


def execute_order(symbol, signal, price):
    balance = get_balance()
    order_usdt = round(balance * RISK_PCT, 2)
    qty = round(order_usdt / price, 6)
    pair = KRAKEN_PAIR_MAP.get(symbol, symbol)
    side = "buy" if signal == "BUY" else "sell"
    try:
        data = {
            "nonce": str(int(time.time() * 1000)),
            "ordertype": "market",
            "type": side,
            "volume": str(qty),
            "pair": pair,
        }
        resp = kraken_request("/0/private/AddOrder", data)
        logger.info(f"Kraken order response: {resp}")
        if resp.get("error"):
            return {"error": str(resp["error"])}, order_usdt
        return resp, order_usdt
    except Exception as e:
        logger.error(f"Errore ordine {symbol}: {e}")
        return {"error": str(e)}, order_usdt


def close_position(symbol, signal, qty):
    pair = KRAKEN_PAIR_MAP.get(symbol, symbol)
    close_side = "sell" if signal == "BUY" else "buy"
    try:
        data = {
            "nonce": str(int(time.time() * 1000)),
            "ordertype": "market",
            "type": close_side,
            "volume": str(qty),
            "pair": pair,
        }
        resp = kraken_request("/0/private/AddOrder", data)
        if resp.get("error"):
            return {"error": str(resp["error"])}
        return resp
    except Exception as e:
        return {"error": str(e)}


async def send_signal(app, symbol, signal, price, sl, tp):
    emoji = "🟢" if signal == "BUY" else "🔴"
    balance = get_balance()
    order_usdt = round(balance * RISK_PCT, 2)
    text = (
        f"{emoji} *{signal} Signal — {symbol}*\n\n"
        f"Prezzo: `${price:,.4f}`\n"
        f"Stop Loss: `${sl:,.4f}`\n"
        f"Take Profit: `${tp:,.4f}`\n"
        f"Timeframe: {INTERVAL}m | ATR x{ATR_SL_MULT}/{ATR_TP_MULT}\n"
        f"Ordine: ~${order_usdt} USDT (2% di ${balance:.0f})\n\n"
        f"Vuoi eseguire questo trade?"
    )
    order_id = f"{symbol}_{signal}_{int(asyncio.get_event_loop().time())}"
    pending_orders[order_id] = {"symbol": symbol, "signal": signal, "price": price, "sl": sl, "tp": tp}
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("Esegui", callback_data=f"exec_{order_id}"),
        InlineKeyboardButton("Salta",  callback_data=f"skip_{order_id}"),
    ]])
    await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="Markdown", reply_markup=keyboard)
    logger.info(f"Segnale: {signal} {symbol} @ {price}")


async def monitor_positions(app):
    while True:
        await asyncio.sleep(MONITOR_INTERVAL)
        if not open_positions:
            continue
        for symbol, pos in list(open_positions.items()):
            price = get_current_price(symbol)
            if price is None:
                continue
            hit_sl = (pos["signal"] == "BUY" and price <= pos["sl"]) or (pos["signal"] == "SELL" and price >= pos["sl"])
            hit_tp = (pos["signal"] == "BUY" and price >= pos["tp"]) or (pos["signal"] == "SELL" and price <= pos["tp"])
            if hit_sl or hit_tp:
                reason = "STOP LOSS" if hit_sl else "TAKE PROFIT"
                resp = close_position(symbol, pos["signal"], pos["qty"])
                if "error" not in resp:
                    del open_positions[symbol]
                    save_trade(symbol, pos["signal"], pos["entry_price"], price, pos["qty"], reason)
                    pnl = (price - pos["entry_price"]) * pos["qty"]
                    if pos["signal"] == "SELL":
                        pnl = -pnl
                    emoji = "🟢" if pnl >= 0 else "🔴"
                    await app.bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=f"{reason} colpito!\n\n*{pos['signal']} {symbol}*\nEntry: `${pos['entry_price']:,.4f}`\nExit: `${price:,.4f}`\n{emoji} P&L: `${pnl:+.4f}`",
                        parse_mode="Markdown"
                    )


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action, order_id = query.data.split("_", 1)
    order = pending_orders.pop(order_id, None)
    if not order:
        await query.edit_message_text("Ordine già gestito o scaduto.")
        return
    if action == "exec":
        resp, order_usdt = execute_order(order["symbol"], order["signal"], order["price"])
        if isinstance(resp, dict) and "error" in resp:
            await query.edit_message_text(f"Errore:\n`{resp['error']}`", parse_mode="Markdown")
        else:
            qty = round(order_usdt / order["price"], 6)
            open_positions[order["symbol"]] = {
                "signal": order["signal"], "entry_price": order["price"],
                "qty": qty, "sl": order["sl"], "tp": order["tp"],
            }
            await query.edit_message_text(
                f"*Ordine eseguito su Kraken*\n\n{order['signal']} {order['symbol']} @ `${order['price']:,.4f}`\nSL: `${order['sl']:,.4f}`\nTP: `${order['tp']:,.4f}`",
                parse_mode="Markdown"
            )
    else:
        await query.edit_message_text(f"Segnale saltato: {order['signal']} {order['symbol']}")


async def scan_loop(app):
    await app.bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=f"*Bot avviato su Kraken*\nScansiono: {', '.join(SYMBOLS)}\nSL: ATR x{ATR_SL_MULT} | TP: ATR x{ATR_TP_MULT}",
        parse_mode="Markdown"
    )
    while True:
        logger.info("Scansione in corso...")
        for symbol in SYMBOLS:
            if symbol in open_positions:
                continue
            df = get_klines(symbol)
            if df is None:
                continue
            signal = check_signal(df)
            if signal:
                price = df.iloc[-1]["close"]
                atr = calc_atr(df)
                sl = price - ATR_SL_MULT * atr if signal == "BUY" else price + ATR_SL_MULT * atr
                tp = price + ATR_TP_MULT * atr if signal == "BUY" else price - ATR_TP_MULT * atr
                await send_signal(app, symbol, signal, price, sl, tp)
            await asyncio.sleep(1)
        await asyncio.sleep(SCAN_INTERVAL)


async def is_authorized(update):
    return str(update.effective_user.id) == str(TELEGRAM_CHAT_ID)

async def cmd_start(update, context):
    if not await is_authorized(update): return
    await update.message.reply_text(
        "*Crypto Signal Bot - Kraken*\n\n"
        "/saldo — saldo e prossimo ordine\n"
        "/status — posizioni aperte\n"
        "/trades — ultimi 5 trade\n"
        "/help — questo messaggio",
        parse_mode="Markdown"
    )

async def cmd_help(update, context):
    await cmd_start(update, context)

async def cmd_saldo(update, context):
    if not await is_authorized(update): return
    balance = get_balance()
    order_usdt = round(balance * RISK_PCT, 2)
    await update.message.reply_text(
        f"*Saldo Kraken*\n\nUSDT: `${balance:.2f}`\nProssimo ordine: `${order_usdt:.2f}` (2%)\nPosizioni aperte: `{len(open_positions)}`",
        parse_mode="Markdown"
    )

async def cmd_status(update, context):
    if not await is_authorized(update): return
    if not open_positions:
        await update.message.reply_text("Nessuna posizione aperta.")
        return
    msg = "*Posizioni aperte:*\n\n"
    for symbol, pos in open_positions.items():
        price = get_current_price(symbol) or 0
        pnl = (price - pos["entry_price"]) * pos["qty"]
        if pos["signal"] == "SELL": pnl = -pnl
        emoji = "🟢" if pnl >= 0 else "🔴"
        msg += f"{emoji} *{pos['signal']} {symbol}*\nEntry: `${pos['entry_price']:,.4f}` → `${price:,.4f}`\nP&L: `${pnl:+.4f}`\n\n"
    await update.message.reply_text(msg, parse_mode="Markdown")

async def cmd_trades(update, context):
    if not await is_authorized(update): return
    trades = load_trades()
    if not trades:
        await update.message.reply_text("Nessun trade ancora.")
        return
    last5 = trades[-5:][::-1]
    total_pnl = sum(t["pnl"] for t in trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    msg = "*Ultimi trade:*\n\n"
    for t in last5:
        emoji = "🟢" if t["pnl"] >= 0 else "🔴"
        msg += f"{emoji} {t['signal']} {t['symbol']} `${t['pnl']:+.4f}` {t['date']}\n"
    msg += f"\nP&L Totale: `${total_pnl:+.4f}` | Win Rate: `{wins}/{len(trades)}`"
    await update.message.reply_text(msg, parse_mode="Markdown")


async def post_init(app: Application) -> None:
    asyncio.create_task(scan_loop(app))
    asyncio.create_task(monitor_positions(app))

def main() -> None:
    from telegram.ext import CommandHandler
    app = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("help",   cmd_help))
    app.add_handler(CommandHandler("saldo",  cmd_saldo))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("trades", cmd_trades))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
