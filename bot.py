import os
import json
from datetime import datetime
import asyncio
import pandas as pd
import numpy as np
import requests
from pybit.unified_trading import HTTP
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CallbackQueryHandler, ContextTypes
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Config da variabili d'ambiente ──────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
BYBIT_API_KEY    = os.environ["BYBIT_API_KEY"]
BYBIT_API_SECRET = os.environ["BYBIT_API_SECRET"]

# ── Parametri strategia ──────────────────────────────────────────────────────
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "AVAXUSDT", "MATICUSDT", "LINKUSDT", "ADAUSDT"
]
INTERVAL        = "15"   # timeframe in minuti
EMA_FAST        = 9
EMA_SLOW        = 21
VOLUME_MULT     = 2.0    # spike volume: 2x la media
ORDER_USDT      = 20     # dimensione ordine in USDT
SCAN_INTERVAL   = 60     # secondi tra ogni scansione
CATEGORY        = "spot"

# ATR settings
ATR_PERIOD      = 14
ATR_SL_MULT     = 1.5   # SL = prezzo ± 1.5x ATR
ATR_TP_MULT     = 3.0   # TP = prezzo ± 3.0x ATR  (risk/reward 1:2)
MONITOR_INTERVAL = 30   # secondi tra i check SL/TP

# ── Bybit client ─────────────────────────────────────────────────────────────
session = HTTP(
    testnet=False,
    api_key=BYBIT_API_KEY,
    api_secret=BYBIT_API_SECRET,
)

# ── Pending orders (in attesa di conferma Telegram) ──────────────────────────
pending_orders: dict[str, dict] = {}

# ── Posizioni aperte da monitorare per SL/TP ─────────────────────────────────
TRADES_FILE = os.environ.get("TRADES_FILE", "/data/trades.json")

def save_trade(symbol, signal, entry, exit_price, qty, reason):
    """Salva il trade chiuso nel file JSON per il diario."""
    try:
        os.makedirs(os.path.dirname(TRADES_FILE), exist_ok=True)
        try:
            with open(TRADES_FILE) as f:
                trades = json.load(f)
        except Exception:
            trades = []
        pnl = (exit_price - entry) * qty if signal == "BUY" else (entry - exit_price) * qty
        trades.append({
            "symbol": symbol,
            "signal": signal,
            "entry_price": entry,
            "exit_price": exit_price,
            "qty": qty,
            "pnl": round(pnl, 6),
            "reason": reason,
            "date": datetime.now().strftime("%d/%m %H:%M")
        })
        with open(TRADES_FILE, "w") as f:
            json.dump(trades, f)
    except Exception as e:
        logger.error(f"Errore save_trade: {e}")

open_positions: dict[str, dict] = {}
# Struttura: { symbol: { signal, entry_price, qty, sl, tp, order_id } }


def get_klines(symbol: str) -> pd.DataFrame | None:
    """
    Scarica le candele da Kraken (API pubblica, nessun blocco geografico).
    Bybit viene usato solo per eseguire gli ordini.
    """
    try:
        symbol_map = {
            "BTCUSDT": "XBTUSD", "ETHUSDT": "ETHUSD", "SOLUSDT": "SOLUSD",
            "BNBUSDT": "BNBUSD", "XRPUSDT": "XRPUSD", "DOGEUSDT": "DOGEUSD",
            "AVAXUSDT": "AVAXUSD", "MATICUSDT": "POLUSD", "LINKUSDT": "LINKUSD",
            "ADAUSDT": "ADAUSD"
        }
        interval_map = {
            "1": 1, "3": 3, "5": 5, "15": 15,
            "30": 30, "60": 60, "120": 120, "240": 240,
            "D": 1440, "W": 10080
        }
        kraken_symbol = symbol_map.get(symbol)
        kraken_interval = interval_map.get(INTERVAL, 15)
        if not kraken_symbol:
            logger.warning(f"Simbolo {symbol} non mappato per Kraken, skip.")
            return None

        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": kraken_symbol, "interval": kraken_interval}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        if result.get("error"):
            logger.error(f"Kraken error {symbol}: {result['error']}")
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
    """Calcola l'ATR (Average True Range) sulle ultime N candele."""
    high = df["high"]
    low  = df["low"]
    close_prev = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low  - close_prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(ATR_PERIOD).mean().iloc[-1]


def check_signal(df: pd.DataFrame) -> str | None:
    """Strategia: EMA crossover + spike di volume."""
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
    """Ottieni il prezzo corrente da Kraken."""
    try:
        symbol_map = {
            "BTCUSDT": "XBTUSD", "ETHUSDT": "ETHUSD", "SOLUSDT": "SOLUSD",
            "BNBUSDT": "BNBUSD", "XRPUSDT": "XRPUSD", "DOGEUSDT": "DOGEUSD",
            "AVAXUSDT": "AVAXUSD", "MATICUSDT": "POLUSD", "LINKUSDT": "LINKUSD",
            "ADAUSDT": "ADAUSD"
        }
        kraken_symbol = symbol_map.get(symbol)
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


async def send_signal(app: Application, symbol: str, signal: str, price: float, sl: float, tp: float) -> None:
    """Manda il segnale su Telegram con SL/TP calcolati da ATR."""
    emoji = "🟢" if signal == "BUY" else "🔴"
    text = (
        f"{emoji} *{signal} Signal — {symbol}*\n\n"
        f"💰 Prezzo: `${price:,.4f}`\n"
        f"🛑 Stop Loss: `${sl:,.4f}`\n"
        f"🎯 Take Profit: `${tp:,.4f}`\n"
        f"📊 Timeframe: {INTERVAL}m | ATR x{ATR_SL_MULT}/{ATR_TP_MULT}\n"
        f"💵 Ordine: ~${ORDER_USDT} USDT\n\n"
        f"Vuoi eseguire questo trade?"
    )

    order_id = f"{symbol}_{signal}_{int(asyncio.get_event_loop().time())}"
    pending_orders[order_id] = {
        "symbol": symbol, "signal": signal,
        "price": price, "sl": sl, "tp": tp
    }

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Esegui", callback_data=f"exec_{order_id}"),
        InlineKeyboardButton("❌ Salta",  callback_data=f"skip_{order_id}"),
    ]])

    await app.bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=text,
        parse_mode="Markdown",
        reply_markup=keyboard,
    )
    logger.info(f"Segnale inviato: {signal} {symbol} @ {price} | SL={sl:.4f} TP={tp:.4f}")


def execute_order(symbol: str, signal: str, price: float) -> dict:
    """Esegue l'ordine market su Bybit."""
    qty = round(ORDER_USDT / price, 6)
    side = "Buy" if signal == "BUY" else "Sell"
    try:
        resp = session.place_order(
            category=CATEGORY,
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=str(qty),
        )
        return resp
    except Exception as e:
        logger.error(f"Errore ordine {symbol}: {e}")
        return {"error": str(e)}


def close_position(symbol: str, signal: str, qty: float) -> dict:
    """Chiude la posizione aperta con ordine market inverso."""
    close_side = "Sell" if signal == "BUY" else "Buy"
    try:
        resp = session.place_order(
            category=CATEGORY,
            symbol=symbol,
            side=close_side,
            orderType="Market",
            qty=str(qty),
        )
        return resp
    except Exception as e:
        logger.error(f"Errore chiusura {symbol}: {e}")
        return {"error": str(e)}


async def monitor_positions(app: Application) -> None:
    """Loop che monitora SL/TP per ogni posizione aperta."""
    while True:
        await asyncio.sleep(MONITOR_INTERVAL)
        if not open_positions:
            continue

        for symbol, pos in list(open_positions.items()):
            price = get_current_price(symbol)
            if price is None:
                continue

            hit_sl = (pos["signal"] == "BUY"  and price <= pos["sl"]) or \
                     (pos["signal"] == "SELL" and price >= pos["sl"])
            hit_tp = (pos["signal"] == "BUY"  and price >= pos["tp"]) or \
                     (pos["signal"] == "SELL" and price <= pos["tp"])

            if hit_sl or hit_tp:
                reason = "🛑 STOP LOSS" if hit_sl else "🎯 TAKE PROFIT"
                resp = close_position(symbol, pos["signal"], pos["qty"])

                if "error" not in resp:
                    del open_positions[symbol]
                    save_trade(symbol, pos["signal"], pos["entry_price"], price, pos["qty"], reason)
                    pnl = (price - pos["entry_price"]) * pos["qty"]
                    if pos["signal"] == "SELL":
                        pnl = -pnl
                    pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                    await app.bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=(
                            f"{reason} colpito!\n\n"
                            f"*{pos['signal']} {symbol}*\n"
                            f"Entry: `${pos['entry_price']:,.4f}`\n"
                            f"Exit: `${price:,.4f}`\n"
                            f"{pnl_emoji} P&L: `${pnl:+.4f}`"
                        ),
                        parse_mode="Markdown"
                    )
                    logger.info(f"{reason} {symbol} @ {price} | PnL={pnl:+.4f}")
                else:
                    logger.error(f"Errore chiusura {reason} {symbol}: {resp['error']}")


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gestisce i bottoni Telegram."""
    query = update.callback_query
    await query.answer()

    data = query.data
    action, order_id = data.split("_", 1)
    order = pending_orders.pop(order_id, None)

    if not order:
        await query.edit_message_text("⚠️ Ordine già gestito o scaduto.")
        return

    if action == "exec":
        resp = execute_order(order["symbol"], order["signal"], order["price"])
        if "error" in resp:
            await query.edit_message_text(
                f"❌ Errore nell'esecuzione:\n`{resp['error']}`",
                parse_mode="Markdown"
            )
        else:
            qty = round(ORDER_USDT / order["price"], 6)
            # Registra posizione aperta per monitoraggio SL/TP
            open_positions[order["symbol"]] = {
                "signal":       order["signal"],
                "entry_price":  order["price"],
                "qty":          qty,
                "sl":           order["sl"],
                "tp":           order["tp"],
                "order_id":     resp.get("result", {}).get("orderId", "N/A"),
            }
            await query.edit_message_text(
                f"✅ *Ordine eseguito!*\n\n"
                f"{order['signal']} {order['symbol']} @ `${order['price']:,.4f}`\n"
                f"🛑 SL: `${order['sl']:,.4f}`\n"
                f"🎯 TP: `${order['tp']:,.4f}`\n"
                f"🔍 Monitoraggio attivo ogni {MONITOR_INTERVAL}s",
                parse_mode="Markdown"
            )
    else:
        await query.edit_message_text(f"⏭️ Segnale saltato: {order['signal']} {order['symbol']}")


async def scan_loop(app: Application) -> None:
    """Loop principale di scansione segnali."""
    await app.bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text="🤖 *Bot avviato!*\nScansiono: " + ", ".join(SYMBOLS) +
             f"\n📐 ATR SL x{ATR_SL_MULT} | TP x{ATR_TP_MULT}",
        parse_mode="Markdown"
    )

    while True:
        logger.info("Scansione in corso...")
        for symbol in SYMBOLS:
            # Salta simboli già in posizione aperta
            if symbol in open_positions:
                continue
            df = get_klines(symbol)
            if df is None:
                continue
            signal = check_signal(df)
            if signal:
                price = df.iloc[-1]["close"]
                atr   = calc_atr(df)
                if signal == "BUY":
                    sl = price - ATR_SL_MULT * atr
                    tp = price + ATR_TP_MULT * atr
                else:
                    sl = price + ATR_SL_MULT * atr
                    tp = price - ATR_TP_MULT * atr
                await send_signal(app, symbol, signal, price, sl, tp)
            await asyncio.sleep(1)

        await asyncio.sleep(SCAN_INTERVAL)


async def main() -> None:
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CallbackQueryHandler(button_handler))

    async with app:
        await app.start()
        # Avvia monitor SL/TP in parallelo al loop di scansione
        await asyncio.gather(
            scan_loop(app),
            monitor_positions(app),
        )
        await app.stop()


if __name__ == "__main__":
    asyncio.run(main())
