import os
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
INTERVAL        = "15"      # timeframe in minuti
EMA_FAST        = 9
EMA_SLOW        = 21
VOLUME_MULT     = 2.0       # spike volume: 2x la media
ORDER_USDT      = 20        # dimensione ordine in USDT
SCAN_INTERVAL   = 60        # secondi tra ogni scansione
CATEGORY        = "spot"    # "spot" o "linear" per futures

# ── Bybit client ─────────────────────────────────────────────────────────────
session = HTTP(
    testnet=False,
    api_key=BYBIT_API_KEY,
    api_secret=BYBIT_API_SECRET,
)

# ── Pending orders (in attesa di conferma Telegram) ──────────────────────────
pending_orders: dict[str, dict] = {}


def get_klines(symbol: str) -> pd.DataFrame | None:
    """
    Scarica le candele da Kraken (API pubblica, nessun blocco geografico).
    Bybit viene usato solo per eseguire gli ordini.
    """
    try:
        # Mappa simboli Bybit → Kraken
        symbol_map = {
            "BTCUSDT": "XBTUSD", "ETHUSDT": "ETHUSD", "SOLUSDT": "SOLUSD",
            "BNBUSDT": "BNBUSD", "XRPUSDT": "XRPUSD", "DOGEUSDT": "DOGEUSD",
            "AVAXUSDT": "AVAXUSD", "MATICUSDT": "MATICUSD", "LINKUSDT": "LINKUSD",
            "ADAUSDT": "ADAUSD"
        }
        # Mappa intervallo minuti → Kraken (in minuti)
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

        # Kraken restituisce {pair: [[time, open, high, low, close, vwap, volume, count]]}
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


def check_signal(df: pd.DataFrame) -> str | None:
    """
    Strategia: EMA crossover + spike di volume.
    Ritorna 'BUY', 'SELL' oppure None.
    """
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["vol_ma"]   = df["volume"].rolling(20).mean()

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    volume_spike = curr["volume"] > df["vol_ma"].iloc[-1] * VOLUME_MULT

    # Crossover rialzista
    if prev["ema_fast"] < prev["ema_slow"] and curr["ema_fast"] > curr["ema_slow"] and volume_spike:
        return "BUY"

    # Crossover ribassista
    if prev["ema_fast"] > prev["ema_slow"] and curr["ema_fast"] < curr["ema_slow"] and volume_spike:
        return "SELL"

    return None


async def send_signal(app: Application, symbol: str, signal: str, price: float) -> None:
    """Manda il segnale su Telegram con bottoni Esegui / Salta."""
    emoji = "🟢" if signal == "BUY" else "🔴"
    text = (
        f"{emoji} *{signal} Signal — {symbol}*\n\n"
        f"💰 Prezzo attuale: `${price:,.4f}`\n"
        f"📊 Timeframe: {INTERVAL}m\n"
        f"📐 Strategia: EMA {EMA_FAST}/{EMA_SLOW} + Volume spike\n"
        f"💵 Ordine: ~${ORDER_USDT} USDT\n\n"
        f"Vuoi eseguire questo trade?"
    )

    order_id = f"{symbol}_{signal}_{int(asyncio.get_event_loop().time())}"
    pending_orders[order_id] = {"symbol": symbol, "signal": signal, "price": price}

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Esegui", callback_data=f"exec_{order_id}"),
            InlineKeyboardButton("❌ Salta",  callback_data=f"skip_{order_id}"),
        ]
    ])

    await app.bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=text,
        parse_mode="Markdown",
        reply_markup=keyboard,
    )
    logger.info(f"Segnale inviato: {signal} {symbol} @ {price}")


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
            await query.edit_message_text(f"❌ Errore nell'esecuzione:\n`{resp['error']}`", parse_mode="Markdown")
        else:
            await query.edit_message_text(
                f"✅ *Ordine eseguito!*\n\n"
                f"{order['signal']} {order['symbol']} @ ${order['price']:,.4f}\n"
                f"Order ID: `{resp.get('result', {}).get('orderId', 'N/A')}`",
                parse_mode="Markdown"
            )
    else:
        await query.edit_message_text(f"⏭️ Segnale saltato: {order['signal']} {order['symbol']}")


async def scan_loop(app: Application) -> None:
    """Loop principale di scansione."""
    await app.bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text="🤖 *Bot avviato!*\nScansiono: " + ", ".join(SYMBOLS),
        parse_mode="Markdown"
    )

    while True:
        logger.info("Scansione in corso...")
        for symbol in SYMBOLS:
            df = get_klines(symbol)
            if df is None:
                continue
            signal = check_signal(df)
            if signal:
                price = df.iloc[-1]["close"]
                await send_signal(app, symbol, signal, price)
            await asyncio.sleep(1)  # pausa tra simboli

        await asyncio.sleep(SCAN_INTERVAL)


async def main() -> None:
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CallbackQueryHandler(button_handler))

    async with app:
        await app.start()
        await scan_loop(app)
        await app.stop()


if __name__ == "__main__":
    asyncio.run(main())
