# 🤖 Crypto Signal Bot — Bybit + Telegram

Bot di segnali con esecuzione semi-automatica su Bybit via Telegram.

## Strategia
- **EMA 9/21 crossover** su timeframe 15 minuti
- **Spike di volume** (2x la media mobile a 20 periodi)
- Segnali BUY e SELL con conferma manuale su Telegram

## Simboli monitorati
BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT, AVAXUSDT, MATICUSDT, LINKUSDT, ADAUSDT

---

## Setup su Railway

### 1. Crea il repository GitHub
```bash
git init
git add .
git commit -m "first commit"
git remote add origin https://github.com/TUO_USERNAME/crypto-bot.git
git push -u origin main
```

### 2. Crea progetto su Railway
1. Vai su [railway.app](https://railway.app)
2. **New Project** → **Deploy from GitHub repo**
3. Seleziona il tuo repository

### 3. Aggiungi le variabili d'ambiente
In Railway → tab **Variables**, aggiungi:

| Variabile | Valore |
|-----------|--------|
| `TELEGRAM_TOKEN` | Il token del tuo bot Telegram |
| `TELEGRAM_CHAT_ID` | Il tuo chat ID Telegram |
| `BYBIT_API_KEY` | La tua API Key Bybit |
| `BYBIT_API_SECRET` | Il tuo API Secret Bybit |

### 4. Deploy
Railway fa il deploy automaticamente. Controlla i log per verificare che il bot sia avviato.

---

## Parametri configurabili (in bot.py)

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `INTERVAL` | `"15"` | Timeframe candele (minuti) |
| `EMA_FAST` | `9` | Periodo EMA veloce |
| `EMA_SLOW` | `21` | Periodo EMA lenta |
| `VOLUME_MULT` | `2.0` | Moltiplicatore spike volume |
| `ORDER_USDT` | `20` | Dimensione ordine in USDT |
| `SCAN_INTERVAL` | `60` | Secondi tra scansioni |
| `CATEGORY` | `"spot"` | Mercato: "spot" o "linear" |

---

## Come funziona

1. Il bot scansiona i simboli ogni 60 secondi
2. Quando rileva un crossover EMA + spike di volume → manda notifica Telegram
3. Tu premi **✅ Esegui** o **❌ Salta**
4. Solo se premi Esegui → ordine market su Bybit

## ⚠️ Note importanti
- Inizia con `ORDER_USDT = 20` (piccolo) per testare
- Verifica sempre i log su Railway
- Non abilitare mai i permessi di prelievo sulle API Bybit
