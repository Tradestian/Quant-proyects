# ============================================================================
#  Librerías
# ============================================================================
import threading, time, asyncio, warnings
from datetime import datetime, time as dtime
from pytz import timezone

import pandas as pd, numpy as np, ta
import matplotlib.pyplot as plt

# -------- ibapi (EXTRACTOR que funciona) ------------------------------------
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
# -------- ib_insync (conexión principal / órdenes) --------------------------
from ib_insync import IB, Stock, Order, util
warnings.filterwarnings("ignore")
util.startLoop()

# ============================================================================
#  Configuración
# ============================================================================
CONFIG = {
    'SYMBOLS_CSV': 'watchlist1.csv',          # lista de símbolos
    'IB_HOST':     '127.0.0.1',
    'IB_PORT':     4002,
    'CLIENT_ID':   10,
    'DURATION':    '30 D',
    'BAR_SIZE':    '30 mins',
    'INTERVAL':    1800,                   # seg. entre ciclos
    'MAX_POSITION_SIZE': 100,
    'RISK_PER_TRADE': 0.02,
    'SHORT_TP_MULTIPLIER': 3,
    'SHORT_SL_MULTIPLIER': 1,
    'ATR_WINDOW': 14,
    'MIN_VOLUME_RATIO': 1.5,
    'USE_REALTIME_DATA': True,
    'MARKET_OPEN':  dtime(9, 30, tzinfo=timezone('US/Eastern')),
    'MARKET_CLOSE': dtime(16, 0, tzinfo=timezone('US/Eastern')),
}

# ============================================================================
#  --- extractor de históricos (TAL CUAL lo enviaste) -------------------------
# ============================================================================
class TradeApp(EWrapper, EClient):
    def _init_(self): 
        EClient._init_(self, self) 
        self.data = {}
    def historicalData(self, reqId, bar):
        row = {"Date":bar.date,"Open":bar.open,"High":bar.high,
               "Low":bar.low,"Close":bar.close,"Volume":bar.volume}
        self.data.setdefault(reqId, []).append(row)
    def historicalDataEnd(self, reqId, start, end):
        super().historicalDataEnd(reqId, start, end)
        print(f"HistoricalDataEnd {reqId}  {start} -> {end}")
        event.set()

def usTechStk(symbol, sec_type="STK", currency="USD", exchange="SMART"):
    c = Contract()
    c.symbol, c.secType, c.currency, c.exchange = symbol, sec_type, currency, exchange
    return c

# ---------- empaquetamos todo en una función --------------------------------
def download_history_batch(tickers, duration, bar_size):
    global event
    event = threading.Event()
    app  = TradeApp()
    app.connect(CONFIG['IB_HOST'], CONFIG['IB_PORT'], clientId=23)
    th   = threading.Thread(target=app.run, daemon=True); th.start()
    time.sleep(1)                           # asegurar conexión

    for tkr in tickers:
        event.clear()
        app.reqHistoricalData(
            reqId=tickers.index(tkr),
            contract=usTechStk(tkr),
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='ADJUSTED_LAST',
            useRTH=1,
            formatDate=1,
            keepUpToDate=0,
            chartOptions=[]
        )
        event.wait()                        # esperar fin de descarga

    # transformar a DataFrames
    df_dict = {}
    for tkr in tickers:
        df = pd.DataFrame(app.data[tickers.index(tkr)])
        df.set_index("Date", inplace=True)
        df_dict[tkr] = df

    app.disconnect(); th.join(1)
    return df_dict
# ----------------------------------------------------------------------------

# ============================================================================
#  Estado y utilidades de trading
# ============================================================================
class TradingState:
    def _init_(self):
        self.orders = pd.DataFrame(columns=[
            'OrderId','Symbol','Qty','Entry','TP','SL','RR','Status','Date'])
        self.real_time = {}
state = TradingState()
ib = IB()

def calculate_indicators(df):
    df = df.copy()
    df['RSI']    = ta.momentum.rsi(df['Close'], 14)
    df['SMA50']  = df['Close'].rolling(50).mean()
    df['ATR']    = ta.volatility.average_true_range(
                    df['High'], df['Low'], df['Close'], CONFIG['ATR_WINDOW'])
    df['vol_ma'] = df['Volume'].rolling(20).mean()

    df['sell_sig'] = (
        (df['RSI'] > 70) &
        (df['Close'] < df['SMA50']) &
        (df['Close'].shift(1) > df['SMA50']) &
        (df['Volume'] > df['vol_ma'] * CONFIG['MIN_VOLUME_RATIO'])
    )
    return df

async def update_rt(symbol):
    tk = await ib.reqMktDataAsync(Stock(symbol,'SMART','USD'))
    state.real_time[symbol] = {'price': tk.marketPrice, 'vol': tk.volume}

async def place_bracket_sell(symbol, entry, atr):
    qty = min(int(
        (float(next(a.value for a in await ib.accountSummaryAsync()
                    if a.tag=='NetLiquidation'))
        * CONFIG['RISK_PER_TRADE'])
        / (CONFIG['SHORT_SL_MULTIPLIER'] * atr)),
        CONFIG['MAX_POSITION_SIZE'])
    if qty < 1:
        print(f"{symbol}: posición < 1 acción, omitiendo")
        return

    tp = round(entry - CONFIG['SHORT_TP_MULTIPLIER']*atr, 2)
    sl = round(entry + CONFIG['SHORT_SL_MULTIPLIER']*atr, 2)

    parent = Order(action='SELL', orderType='MKT', totalQuantity=qty, transmit=False)
    tp_ord = Order(action='BUY', orderType='LMT', lmtPrice=tp,
                   totalQuantity=qty, tif='GTC', parentId=parent.orderId, transmit=False)
    sl_ord = Order(action='BUY', orderType='STP', auxPrice=sl,
                   totalQuantity=qty, tif='GTC', parentId=parent.orderId, transmit=True)

    for o in (parent, tp_ord, sl_ord):
        tr = await ib.placeOrderAsync(Stock(symbol,'SMART','USD'), o)
        print(f"{symbol}: {o.action} enviada -> {tr.orderStatus.status}")

    state.orders = pd.concat([state.orders, pd.DataFrame([{
        'OrderId': parent.orderId, 'Symbol': symbol, 'Qty': qty,
        'Entry': entry, 'TP': tp, 'SL': sl,
        'RR': f"1:{CONFIG['SHORT_TP_MULTIPLIER']/CONFIG['SHORT_SL_MULTIPLIER']:.2f}",
        'Status': 'Active',
        'Date': datetime.now(timezone('US/Eastern')).strftime('%F %T')
    }])], ignore_index=True)

async def process_symbol(symbol, df):
    df = calculate_indicators(df)
    last = df.iloc[-1]
    print(f"[{symbol}] Close={last.Close:.2f} | RSI={last.RSI:.1f} | "
          f"sell_sig={last.sell_sig}")
    if not last.sell_sig:
        return

    if CONFIG['USE_REALTIME_DATA']:
        await update_rt(symbol)
        rt  = state.real_time[symbol]
        if (rt['price'] < last.SMA50) and \
           (rt['vol']   > last.vol_ma * CONFIG['MIN_VOLUME_RATIO']):
            await place_bracket_sell(symbol, rt['price'], last.ATR)
        else:
            print(f"{symbol}: confirmación RT fallida")
    else:
        await place_bracket_sell(symbol, last.Close, last.ATR)

# ============================================================================
#  Bucle principal
# ============================================================================
async def trading_cycle(tickers):
    print(f"\n=== Ciclo {datetime.now().strftime('%F %T')} ===")
    hist = download_history_batch(tickers, CONFIG['DURATION'], CONFIG['BAR_SIZE'])
    for sym, df in hist.items():
        await process_symbol(sym, df)

async def day_loop(tickers):
    est = timezone('US/Eastern')
    while True:
        now = datetime.now(est).time()
        if now >= CONFIG['MARKET_CLOSE']:
            print("Mercado cerrado 🚪"); break
        if CONFIG['MARKET_OPEN'] <= now < CONFIG['MARKET_CLOSE']:
            await trading_cycle(tickers)
            await asyncio.sleep(CONFIG['INTERVAL'])
        else:
            await asyncio.sleep(60)

def load_tickers():
    try:
        return pd.read_csv(CONFIG['SYMBOLS_CSV'])['Symbol'].unique().tolist()
    except Exception as e:
        print(f"CSV error: {e}")
        return ["AAPL","AMZN"]      # fallback

async def main():
    tickers = load_tickers()
    ib.connect(CONFIG['IB_HOST'], CONFIG['IB_PORT'], clientId=CONFIG['CLIENT_ID'])
    try:
        await day_loop(tickers)
    finally:
        ib.disconnect()

if __name__ == "__main__":
    asyncio.run(main())