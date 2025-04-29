import threading, time
import pandas as pd
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from datetime import datetime

# Configuración
CONFIG = {
    'IB_HOST':      '127.0.0.1',
    'IB_PORT':      4002,
    'CLIENT_ID':    23,
    'DURATION':     '30 D',
    'BAR_SIZE':     '30 mins',
    'SYMBOLS_FILE': 'watchlist1.csv',
    'OUTPUT_FILE':  'historical_data.csv'
}

# Clase app para recibir históricos
class TradeApp(EWrapper, EClient):
    def __init__(self): 
        EClient.__init__(self, self)
        self.data = {}
        self.failed_requests = set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=None):
        print(f"⚠️ Error {errorCode} en reqId {reqId}: {errorString}")
        if errorCode == 200:
            self.failed_requests.add(reqId)
            event.set()
        if errorCode == 504:
            print(f"❗ Error 504: Not connected. Revisa la conexión al TWS o IB Gateway.")

    def historicalData(self, reqId, bar):
        row = {
            "Date": bar.date, "Open": bar.open, "High": bar.high,
            "Low": bar.low, "Close": bar.close, "Volume": bar.volume
        }
        self.data.setdefault(reqId, []).append(row)

    def historicalDataEnd(self, reqId, start, end):
        print(f"✔️ Datos completos para reqId {reqId}")
        event.set()

# Crear contrato
def us_stock_contract(symbol):
    contract = Contract()
    contract.symbol   = symbol
    contract.secType  = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract

# Cargar símbolos desde CSV
def load_symbols_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df['Symbol'].dropna().unique().tolist()
    except Exception as e:
        print(f"❌ Error al leer CSV: {e}")
        return []

# Descargar datos y guardar Excel
def download_and_save_excel(tickers, duration, bar_size):
    global event
    event = threading.Event()
    
    app = TradeApp()
    app.connect(CONFIG['IB_HOST'], CONFIG['IB_PORT'], clientId=CONFIG['CLIENT_ID'])
    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()
    time.sleep(1)

    for idx, symbol in enumerate(tickers):
        print(f"📥 Descargando: {symbol}")
        event.clear()
        app.reqHistoricalData(
            reqId=idx,
            contract=us_stock_contract(symbol),
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='ADJUSTED_LAST',
            useRTH=1,
            formatDate=1,
            keepUpToDate=0,
            chartOptions=[]
        )
        event.wait()

    all_dfs = []
    for idx, symbol in enumerate(tickers):
        if idx in app.failed_requests:
            print(f"🚫 Skipping {symbol} (failed security definition)")
            continue  # Salta este símbolo y pasa al siguiente

        df = pd.DataFrame(app.data.get(idx, []))
        if df.empty:
            print(f"⚠️ Sin datos para {symbol}")
            continue
        df['Date'] = df['Date'].apply(lambda x: ' '.join(str(x).split(' ')[:2]))  # <-- LIMPIAR fecha
        df.set_index("Date", inplace=True)
        df.index = pd.to_datetime(df.index)
        df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
        all_dfs.append(df)

    if all_dfs:
        full_df = pd.concat(all_dfs, axis=1)
        full_df.sort_index(inplace=True)
        full_df.to_csv(CONFIG['OUTPUT_FILE'])
        print(f"\n✅ Archivo guardado: {CONFIG['OUTPUT_FILE']}")
    else:
        print("❌ No se generó archivo: no se obtuvieron datos válidos.")

    app.disconnect()
    thread.join(1)

# --- EJECUCIÓN ---
if __name__ == "__main__":
    symbols = load_symbols_from_csv(CONFIG['SYMBOLS_FILE'])
    if not symbols:
        print("🚫 No se encontraron símbolos para procesar.")
    else:
        download_and_save_excel(symbols, CONFIG['DURATION'], CONFIG['BAR_SIZE'])
