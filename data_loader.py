import yfinance as yf 

def load_data(stock):
    data = yf .download(stock, start="2018-01-01", end="2026-04-11")
    return data