import yfinance as yf
import pandas as pd

def collect_raw_data(feature_names):
    data_frames_raw_dict = {}

    for name in feature_names :
        filename = f'../data/{name}.csv'
        df = pd.read_csv(filename)
        if "observation_date" in df.columns:
            df = df.rename(columns={"observation_date": "Date"})
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.resample("D").ffill()
        data_frames_raw_dict[name] = df
    
    return data_frames_raw_dict

def get_data_yfinance(tickers,start,end):
    indices_data = yf.download(tickers, start=start, end=end)
    for ticker in tickers :
        indices_data['Close'][ticker].to_csv(f'../data/{ticker}.csv')
    return indices_data