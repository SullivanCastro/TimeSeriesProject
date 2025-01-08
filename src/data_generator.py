import yfinance as yf
import pandas as pd
import argparse
import datetime

# Top global equity indices and their Yahoo Finance tickers
INDEX_TICKERS = {
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "Nasdaq 100": "^NDX",
    "FTSE 100": "^FTSE",
    "DAX": "^GDAXI",
    "CAC 40": "^FCHI",
    "Nikkei 225": "^N225",
    "Hang Seng": "^HSI",
    "Shanghai Composite": "000001.SS",
    "BSE Sensex": "^BSESN",
    "ASX 200": "^AXJO",
    "TSX Composite": "^GSPTSE",
    "KOSPI": "^KS11",
    "Straits Times": "^STI",
}

def download_data(start_date, end_date, indices=None, target_col="Close"):
    """
    Download daily close prices for the specified indices from Yahoo Finance.
    
    Parameters:
    - start_date (str): Start date in the format 'YYYY-MM-DD'.
    - end_date (str): End date in the format 'YYYY-MM-DD'.
    - indices (list): List of index names to download. If None, all indices are downloaded.
    
    Returns:
    - pd.DataFrame: DataFrame with close prices, columns as index names, and index as timestamp.
    """
    if indices is None:
        indices = list(INDEX_TICKERS.keys())
    
    data_frames = []  # List to store individual DataFrames
    for index in indices:
        ticker = INDEX_TICKERS.get(index)
        if ticker:
            print(f"Downloading data for {index}...")
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not df.empty:
                col = df[target_col]
                col.columns = [index]
                data_frames.append(col)  # Rename series to the index name
            else:
                print(f"No data found for {index} in the given date range.")
        else:
            print(f"Index '{index}' not found in the predefined list.")
    
    # Combine all DataFrames using pd.concat
    combined_data = pd.concat(data_frames, axis=1)
    return combined_data

def main():
    parser = argparse.ArgumentParser(description="Download daily close prices of top global equity indices.")
    parser.add_argument("--start", type=str, required=True, help="Start date in the format 'YYYY-MM-DD'.")
    parser.add_argument("--end", type=str, required=True, help="End date in the format 'YYYY-MM-DD'.")
    parser.add_argument("--indices", type=str, nargs="*", help="List of index names to download (default: all indices).")
    parser.add_argument("--target_col", type=str, default="Close", help="Column to download (default: Close).")
    parser.add_argument("--output", type=str, default="data.csv", help="Output file name (default: indices_data.csv).")
    
    args = parser.parse_args()

        # Validate date format
    try:
        datetime.datetime.strptime(args.start, "%Y-%m-%d")
        datetime.datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError:
        print("Error: Incorrect date format. Use 'YYYY-MM-DD'.")
        return
    
    # Download data
    indices_data = download_data(args.start, args.end, args.indices)
    
    # Save to CSV
    indices_data.to_csv(args.output)
    print(f"Data saved to {args.output}")

if __name__ == "__main__":
    main()