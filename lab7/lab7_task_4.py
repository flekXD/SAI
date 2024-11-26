import json
import numpy as np
import pandas as pd
from sklearn import covariance, cluster
import yfinance as yf

# Function to download stock quotes from Yahoo Finance
def quotes_yahoo(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df['Date'] = df.index
    return df

# Load company symbols mapping
input_file = 'company_symbol_mapping.json'
with open(input_file, 'r') as f:
    company_symbols_map = json.load(f)

symbols, names = np.array(list(company_symbols_map.items())).T

# Parameters for analysis
start_date = "2003-07-03"
end_date = "2007-05-04"

# Collect data for all companies
all_quotes = {}
common_dates = None

for symbol in symbols:
    print(f"Downloading data for {symbol}...")
    try:
        quotes = quotes_yahoo(symbol, start_date, end_date)
        if quotes.empty:
            print(f"No data found for {symbol}")
            continue  # Skip this symbol if no data is returned
        quotes['variation'] = quotes['Close'] - quotes['Open']
        all_quotes[symbol] = quotes.set_index('Date')['variation']
        
        # Determine the common date range
        if common_dates is None:
            common_dates = quotes.index
        else:
            common_dates = common_dates.intersection(quotes.index)
    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")

# Align all data to the common date range
aligned_quotes = []
valid_symbols = []

for symbol, series in all_quotes.items():
    # Reindex to align with common dates
    series = series.reindex(common_dates)
    if not series.isna().all():  # Exclude stocks with completely missing data
        aligned_quotes.append(series.fillna(0).values)  # Replace NaN with 0
        valid_symbols.append(symbol)

# Convert to NumPy array
if aligned_quotes:
    X = np.array(aligned_quotes).T  # Transpose to match sklearn's requirements
else:
    print("No valid data available for analysis.")
    X = None

# Ensure X is valid for modeling
if X is None or X.shape[1] < 2:
    raise ValueError("Insufficient data for covariance modeling.")

# Normalize data
X /= X.std(axis=0, where=~np.isnan(X))  # Handle possible NaN during standardization

# Covariance model and clustering
edge_model = covariance.GraphicalLassoCV()
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

_, labels = cluster.affinity_propagation(edge_model.covariance_)

# Display clustering results
num_labels = labels.max()
print("\nClustering results:")
for i in range(num_labels + 1):
    cluster_companies = [valid_symbols[j] for j in range(len(valid_symbols)) if labels[j] == i]
    print(f"Cluster {i + 1}: {', '.join(cluster_companies)}")
