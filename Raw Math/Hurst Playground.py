import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd


def generalized_hurst(ts, q_vals=np.linspace(0.5, 5, 20), max_lag=300, preview=True): #qvals: start at 0.5, twenty values 2 5. This measures the nth moment
    # print(type(ts))
    # ts = np.array(ts) #ts= timeseries data, 
    # if preview:
    #     print(ts[:5]) #preview the data imported
    lags = range(1, max_lag)
    H_q = []

    for q in q_vals:
        moments = []
        for lag in lags:
            diffs = np.abs(ts[lag:] - ts[:-lag])
            moment = np.mean(diffs**(2*q))
            moments.append(moment)

        log_lags = np.log(lags)
        log_moments = np.log(moments)
        H, _ = np.polyfit(log_lags, log_moments, 1)
        H_q.append(H / (2 * q))

    return np.array(q_vals), np.array(H_q)

# Download USD/CAD exchange rate (CAD=X)
data = yf.download("CAD=X", start="2018-01-01", end="2024-12-31")
log_prices = np.log(data['Close'].dropna().values)

# Compute and plot spectrum
q_vals, H_q_vals = generalized_hurst(log_prices)

plt.figure(figsize=(8, 5))
plt.plot(q_vals, H_q_vals, marker='o', color='purple')
plt.title("Multifractal Spectrum H(q) vs q for USD/CAD")
plt.xlabel("q")
plt.ylabel("H(q)")
plt.grid(True)
plt.tight_layout()
plt.show()