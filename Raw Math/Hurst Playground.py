# %%
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, Math

plt.ion
print("imports ok")
#%%
def generalized_hurst(data,q_vals=np.linspace(0.5, 5, 10), max_lag=10, preview=True, walkthrough=False): #qvals: start at 0.5, twenty values 2 5. This measures the nth moment
    # print(type(ts))
    # ts = np.array(ts) #ts= timeseries data, 
    # if preview:
    #     print(ts[:5]) preview the data imported
    close_data=pd.DataFrame(data['Close'].dropna().values)
    log_data = np.log(close_data)
    if walkthrough:
        log_data.compare_to.summary(close_data, name1="log", name2="raw data")

    close_data=np.array(close_data)
    log_data=np.array(log_data)
    lags = range(1, max_lag)
    H_q = []

    for q in q_vals:
        moments = []
        for lag in lags:
            diffs = np.abs(log_data[lag:] - log_data[:-lag]) 
                #elegant lag solution, 
                #if [1,2,3,4]
                ##t[1:]=[2,3,4]
                ##t[:-1]=[1,2,3]
                ##-> diffs= abs[2-1,2-3,3-4]
          
            moment = np.mean(diffs**(2*q))
            moments.append(moment)

        log_lags = np.log(lags) #linearise X var
        log_moments = np.log(moments) #linearise y var
        H, _ = np.polyfit(log_lags, log_moments, 1)
        H_q.append(H / (2 * q))

    return np.array(q_vals), np.array(H_q)
# %%
# Download USD/CAD exchange rate (CAD=X)
data = yf.download("AUD=X", start="2018-01-01", end="2024-01-31")

plt.figure(figsize=(8,5))
plt.plot(data, marker='o', color='purple', )
plt.show()
# plt.title("Multifractal Spectrum H(q) vs q for USD/CAD")
# plt.xlabel("q")
# plt.ylabel("H(q)")
# Compute and plot spectrum
#%%
q_vals, H_q_vals = generalized_hurst(data, walkthrough=True)

plt.figure(figsize=(8, 5))
plt.plot(q_vals, H_q_vals, marker='o', color='purple')
plt.title("Multifractal Spectrum H(q) vs q for USD/AUD")
plt.xlabel("moment")
plt.ylabel("H(q)")
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
