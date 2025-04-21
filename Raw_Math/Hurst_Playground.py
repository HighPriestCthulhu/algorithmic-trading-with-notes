# %%
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, Math, Markdown

plt.ion
print("imports ok - Hurst_Playground")
#%% Depreciated Hurst Function - Not resilient to negative numbers
#Note: hurst takes the moments of the  increments as it's arguments, not the moments of the data. 
# def generalized_hurst(data,q_vals=np.linspace(0.5, 5, 10), max_lag=3, preview=True, walkthrough=False): #qvals: start at 0.5, twenty values 2 5. This measures the nth moment
#     # print(type(ts))
#     # ts = np.array(ts) #ts= timeseries data, 
#     # if preview:
#     #     print(ts[:5]) preview the data imported
#     close_data=pd.DataFrame(data['Close'].dropna().values)
#     log_data = np.log(close_data)
#     if walkthrough:
#         print("broad overview of data:")
#         log_data.compare_to.summary(close_data, name1="log", name2="raw data")
#         rightingnum=(close_data[0]-log_data[0])[0]
#         print("data must be converted to log")
#         plt.figure(figsize=(8,5))
#         plt.plot(close_data, color='purple', label="OG Data")
#         plt.plot(log_data+rightingnum, color='blue', label='Log-Transformed')
#         plt.legend()
#         plt.title("Log Vs Raw Data, with log data origin corrected")
#         plt.show()
#         #EXPLAIN HURST
#         display(Markdown(r"""
# In a multifractal system, the scaling relationship is:

# $\mathbb{E} \left[ \left| X(t + \tau) - X(t) \right|^{2q} \right] \sim \tau^{2qH(q)}$
# """))
#         #EXPLAIN LINEARISING
#         display(Markdown(r"""

# $\log \left( \mathbb{E} \left[ \left| X(t + \tau) - X(t) \right|^{2q} \right] \right) \sim \log(C) + 2qH(q) \log(\tau)$

# This yields a straight-line relationship of the form:

# $y = a + b \cdot x$

# Where:
# - \tau = some lag between numbers
# - $y = \log \left( \mathbb{E}[|X(t+\tau) - X(t)|^{2q}] \right)$  
# - $x = \log(\tau)$  
# - $\text{slope} = 2qH(q)$
#                          """))

    
#     close_data=np.array(close_data)
#     log_data=np.array(log_data)
#     if walkthrough:
#         print("mean price:", np.mean(close_data))
#         print("mean log‑price:", np.mean(log_data))
#     lags = range(1, max_lag)
#     H_q = []
#     for q in q_vals:
#         moments = []
#         running_diffs=[]
        
#         for lag in lags:
#             diffs = (log_data[lag:] - log_data[:-lag])
#                 #elegant lag solution, 
#                 #if [1,2,3,4]
#                 ##t[1:]=[2,3,4]
#                 ##t[:-1]=[1,2,3]
#                 ##-> diffs= abs[2-1,2-3,3-4]
#             moment = np.mean(np.abs(diffs)**(2*q))
#             moments.append(moment)
#             running_diffs.append(diffs)
#         print("",q,"'th moment=",(moment))


#         log_lags = np.log(lags)
#         log_moments = np.log(moments)
#         H, _ = np.polyfit(log_lags, log_moments, 1)
#         H_q.append(H / (2 * q))

#     return np.array(q_vals), np.array(H_q)
#%%
def get_hurst_single(data, qval=1, max_lag=1000, preview =False):
    data = np.array(data)
    max_lag = min(max_lag, len(data) // 2)
    moments = []
    lags = range(1, max_lag)

    for lag in lags:
        diffs = data[lag:] - data[:-lag]
        moment = np.mean(np.abs(diffs) ** (2 * qval))
        moments.append(moment)

    log_lags = np.log(lags)
    epsilon = 1e-10
    log_moments = np.log(np.maximum(moments, epsilon))
    H, _ = np.polyfit(log_lags, log_moments, 1)

    if preview:
        print("Previewing")
        plt.plot(log_lags, log_moments, label="log moments")
        plt.plot(log_lags, H * log_lags + _, label=f"fit (H={H / (2 * qval):.3f})")
        plt.legend()
        plt.xlabel("log(lag)")
        plt.ylabel("log(moment)")
        plt.title("Hurst Exponent Estimate")
        plt.show()

    return H / (2 * qval)

#%% new hurst general function
def hurst_generalised(data,q_vals=np.linspace(0.5, 5, 10), max_lag=10, preview=True, walkthrough=False):
    H_q = []
    for i in q_vals:
        H_q.append(get_hurst_single(data,qval=i,max_lag=max_lag))
    return q_vals, H_q

# %%
# Download exchange rate (CAD=X)
def main():
    data = yf.download("EUR=X", start="2018-01-01", end="2025-04-15")
    print(hurst_generalised(pd.DataFrame(data['Close'].dropna().values)))
    q_vals, H_q_vals = hurst_generalised(pd.DataFrame(data['Open'].dropna().values), max_lag=100)

    plt.figure(figsize=(8, 5))
    plt.plot(q_vals, H_q_vals, color='purple', marker='x')
    plt.title("Multifractal Spectrum H(q) vs q")
    plt.xlabel("moment")
    plt.ylabel("H(q)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


# %%
