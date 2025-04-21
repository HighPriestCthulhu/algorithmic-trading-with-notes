import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, Math, Markdown
import inspect
import Hurst_Playground
#%%
data = yf.download("CAD=X", start="2018-01-01", end="2023-04-15")
close_data=pd.DataFrame(data['Close'].dropna().values)
#%%
#z(t) - z(t-tau) for all possible tau
#np.var()
def var_test(a, max_lag = 5):
    max_lag = min(max_lag, len(data) // 2)
    lags = range(1, max_lag)
    difs=[]
    a=np.log(a)
    print("a:", a)
    a=np.array(a)
    for i in lags:
        difs.append(a[i:]-a[:-i])
        print(a[:-i]-a[i:])
    return(difs)
    
a=close_data

z= var_test(a)

#%%
#Some play tests to ensure that the hurst function is working correctly



# print(Hurst_Playground.get_hurst_single(close_data))
# a=np.cumsum(np.random.randn(10000)-.1) 
# print(Hurst_Playground.get_hurst_single(a, max_lag=1000, preview=True, qval=2))
# mean_revert = np.random.normal(loc=0, scale=1, size=10000)*np.random.randn(10000)
# for i in range(1, len(mean_revert)):
#     mean_revert[i] += .05 * (0 - mean_revert[i-1]) * np.random.randint(2)/10 # mean-reversion
# print(Hurst_Playground.get_hurst_single(mean_revert,max_lag=1000, preview=True, qval=2))
