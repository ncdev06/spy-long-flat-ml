#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import ta


# In[3]:


def make_features(df: pd.DataFrame):
    out = df.copy()
    out['ret1'] = out['close'].pct_change()
    out['logret1'] = np.log1p(out['ret1'])

    out['sma5'] = out['close'].rolling(5).mean()
    out['sma20'] = out['close'].rolling(20).mean()
    out['ema12'] = out['close'].ewm(span=12, adjust=False).mean()
    out['ema26'] = out['close'].ewm(span=26, adjust=False).mean()

    out['rsi14'] = ta.momentum.rsi(out['close'], window=14)

    high, low, close = out['high'], out['low'], out['close']
    out['atr14'] = ta.volatility.average_true_range(high, low, close, window=14)
    out['ret_std20'] = out['logret1'].rolling(20).std()

    out['close_sma5'] = out['close'] / out['sma5'] - 1.0
    out['close_sma20'] = out['close'] / out['sma20'] - 1.0

    out['label'] = (out['close'].shift(-1) > out['close']).astype(int)

    out = out.dropna().copy()

    feature_cols = [c for c in out.columns if c not in ['label']]
    out[feature_cols] = out[feature_cols].shift(1)
    out = out.dropna().copy()

    return out


# In[ ]:




