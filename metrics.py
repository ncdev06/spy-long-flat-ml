#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[6]:


def performance(df: pd.DataFrame, col="strat_ret", freq=252):
    r = df[col].dropna()
    ann_ret = (1 + r).prod()**(freq/len(r)) - 1
    ann_vol = r.std() * np.sqrt(freq)
    sharpe = 0.0 if ann_vol == 0 else ann_ret / ann_vol

    downside = r[r < 0]
    dd_vol = downside.std() * np.sqrt(freq)
    sortino = 0 if dd_vol == 0 else ann_ret / dd_vol

    equity = (1 + r).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak) - 1
    max_dd = drawdown.min()

    win_rate = (r > 0).mean()
    turnover = df['delta_pos'].sum() / (len(df)/freq)

    return { 
        "n_days": int(len(r)),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "turnover_pa": float(turnover)
    }


# In[ ]:





# In[ ]:




