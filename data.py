#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
from yahooquery import Ticker


# In[12]:


def load_spy(start="2005-01-01", end=None):
    t = Ticker("SPY")
    df = t.history(start=start, end=end, interval="1d")
    df = df.reset_index()
    df = df[df['symbol']=="SPY"].copy()
    df = df.rename(columns=str.lower)
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert(None)
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']].sort_values('date')
    df = df.set_index('date')
    return df


# In[1]:


#df = load_spy()
#print(df.head())
#print(df.columns)


# In[ ]:




