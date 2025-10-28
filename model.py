#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler


# In[2]:


def rolling_predict(df: pd.DataFrame, start_year=2012, model_name="logreg"):
    features = [c for c in df.columns if c not in ['label']]
    X = df[features].values
    y = df['label'].values
    dates = df.index

    preds = pd.Series(index=dates, dtype=float)

    months = sorted(set((d.year, d.month) for d in dates))
    scaler = StandardScaler()

    for (y0, m0) in months:
        if y0 < start_year:
            continue
        mask_train = [(d.year < y0) or (d.year == y0 and d.month < m0) for d in dates]
        mask_test = [(d.year == y0 and d.month == m0) for d in dates]

        if sum(mask_train) < 252:
            continue

        Xtr, ytr = X[mask_train], y[mask_train]
        Xte = X[mask_test]

        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        if model_name == "rf":
            base = RandomForestClassifier(
                n_estimators=300, max_depth=6, min_samples_leaf=10, n_jobs=-1, random_state=42)
        else: 
            base = LogisticRegression(max_iter=1000, n_jobs=-1)

        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        clf.fit(Xtr_s, ytr)

        p_up = clf.predict_proba(Xte_s)[:, 1]
        preds.loc[df.index[mask_test]] = p_up

    return preds


# In[ ]:




