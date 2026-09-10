from __future__ import annotations

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _build_estimator(model_name: str):
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=42,
        )

    if model_name != "logreg":
        raise ValueError("model_name must be 'logreg' or 'rf'")

    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )


def rolling_predict(
    df: pd.DataFrame,
    start_year: int = 2012,
    model_name: str = "logreg",
    min_train_days: int = 504,
    calibration_splits: int = 3,
) -> pd.Series:
    """Generate out-of-sample probabilities with expanding monthly retraining.

    Every test month is predicted using only rows strictly earlier than that month.
    Probability calibration also uses TimeSeriesSplit, avoiding shuffled or
    future-to-past validation inside the training window.
    """
    if "label" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'label' column.")

    feature_cols = [column for column in df.columns if column != "label"]
    dates = pd.DatetimeIndex(df.index)
    month_index = dates.to_period("M")
    predictions = pd.Series(index=dates, dtype=float, name="p_up")

    for month in month_index.unique():
        if month.year < start_year:
            continue

        train_mask = month_index < month
        test_mask = month_index == month

        if int(train_mask.sum()) < min_train_days or not test_mask.any():
            continue

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, "label"]
        X_test = df.loc[test_mask, feature_cols]

        if y_train.nunique() < 2:
            continue

        base = _build_estimator(model_name)
        calibration_cv = TimeSeriesSplit(n_splits=calibration_splits)
        model = CalibratedClassifierCV(base, method="sigmoid", cv=calibration_cv)
        model.fit(X_train, y_train)

        predictions.loc[df.index[test_mask]] = model.predict_proba(X_test)[:, 1]

    return predictions
