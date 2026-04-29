# Close-to-Close と Open-to-Close の2種類のリターンを計算する。
# シグナル推定にはCC、戦略評価にはOCを使う（論文の定義に従う）。
import pandas as pd
from .fetch_data import JP_TICKERS


def compute_returns(close: pd.DataFrame, open_: pd.DataFrame) -> dict:
    cc = close.pct_change().fillna(0)
    oc = (close[JP_TICKERS] / open_[JP_TICKERS] - 1).fillna(0)

    # 両方に値がある日付のみ残す
    common_idx = cc.dropna(how="any").index.intersection(oc.dropna(how="any").index)
    return {"cc": cc.loc[common_idx], "oc": oc.loc[common_idx]}


def standardize_window(cc: pd.DataFrame, window_idx: pd.DatetimeIndex) -> dict:
    sub   = cc.loc[window_idx]
    mu    = sub.mean()
    sigma = sub.std().replace(0, 1e-8)
    return {"z": (sub - mu) / sigma, "mu": mu, "sigma": sigma}


def standardize_single(ret_row: pd.Series, mu: pd.Series, sigma: pd.Series) -> pd.Series:
    return (ret_row - mu) / sigma.replace(0, 1e-8)
