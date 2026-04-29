# シグナルからロングショートポートフォリオを構築する。
# 上位q・下位qを等ウェイトで持つシンプルな構成。
import numpy as np
import pandas as pd
from ..data.fetch_data import JP_TICKERS


def long_short_weights(signal: pd.Series, q: float = 0.3) -> pd.Series:
    n = len(signal)
    k = max(1, int(np.floor(n * q)))

    ranked  = signal.rank(method="first", ascending=False)
    weights = pd.Series(0.0, index=signal.index)

    weights[ranked <= k]        =  1.0 / k   # ロング
    weights[ranked > (n - k)]   = -1.0 / k   # ショート
    return weights


def build_mom_signal(cc_jp: pd.DataFrame, t_idx: int, L: int = 60) -> pd.Series:
    # 単純モメンタム：過去L日の平均リターン
    return cc_jp.iloc[max(0, t_idx - L): t_idx].mean()


def build_plain_pca_signal(
    cc: pd.DataFrame, t_idx: int, L: int = 60, K: int = 3
) -> pd.Series:
    # 正則化なしの素のPCA（ベースライン比較用）
    from .subspace_pca import compute_regularized_eigenvectors, compute_signal, US_TICKERS, ALL_TICKERS

    window = cc.iloc[t_idx - L: t_idx][ALL_TICKERS]
    mu     = window.mean()
    sigma  = window.std().replace(0, 1e-8)

    z_window = (window - mu) / sigma
    Ct = z_window.corr(min_periods=20).values
    Ct = np.where(np.isnan(Ct), 0.0, Ct)
    np.fill_diagonal(Ct, 1.0)

    # lambda=0 で正則化なし
    VU_K, VJ_K = compute_regularized_eigenvectors(Ct, np.eye(len(ALL_TICKERS)), lam=0.0, K=K)

    z_U = (cc.iloc[t_idx][US_TICKERS].values - mu[US_TICKERS].values) / sigma[US_TICKERS].values
    return pd.Series(compute_signal(z_U, VU_K, VJ_K), index=JP_TICKERS)
