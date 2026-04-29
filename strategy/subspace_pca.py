# 論文のコアロジック：部分空間正則化付きPCAでシグナルを計算する。
# 事前部分空間（グローバル・国スプレッド・シクリカル）を使って
# 短いウィンドウでの推定誤差を抑えるのがポイント。
import numpy as np
import pandas as pd
from ..data.fetch_data import US_TICKERS, JP_TICKERS

ALL_TICKERS = US_TICKERS + JP_TICKERS
N_US = len(US_TICKERS)   # 11
N_JP = len(JP_TICKERS)   # 17
N    = N_US + N_JP        # 28

# シクリカル・ディフェンシブの分類（論文付録より）
US_CYCLIC    = ["XLB", "XLE", "XLF", "XLRE"]
US_DEFENSIVE = ["XLK", "XLP", "XLU", "XLV"]
JP_CYCLIC    = ["1618.T", "1625.T", "1629.T", "1631.T"]
JP_DEFENSIVE = ["1617.T", "1621.T", "1627.T", "1630.T"]


def _gram_schmidt_orthogonalize(v: np.ndarray, basis: list) -> np.ndarray:
    # グラムシュミット法で既存ベクトルと直交化する
    for b in basis:
        v = v - (v @ b) * b
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("直交化に失敗しました（ラベルを確認してください）")
    return v / norm


def build_prior_subspace() -> np.ndarray:
    """事前部分空間 V0 (28×3) を構築する"""
    ticker_index = {t: i for i, t in enumerate(ALL_TICKERS)}

    # v1: 全銘柄に等ウェイト（市場全体の動き）
    v1 = np.ones(N) / np.sqrt(N)

    # v2: 米国プラス・日本マイナス（日米の差）
    raw2 = np.zeros(N)
    for t in US_TICKERS:
        raw2[ticker_index[t]] =  1.0
    for t in JP_TICKERS:
        raw2[ticker_index[t]] = -1.0
    v2 = _gram_schmidt_orthogonalize(raw2, [v1])

    # v3: シクリカルプラス・ディフェンシブマイナス（景気感応度の差）
    raw3 = np.zeros(N)
    for t in US_CYCLIC + JP_CYCLIC:
        if t in ticker_index:
            raw3[ticker_index[t]] =  1.0
    for t in US_DEFENSIVE + JP_DEFENSIVE:
        if t in ticker_index:
            raw3[ticker_index[t]] = -1.0
    v3 = _gram_schmidt_orthogonalize(raw3, [v1, v2])

    return np.column_stack([v1, v2, v3])  # (28, 3)


def build_prior_correlation(z_full: pd.DataFrame, V0: np.ndarray) -> np.ndarray:
    """長期データからC0（事前相関行列）を構築する"""
    # XLC・XLREは上場前期間がないため、pairwiseで計算する
    C_full = z_full.corr(min_periods=30).values
    C_full = np.where(np.isnan(C_full), 0.0, C_full)
    np.fill_diagonal(C_full, 1.0)

    D0    = np.diag(V0.T @ C_full @ V0)
    C0_raw = V0 @ np.diag(D0) @ V0.T

    # 対角を1に正規化して相関行列の形に整える
    diag_vals   = np.where(np.diag(C0_raw) <= 0, 1e-8, np.diag(C0_raw))
    D_half_inv  = np.diag(1.0 / np.sqrt(diag_vals))
    C0 = D_half_inv @ C0_raw @ D_half_inv
    np.fill_diagonal(C0, 1.0)
    return C0


def compute_regularized_eigenvectors(
    Ct: np.ndarray, C0: np.ndarray, lam: float = 0.9, K: int = 3
) -> tuple:
    """正則化相関行列の上位K固有ベクトルを米国・日本ブロックに分割して返す"""
    C_reg = (1 - lam) * Ct + lam * C0

    eigenvalues, eigenvectors = np.linalg.eigh(C_reg)
    idx = np.argsort(eigenvalues)[::-1]
    Vt_K = eigenvectors[:, idx][:, :K]  # 降順で上位Kだけ取る

    return Vt_K[:N_US, :], Vt_K[N_US:, :]  # VU_K, VJ_K


def compute_signal(z_U: np.ndarray, VU_K: np.ndarray, VJ_K: np.ndarray) -> np.ndarray:
    """米国リターンをファクター空間に射影して日本側シグナルを復元する（式20）"""
    ft      = VU_K.T @ z_U   # 共通ファクタースコア
    return VJ_K @ ft          # 日本側への復元


class SubspacePCASignal:
    """
    毎日のシグナル計算をまとめたクラス。
    fit() で事前モデルを構築し、compute_signal_today() で当日シグナルを取得する。
    """

    def __init__(
        self,
        L: int  = 60,
        K: int  = 3,
        lam: float = 0.9,
        prior_start: str = "2010-01-01",
        prior_end:   str = "2014-12-31",
    ):
        self.L    = L
        self.K    = K
        self.lam  = lam
        self.prior_start = prior_start
        self.prior_end   = prior_end
        self.V0 = build_prior_subspace()
        self.C0 = None

    def fit(self, cc: pd.DataFrame):
        """C0（事前相関行列）をデータから推定する"""
        cc_prior = cc.loc[self.prior_start:self.prior_end, ALL_TICKERS]
        mu    = cc_prior.mean()
        sigma = cc_prior.std().replace(0, 1e-8)
        z_prior   = (cc_prior - mu) / sigma
        self.C0   = build_prior_correlation(z_prior, self.V0)
        self._cc  = cc[ALL_TICKERS].copy()
        return self

    def compute_signals_rolling(self) -> pd.DataFrame:
        """バックテスト用：全期間のシグナルをまとめて計算する"""
        cc    = self._cc
        dates = cc.index
        signals = {}

        for i in range(self.L, len(dates)):
            t            = dates[i]
            window_dates = dates[i - self.L: i]

            sub   = cc.loc[window_dates]
            mu    = sub.mean()
            sigma = sub.std().replace(0, 1e-8)

            z_window = (sub - mu) / sigma
            Ct = z_window.corr(min_periods=20).values
            Ct = np.where(np.isnan(Ct), 0.0, Ct)
            np.fill_diagonal(Ct, 1.0)

            VU_K, VJ_K = compute_regularized_eigenvectors(Ct, self.C0, self.lam, self.K)

            us_cols = [c for c in US_TICKERS if c in cc.columns]
            z_U = (cc.loc[t, us_cols].values - mu[us_cols].values) / sigma[us_cols].values

            signals[t] = dict(zip(JP_TICKERS, compute_signal(z_U, VU_K, VJ_K)))

        return pd.DataFrame(signals).T

    def compute_signal_today(self, cc_recent: pd.DataFrame) -> pd.Series:
        """直近L+1日のデータから当日のシグナルを計算する"""
        assert self.C0 is not None, "先にfit()を呼んでください"
        assert len(cc_recent) >= self.L + 1

        window    = cc_recent.iloc[-(self.L + 1):-1][ALL_TICKERS]
        today_row = cc_recent.iloc[-1][ALL_TICKERS]

        mu    = window.mean()
        sigma = window.std().replace(0, 1e-8)

        z_window = (window - mu) / sigma
        Ct = z_window.corr(min_periods=20).values
        Ct = np.where(np.isnan(Ct), 0.0, Ct)
        np.fill_diagonal(Ct, 1.0)

        VU_K, VJ_K = compute_regularized_eigenvectors(Ct, self.C0, self.lam, self.K)

        z_U = (today_row[US_TICKERS].values - mu[US_TICKERS].values) / sigma[US_TICKERS].values
        return pd.Series(compute_signal(z_U, VU_K, VJ_K), index=JP_TICKERS)
