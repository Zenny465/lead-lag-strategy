# 4戦略（MOM / PCA_PLAIN / PCA_SUB / DOUBLE）のバックテストを実行する。
# 戦略リターンは日本ETFのOpen-to-Closeリターンで評価する。
import pandas as pd
from ..data.fetch_data import JP_TICKERS, ALL_TICKERS
from ..strategy.portfolio import long_short_weights, build_mom_signal, build_plain_pca_signal
from ..strategy.subspace_pca import SubspacePCASignal


def run_backtest(
    cc: pd.DataFrame,
    oc: pd.DataFrame,
    L: int   = 60,
    K: int   = 3,
    lam: float = 0.9,
    q: float   = 0.3,
    prior_start: str = "2010-01-01",
    prior_end:   str = "2014-12-31",
) -> dict:
    cc_all = cc[ALL_TICKERS]
    cc_jp  = cc[JP_TICKERS]
    dates  = cc_all.index

    print("部分空間正則化PCAモデルを構築中 ...")
    model = SubspacePCASignal(L=L, K=K, lam=lam, prior_start=prior_start, prior_end=prior_end)
    model.fit(cc_all)
    sub_signals = model.compute_signals_rolling()  # 日付tのシグナル → t+1で使用

    ret_mom, ret_plain, ret_sub, ret_double = {}, {}, {}, {}

    print("バックテスト実行中 ...")
    for i in range(L, len(dates) - 1):
        t  = dates[i]
        t1 = dates[i + 1]
        if t1 not in oc.index:
            continue

        oc_t1 = oc.loc[t1, JP_TICKERS]

        # MOM
        w = long_short_weights(build_mom_signal(cc_jp, i, L), q)
        ret_mom[t1] = float((w * oc_t1).sum())

        # PCA_PLAIN
        w = long_short_weights(build_plain_pca_signal(cc_all, i, L, K), q)
        ret_plain[t1] = float((w * oc_t1).sum())

        if t not in sub_signals.index:
            continue

        sig_sub = sub_signals.loc[t, JP_TICKERS]

        # PCA_SUB（提案手法）
        w = long_short_weights(sig_sub, q)
        ret_sub[t1] = float((w * oc_t1).sum())

        # DOUBLE：MOMとPCA_SUBを2×2ソートで組み合わせる
        sig_mom = build_mom_signal(cc_jp, i, L)
        long_set  = sig_mom[sig_mom >= sig_mom.median()].index.intersection(
                        sig_sub[sig_sub >= sig_sub.median()].index)
        short_set = sig_mom[sig_mom <  sig_mom.median()].index.intersection(
                        sig_sub[sig_sub <  sig_sub.median()].index)

        w = pd.Series(0.0, index=JP_TICKERS)
        if len(long_set)  > 0: w[long_set]  =  1.0 / len(long_set)
        if len(short_set) > 0: w[short_set] = -1.0 / len(short_set)
        ret_double[t1] = float((w * oc_t1).sum())

    return {
        "MOM":       pd.Series(ret_mom),
        "PCA_PLAIN": pd.Series(ret_plain),
        "PCA_SUB":   pd.Series(ret_sub),
        "DOUBLE":    pd.Series(ret_double),
    }
