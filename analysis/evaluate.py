# パフォーマンス指標の計算と結果表示をまとめたモジュール。
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

JP_SECTOR_NAMES = {
    "1617.T": "食品",
    "1618.T": "エネルギー資源",
    "1619.T": "建設・資材",
    "1620.T": "素材・化学",
    "1621.T": "医薬品",
    "1622.T": "自動車・輸送機",
    "1623.T": "鉄鋼・非鉄",
    "1624.T": "機械",
    "1625.T": "電機・精密",
    "1626.T": "情報通信・サービス",
    "1627.T": "電力・ガス",
    "1628.T": "運輸・物流",
    "1629.T": "商社・卸売",
    "1630.T": "小売",
    "1631.T": "銀行",
    "1632.T": "金融（除く銀行）",
    "1633.T": "不動産",
}

US_SECTOR_NAMES = {
    "XLB":  "素材",
    "XLC":  "通信サービス",
    "XLE":  "エネルギー",
    "XLF":  "金融",
    "XLI":  "資本財",
    "XLK":  "テクノロジー",
    "XLP":  "生活必需品",
    "XLRE": "不動産",
    "XLU":  "公益事業",
    "XLV":  "ヘルスケア",
    "XLY":  "一般消費財",
}


def compute_metrics(ret: pd.Series) -> dict:
    """年率リターン・リスク・R/R・最大DDを計算する"""
    if len(ret) == 0:
        return {"AR": np.nan, "RISK": np.nan, "RR": np.nan, "MDD": np.nan}

    ar   = ret.mean() * 252
    risk = ret.std()  * np.sqrt(252)
    rr   = ar / risk if risk != 0 else np.nan

    cumret      = (1 + ret).cumprod()
    rolling_max = cumret.cummax()
    mdd         = ((cumret - rolling_max) / rolling_max).min()

    return {"AR": ar, "RISK": risk, "RR": rr, "MDD": mdd}


def print_signal_table(signal: pd.Series, signal_date: pd.Timestamp, q: float = 0.3):
    n = len(signal)
    k = max(1, int(np.floor(n * q)))
    sorted_sig = signal.sort_values(ascending=False)

    print(f"\n{'='*65}")
    print(f"  本日のシグナルランキング  ({signal_date.strftime('%Y-%m-%d')} 米国終値基準)")
    print(f"{'='*65}")
    print(f"  {'順位':>4}  {'銘柄コード':<10}  {'業種名':<18}  {'シグナル':>8}  推奨")
    print(f"  {'-'*58}")

    for rank, (ticker, val) in enumerate(sorted_sig.items(), 1):
        name = JP_SECTOR_NAMES.get(ticker, ticker)
        rec  = "▲ 買い" if rank <= k else ("▼ 売り" if rank > n - k else "")
        print(f"  {rank:>4}  {ticker:<10}  {name:<18}  {val:>+8.4f}  {rec}")

    print(f"{'='*65}")
    print(f"  買い: {', '.join(sorted_sig.head(k).index)}")
    print(f"  売り: {', '.join(sorted_sig.tail(k).index)}")
    print(f"{'='*65}\n")


def print_performance_table(results: dict):
    print(f"\n{'='*55}")
    print("  パフォーマンス比較（バックテスト結果）")
    print(f"{'='*55}")
    print(f"  {'戦略':<12}  {'年率リターン':>10}  {'リスク':>8}  {'R/R':>6}  {'最大DD':>8}")
    print(f"  {'-'*51}")

    for name, ret in results.items():
        m = compute_metrics(ret)
        print(f"  {name:<12}  {m['AR']*100:>9.2f}%  {m['RISK']*100:>7.2f}%  "
              f"{m['RR']:>6.2f}  {m['MDD']*100:>7.2f}%")

    print(f"{'='*55}\n")


def plot_cumulative_returns(results: dict, save_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {"PCA_SUB": "#1f77b4", "DOUBLE": "#ff7f0e", "PCA_PLAIN": "#2ca02c", "MOM": "#d62728"}
    styles = {"PCA_SUB": "-",       "DOUBLE": "--",       "PCA_PLAIN": "-.",       "MOM": ":"}

    for name, ret in results.items():
        if len(ret) == 0:
            continue
        cumret = (1 + ret).cumprod()
        ax.plot(cumret.index, cumret.values,
                label=name, color=colors.get(name), linestyle=styles.get(name, "-"), linewidth=1.5)

    ax.set_title("各戦略の累積リターン推移", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Wealth")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"グラフを保存しました: {save_path}")
    else:
        plt.show()
    plt.close(fig)
