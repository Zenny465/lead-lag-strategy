# -*- coding: utf-8 -*-
import sys
import io
import os
import warnings
warnings.filterwarnings("ignore")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.rcParams["font.family"] = "Noto Sans JP"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lead_lag_strategy.data.fetch_data import fetch_prices, US_TICKERS, JP_TICKERS, ALL_TICKERS
from lead_lag_strategy.data.preprocess import compute_returns
from lead_lag_strategy.strategy.subspace_pca import SubspacePCASignal
from lead_lag_strategy.analysis.evaluate import JP_SECTOR_NAMES, US_SECTOR_NAMES

L = 60
K = 3
LAM = 0.9
Q = 0.3
PRIOR_START = "2010-01-01"
PRIOR_END   = "2014-12-31"

HERE = os.path.dirname(os.path.abspath(__file__))
PNG_PATH = os.path.join(HERE, "today_signal.png")


def save_signal_chart(signal: pd.Series, us_ret: pd.Series, signal_date: pd.Timestamp):
    n = len(signal)
    k = max(1, int(np.floor(n * Q)))
    sorted_sig = signal.sort_values(ascending=False)

    colors = []
    for i, _ in enumerate(sorted_sig.index):
        if i < k:
            colors.append("#27ae60")      # 買い → 緑
        elif i >= n - k:
            colors.append("#e74c3c")      # 売り → 赤
        else:
            colors.append("#95a5a6")      # 中立 → グレー

    labels = [f"{t}\n{JP_SECTOR_NAMES.get(t,'')}" for t in sorted_sig.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                    gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor("#1a1a2e")

    # --- 左：日本業種シグナルバー ---
    ax1.set_facecolor("#16213e")
    ax1.barh(range(n), sorted_sig.values[::-1], color=colors[::-1], height=0.6)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(labels[::-1], fontsize=8.5, color="white")
    ax1.set_xlabel("シグナル強度", color="white", fontsize=10)
    ax1.set_title(f"本日の投資シグナル  {signal_date.strftime('%Y-%m-%d')}",
                  color="white", fontsize=13, fontweight="bold", pad=12)
    ax1.axvline(0, color="white", linewidth=0.8, alpha=0.5)
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#444466")

    # 数値ラベル
    for i, v in enumerate(sorted_sig.values[::-1]):
        ax1.text(v + (0.002 if v >= 0 else -0.002), i,
                 f"{v:+.4f}",
                 va="center", ha="left" if v >= 0 else "right",
                 fontsize=7.5, color="white")

    # 凡例
    patches = [
        mpatches.Patch(color="#27ae60", label=f"▲ 買い推奨（上位{k}銘柄）"),
        mpatches.Patch(color="#e74c3c", label=f"▼ 売り推奨（下位{k}銘柄）"),
        mpatches.Patch(color="#95a5a6", label="中立"),
    ]
    ax1.legend(handles=patches, loc="lower right",
               facecolor="#16213e", edgecolor="#444466",
               labelcolor="white", fontsize=9)

    # --- 右：米国業種リターン（参考） ---
    ax2.set_facecolor("#16213e")
    us_sorted = us_ret.sort_values(ascending=False)
    us_labels = [f"{t}\n{US_SECTOR_NAMES.get(t,'')}" for t in us_sorted.index]
    us_colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in us_sorted.values]

    ax2.barh(range(len(us_sorted)), us_sorted.values[::-1],
             color=us_colors[::-1], height=0.6)
    ax2.set_yticks(range(len(us_sorted)))
    ax2.set_yticklabels(us_labels[::-1], fontsize=8.5, color="white")
    ax2.set_xlabel("前日リターン", color="white", fontsize=10)
    ax2.set_title("米国業種リターン（前日）",
                  color="white", fontsize=12, fontweight="bold", pad=12)
    ax2.axvline(0, color="white", linewidth=0.8, alpha=0.5)
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444466")

    for i, v in enumerate(us_sorted.values[::-1]):
        ax2.text(v + (0.0005 if v >= 0 else -0.0005), i,
                 f"{v*100:+.2f}%",
                 va="center", ha="left" if v >= 0 else "right",
                 fontsize=7.5, color="white")

    plt.tight_layout(pad=2.0)
    fig.savefig(PNG_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def print_signal_table(signal: pd.Series, signal_date: pd.Timestamp):
    n = len(signal)
    k = max(1, int(np.floor(n * Q)))
    sorted_sig = signal.sort_values(ascending=False)

    print(f"\n{'='*65}")
    print(f"  本日のシグナルランキング  ({signal_date.strftime('%Y-%m-%d')} 米国終値基準)")
    print(f"{'='*65}")
    print(f"  {'順位':>4}  {'銘柄':^10}  {'業種名':<18}  {'シグナル':>8}  推奨")
    print(f"  {'-'*58}")
    for rank, (ticker, val) in enumerate(sorted_sig.items(), 1):
        name = JP_SECTOR_NAMES.get(ticker, ticker)
        rec = "▲ 買い" if rank <= k else ("▼ 売り" if rank > n - k else "")
        print(f"  {rank:>4}  {ticker:<10}  {name:<18}  {val:>+8.4f}  {rec}")
    print(f"{'='*65}")
    print(f"  買い: {', '.join(sorted_sig.head(k).index)}")
    print(f"  売り: {', '.join(sorted_sig.tail(k).index)}")
    print(f"{'='*65}\n")


def main():
    print("=" * 65)
    print("  日米業種リードラグ投資戦略システム")
    print("  Lead-Lag Strategy: Japan / U.S. Sector ETFs")
    print("=" * 65)

    print("\nデータを取得中 ...")
    prices = fetch_prices(start="2010-01-01", use_cache=True)
    close = prices["close"].ffill()
    open_ = prices["open"].ffill()
    print(f"取得完了: {close.index[0].date()} 〜 {close.index[-1].date()}  ({len(close)} 営業日)")

    ret = compute_returns(close, open_)
    cc = ret["cc"]

    print("シグナルを計算中 ...")
    model = SubspacePCASignal(L=L, K=K, lam=LAM, prior_start=PRIOR_START, prior_end=PRIOR_END)
    model.fit(cc)

    cc_recent = cc[ALL_TICKERS].iloc[-(L + 1):]
    signal = model.compute_signal_today(cc_recent)
    signal_date = cc_recent.index[-1]
    us_ret = cc_recent.iloc[-1][US_TICKERS]

    print_signal_table(signal, signal_date)

    print(f"グラフを保存中 → {PNG_PATH}")
    save_signal_chart(signal, us_ret, signal_date)
    print(f"保存完了: today_signal.png")
    print("\n完了。グラフファイルをダブルクリックすると画像が開きます。")


if __name__ == "__main__":
    main()
