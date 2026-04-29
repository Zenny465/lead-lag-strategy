# yfinanceでETF価格を取得する。2回目以降はCSVキャッシュから読む。
import yfinance as yf
import pandas as pd
from pathlib import Path

US_TICKERS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
JP_TICKERS = [
    "1617.T", "1618.T", "1619.T", "1620.T", "1621.T", "1622.T", "1623.T",
    "1624.T", "1625.T", "1626.T", "1627.T", "1628.T", "1629.T", "1630.T",
    "1631.T", "1632.T", "1633.T",
]
ALL_TICKERS = US_TICKERS + JP_TICKERS

# キャッシュはプロジェクト直下に置く
CACHE_DIR = Path(__file__).parent.parent / "cache"


def fetch_prices(start: str = "2010-01-01", end: str = None, use_cache: bool = True) -> dict:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_close = CACHE_DIR / "close.csv"
    cache_open  = CACHE_DIR / "open.csv"

    if use_cache and cache_close.exists() and cache_open.exists():
        close = pd.read_csv(cache_close, index_col=0, parse_dates=True)
        open_ = pd.read_csv(cache_open,  index_col=0, parse_dates=True)

        last_cached = close.index[-1]
        today = pd.Timestamp.now().normalize()

        # 前日まで取得済みなら再ダウンロード不要
        if (today - last_cached).days <= 1:
            return {"close": close, "open": open_}

        new_start = (last_cached + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        new_data  = _download(new_start, end)

        if new_data["close"] is not None and len(new_data["close"]) > 0:
            close = pd.concat([close, new_data["close"]]).drop_duplicates()
            open_ = pd.concat([open_, new_data["open"]]).drop_duplicates()
            close.to_csv(cache_close)
            open_.to_csv(cache_open)

        return {"close": close, "open": open_}

    data = _download(start, end)
    if data["close"] is not None:
        data["close"].to_csv(cache_close)
        data["open"].to_csv(cache_open)
    return data


def _download(start: str, end: str = None) -> dict:
    print(f"価格データをダウンロード中 ({start} 〜) ...")
    raw = yf.download(ALL_TICKERS, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        open_ = raw["Open"]
    else:
        close = raw[["Close"]]
        open_ = raw[["Open"]]

    # 祝日などの欠損は前日の値で補完
    close = close.ffill()
    open_ = open_.ffill()

    return {"close": close, "open": open_}


def get_common_dates(close: pd.DataFrame) -> pd.DatetimeIndex:
    return close.dropna(how="any").index
