'''
pulling current data from finnhub
and saving it to a parquet file
'''
# fetch_finnhub_btc.py
import os, time, datetime as dt, requests, pandas as pd
from pathlib import Path

API_KEY = "d1n18rpr01qlvnp50uqgd1n18rpr01qlvnp50ur0"
BASE    = "https://finnhub.io/api/v1"

OUT_TWEETS = Path("data/raw_tweets_2023_now.parquet")
OUT_OHLCV  = Path("data/raw_ohlcv_2023_now.parquet")

HEADERS = {"X-Finnhub-Token": API_KEY}

def daterange(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)

def get(endpoint: str, **params):
    params["token"] = API_KEY
    r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    start = dt.date(2023, 1, 1)
    end   = dt.date.today()

    tweet_rows, price_rows = [], []

    for day in daterange(start, end):
        nxt = day + dt.timedelta(days=1)

        # --- social sentiment (tweets + reddit) ---
        js = get("stock/social-sentiment",
                 symbol="BINANCE:BTCUSDT",
                 _from=day.isoformat(),
                 to=nxt.isoformat())
        for src in ("twitter",):  # we ignore reddit for now
            for obj in js.get(src, []):
                tweet_rows.append({
                    "datetime": obj["atTime"],      # e.g. '2024-05-07 15:22:00'
                    "text":      obj["headline"],
                    "followers": obj["followers"],
                    "sentiment": obj["sentiment"],  # -1..1
                    "ticker":    obj["ticker"],
                })

        # --- OHLCV candle (1-day) ---
        ts_from = int(dt.datetime.combine(day, dt.time()).timestamp())
        ts_to   = int(dt.datetime.combine(nxt, dt.time()).timestamp())
        candles = get("crypto/candle",
                      symbol="BINANCE:BTCUSDT",
                      resolution="D",
                      _from=ts_from, to=ts_to)
        if candles.get("s") == "ok":
            price_rows.append({
                "date": day,
                "open":  candles["o"][0],
                "high":  candles["h"][0],
                "low":   candles["l"][0],
                "close": candles["c"][0],
                "volume":candles["v"][0],
            })

        # ---- polite(rate) ----
        time.sleep(1.0)          # 1 req/s = 60/min  << free cap

    # ---- persist ----
    pd.DataFrame(tweet_rows).to_parquet(OUT_TWEETS, index=False)
    pd.DataFrame(price_rows).to_parquet(OUT_OHLCV, index=False)
    print(f"Saved {OUT_TWEETS} & {OUT_OHLCV}")

if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("export FINNHUB_API_KEY=<your-key>")
    main()
