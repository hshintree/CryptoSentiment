# dataset_builder.py  ── build "2023-present" Bitcoin sentiment dataset
#
# Requirements (all free-tier):
#   pip install requests tweepy praw psaw pyarrow pandas tqdm ratelimit
#
# ── CONFIG ──────────────────────────────────────────────────────────────

FINNHUB_TOKEN      = "YOUR_FINNHUB_TOKEN"
X_BEARER_TOKEN     = "YOUR_X_BEARER"
REDDIT_CLIENT_ID   = "YOUR_REDDIT_ID"
REDDIT_SECRET      = "YOUR_REDDIT_SECRET"
SUBREDDITS         = ["Bitcoin", "btc", "CryptoCurrency"]
CRYPTO_SYMBOL      = "BINANCE:BTCUSDT"              # Finnhub format
START_DATE         = "2023-01-01"
END_DATE           = "2024-12-31"
TIMEFRAME_SEC      = 15*60                          # 15-minute candles
OUT_PARQUET        = "tweets_reddit_btc_2023_2024.parquet.gz"

# ── IMPORTS ─────────────────────────────────────────────────────────────
import os, time, math, json, requests, datetime as dt, pandas as pd
import tweepy, praw
from ratelimit import limits, sleep_and_retry
from tqdm.auto   import tqdm

# ── HELPERS ─────────────────────────────────────────────────────────────

def daterange(start: dt.datetime, end: dt.datetime, step_days=6):
    """Iterate [start,end] in 6-day chunks (fits free X 7-day window)."""
    cur = start
    delta = dt.timedelta(days=step_days)
    while cur < end:
        nxt = min(cur + delta, end)
        yield cur, nxt
        cur = nxt

# Finnhub – free tier = 60 req/min
@sleep_and_retry
@limits(calls=55, period=60)
def finnhub_get(endpoint: str, params: dict):
    params["token"] = FINNHUB_TOKEN
    r = requests.get(f"https://finnhub.io/api/v1/{endpoint}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_candles(from_ts: int, to_ts: int):
    return finnhub_get(
        "crypto/candle",
        {"symbol": CRYPTO_SYMBOL, "resolution": 15, "from": from_ts, "to": to_ts},
    )

def nearest_candle(ts, candles):
    idx = max(0, min(len(candles["t"])-1,
                     int((ts - candles["t"][0]) // TIMEFRAME_SEC)))
    return {k: candles[k][idx] for k in "c h l v".split()}

# ── X / TWITTER ─────────────────────────────────────────────────────────

tclient = tweepy.Client(bearer_token=X_BEARER_TOKEN, wait_on_rate_limit=True)

def fetch_tweets(q, start, end):
    tweets = []
    for s,e in daterange(start, end):
        for tw in tweepy.Paginator(
                tclient.search_recent_tweets,
                query=q,
                start_time=s.isoformat("T")+"Z",
                end_time  =e.isoformat("T")+"Z",
                tweet_fields = ["created_at","public_metrics","lang","author_id"],
                user_fields  = ["public_metrics"],
                expansions   = ["author_id"],
                max_results  = 100,
        ).flatten(limit=10000):                # 100 × 100 = 10 k / 15 min
            tweets.append(tw)
    return tweets

# ── REDDIT via PUSHSHIFT (psaw) ─────────────────────────────────────────

from psaw import PushshiftAPI
ps = PushshiftAPI()

def fetch_reddit(subs, start, end):
    posts = []
    gen = ps.search_submissions(after=int(start.timestamp()),
                                before=int(end.timestamp()),
                                subreddit=",".join(subs),
                                limit=5_000,
                                filter=["id","created_utc","subreddit_subscribers",
                                        "author","author_flair_text","score","title"])
    for p in gen:
        posts.append(p)
    return posts

# ── MAIN ────────────────────────────────────────────────────────────────

def main():
    since = dt.datetime.fromisoformat(START_DATE)
    until = dt.datetime.fromisoformat(END_DATE)

    all_rows = []
    print("⚡ Pulling data …")
    chunk = dt.timedelta(days=30)                # pull 1-month candle blocks

    for window_start in tqdm(list(daterange(since, until, 30))):
        ws, we = window_start
        cjson = fetch_candles(int(ws.timestamp()), int(we.timestamp()))
        if cjson.get("s") != "ok":
            continue

        # tweets
        q = "(bitcoin OR btc) lang:en -is:retweet"
        tws = fetch_tweets(q, ws, we)
        for tw in tws:
            candles = nearest_candle(int(tw.created_at.timestamp()), cjson)
            m = tw.public_metrics
            all_rows.append({
                "source": "twitter",
                "text": tw.text,
                "created_at": tw.created_at,
                "influence": m["followers_count"] if m else None,
                **candles
            })

        # reddit
        rds = fetch_reddit(SUBREDDITS, ws, we)
        for rd in rds:
            candles = nearest_candle(rd.created_utc, cjson)
            all_rows.append({
                "source": "reddit",
                "text": rd.title,
                "created_at": dt.datetime.utcfromtimestamp(rd.created_utc),
                "influence": rd.subreddit_subscribers + rd.score,
                **candles
            })

    # dataframe → parquet
    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT_PARQUET, compression="gzip", engine="pyarrow")
    print(f"✅ saved {len(df):,} rows to {OUT_PARQUET}")

if __name__ == "__main__":
    main()
