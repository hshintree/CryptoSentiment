from dataset_loader import DatasetLoader
import pandas as pd, os, glob, textwrap

dl  = DatasetLoader("config.yaml")
df  = dl.load_dataset()           # raw (many tweets per day)
cols = list(df.columns)

print("\n─ Columns ({} total) ─".format(len(cols)))
print(textwrap.fill(", ".join(cols), width=88))

# 1️⃣ Is there a Close column?
assert 'Close' in df.columns, "❌ No `Close` column – price merge probably failed"

# 2️⃣ How many tweets have *any* price info?
pct_price = 100 * df['Close'].notna().mean()
print(f"\n✓ {pct_price:05.2f}% of tweet rows carry a Close price")

# 3️⃣ Show the first date that *does* have price
row = df[df['Close'].notna()].iloc[0]
print("\nExample (first available):")
print(row[['Tweet Date', 'Tweet Content', 'Close']].to_string())

#Quick check for per day aggregation
dl = DatasetLoader("config.yaml")
day_df = dl.load_dataset(aggregate=True)

# Every date should now be unique and have ONE Close value
assert day_df['Tweet Date'].is_unique, "duplicate dates after aggregation"
assert day_df['Close'].notna().all(),   "some aggregated rows lost their price"

print("✓ Per-day frame looks good:", day_df.shape)
print(day_df.head()[['Tweet Date', 'Close']])