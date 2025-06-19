from dataset_loader import DatasetLoader
import pandas as pd, os, glob, textwrap

print("Starting dataset loader test...")
dl  = DatasetLoader("config.yaml")

print("\n=== Loading Raw Dataset ===")
df  = dl.load_dataset(save_to_csv=True)           # raw (many tweets per day)
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
print(row[['date', 'Tweet Content', 'Close']].to_string())

print("\n=== Loading Aggregated Dataset (Fast Method) ===")
# Use the faster method that loads from saved CSV
try:
    day_df = dl.aggregate_saved_dataset()
    print("✓ Successfully loaded and aggregated saved dataset")
except FileNotFoundError:
    print("Saved dataset not found, falling back to full loading...")
    day_df = dl.load_dataset(aggregate=True, save_to_csv=True)

# Every date should now be unique and have ONE Close value
assert day_df['date'].is_unique, "duplicate dates after aggregation"
assert day_df['Close'].notna().all(),   "some aggregated rows lost their price"

print("✓ Per-day frame looks good:", day_df.shape)
print(day_df.head()[['date', 'Close']])

print("\n=== Test Complete ===")
print(f"Raw dataset: {df.shape}")
print(f"Aggregated dataset: {day_df.shape}")
print("Both datasets have been saved to the 'data' folder")