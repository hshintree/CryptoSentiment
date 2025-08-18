import pandas as pd

df = pd.read_csv("signals_per_tweet.csv")
print(df["Pred_Label"].value_counts())