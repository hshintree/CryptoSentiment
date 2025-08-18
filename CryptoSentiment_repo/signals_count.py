import pandas as pd

df = pd.read_csv("data/labeled_test.csv")
print(df["Label"].value_counts())