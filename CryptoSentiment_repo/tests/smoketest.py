#!/usr/bin/env python3
import nltk

# auto-download WordNet if needed
for res in ('wordnet', 'omw-1.4'):
    try: nltk.data.find(f'corpora/{res}')
    except LookupError: nltk.download(res, quiet=True)

from dataset_loader import DatasetLoader
from gpu_scripts.preprocessor    import Preprocessor
from market_labeler  import MarketLabeler

dl  = DatasetLoader("config.yaml")
pp  = Preprocessor("config.yaml")
ml  = MarketLabeler("config.yaml")

df   = dl.load_dataset(aggregate=True)
df   = pp.preprocess(df)
df   = ml.label_data(df)

assert 'Close' in df.columns and df['Close'].notna().all()
assert 'Label' in df.columns

print("🔄 Smoke-test finished — final shape", df.shape)
