from dataset_loader import DatasetLoader
from preprocessor import Preprocessor

dl   = DatasetLoader("config.yaml")
pp   = Preprocessor("config.yaml")

raw  = dl.load_dataset(aggregate=True)      # one row per day is easiest
proc = pp.preprocess(raw)

expect = {'RSI', 'ROC'}
missing = expect.difference(proc.columns)
assert not missing, f"❌ Preprocessor did not add {missing}"
print("✓ Preprocessor added RSI & ROC, shape:", proc.shape)