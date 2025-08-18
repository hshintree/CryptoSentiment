import yaml
from dataset_loader import DatasetLoader
from preprocessor    import Preprocessor
from market_labeler  import MarketLabeler
from model           import Model
from trainer         import Trainer

CFG = "config.yaml"                        # single source-of-truth
dl  = DatasetLoader(CFG)

# ---- 1. load ALL tweets (multiple per day) so each tweet gets its own label
data = dl.load_dataset(aggregate=False)    # ~340 k rows

# ---- 2. technical feature engineering
data = Preprocessor(CFG).preprocess(data)  # adds RSI, ROC, etc.

# ---- 3. market-derived Triple-Barrier labels
data = MarketLabeler(CFG).label_data(data) # Bullish / Neutral / Bearish

# ---- 4. previous-window label (used in prompts)
data["Previous Label"] = data["Label"].shift(1).fillna("Neutral")

# ---- 5. build model from YAML → dict
with open(CFG) as f:
    params = yaml.safe_load(f)["model"]
model = Model(params)

# ---- 6. train – 5-fold grouped CV exactly like the paper
Trainer(model, data, CFG).train()
