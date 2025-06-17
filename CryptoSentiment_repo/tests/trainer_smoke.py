import torch, yaml
from dataset_loader import DatasetLoader
from preprocessor    import Preprocessor
from market_labeler  import MarketLabeler
from trainer         import Trainer
from model           import Model      # your implementation

dl  = DatasetLoader("config.yaml")
pp  = Preprocessor("config.yaml")
ml  = MarketLabeler("config.yaml")

data = dl.load_dataset(aggregate=True)
data = pp.preprocess(data)
data = ml.label_data(data)
data['Previous Label'] = data['Label'].shift(1).fillna('Neutral')   # quick fix

model = Model("config.yaml")    # however you construct it
trainer = Trainer(model, data, "config.yaml")
trainer._prepare_data(data)     # should run with no errors
trainer.train()
print("✅ trainer smoke-prep passed")