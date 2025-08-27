from dataset_loader import DatasetLoader
from gpu_scripts.preprocessor    import Preprocessor
from market_labeler  import MarketLabeler

dl   = DatasetLoader("config.yaml")
pp   = Preprocessor("config.yaml")
ml   = MarketLabeler("config.yaml")

data = dl.load_dataset(aggregate=True)
data = pp.preprocess(data)
labeled = ml.label_data(data)

for col in ['Upper Barrier', 'Lower Barrier', 'Vertical Barrier', 'Label']:
    assert col in labeled.columns, f"{col} missing!"
print("✓ Labeler added barrier columns. Sample:")
print(labeled[['Tweet Date','Close','Upper Barrier','Lower Barrier','Label']].head())