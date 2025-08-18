from pathlib import Path
import pandas as pd
from dataset_loader import DatasetLoader

class BigLoader(DatasetLoader):
    """
    Specialized loader for the complete 2015-2023 dataset.
    Focuses on core features only: tweets, prices, and volume.
    Combines both PreBit (2015-2021) and Kaggle (2021-2023) data.
    """
    