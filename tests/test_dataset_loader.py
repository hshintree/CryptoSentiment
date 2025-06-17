import pytest
pd = pytest.importorskip("pandas")
import yaml
from CryptoSentiment_repo.dataset_loader import DatasetLoader


def test_load_prebit(tmp_path):
    csv = tmp_path / "prebit.csv"
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=3),
        "tweet": ["a", "b", "c"],
        "close": [1, 2, 3],
    })
    df.to_csv(csv, index=False)

    cfg = {"data": {"prebit_dataset_path": str(csv)}, "market_labeling": {"barrier_window": "2-3"}}
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    loader = DatasetLoader(config_path=str(cfg_path))
    data = loader.load_prebit_data()
    assert {"Tweet Date", "Tweet Content", "Close"} <= set(data.columns)
