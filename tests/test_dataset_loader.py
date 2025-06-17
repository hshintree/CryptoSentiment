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
    data = loader.load_dataset()
    assert {"Tweet Date", "Tweet Content", "Close"} <= set(data.columns)


def test_load_prebit_dir(tmp_path):
    dir_path = tmp_path / "prebit"
    dir_path.mkdir()
    for year in [2019, 2020]:
        df = pd.DataFrame({
            "date": pd.date_range(f"{year}-01-01", periods=2),
            "text_split": ["a", "b"],
        })
        df.to_csv(dir_path / f"combined_tweets_{year}_labeled.csv", index=False)

    price = pd.DataFrame({
        "date": pd.date_range("2019-01-01", periods=2),
        "Close_x": [1.0, 2.0],
    })
    price.to_csv(dir_path / "price_label.csv", index=False)

    cfg = {"data": {"prebit_dataset_dir": str(dir_path)}, "market_labeling": {"barrier_window": "2-3"}}
    cfg_path = tmp_path / "config2.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    loader = DatasetLoader(config_path=str(cfg_path))
    data = loader.load_dataset()
    assert len(data) == 4
    assert "Close" in data.columns
