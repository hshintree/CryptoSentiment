import pytest
pd = pytest.importorskip("pandas")
import yaml
from CryptoSentiment_repo.market_labeler import MarketLabeler


def test_triple_barrier_labeling(tmp_path):
    df = pd.DataFrame({
        "Close": [10, 11, 12, 11, 10, 9, 11, 13],
        "Tweet Date": pd.date_range("2020-01-01", periods=8),
        "Tweet Content": ["x"] * 8,
    })
    cfg = {"market_labeling": {"barrier_window": "2-2"}}
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    labeler = MarketLabeler(config_path=str(cfg_path))
    labeled = labeler.label_data(df)
    assert set(labeled["Label"].unique()) <= {"Bullish", "Bearish", "Neutral"}
    assert "Upper Barrier" in labeled.columns
    assert "Lower Barrier" in labeled.columns
    assert "Vertical Barrier" in labeled.columns
