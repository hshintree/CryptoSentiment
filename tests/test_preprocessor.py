import pytest
pd = pytest.importorskip("pandas")
import yaml
from CryptoSentiment_repo.preprocessor import Preprocessor


def test_preprocess_indicators(tmp_path):
    df = pd.DataFrame({
        "Tweet Date": pd.date_range("2020-01-01", periods=5),
        "Tweet Content": ["test"] * 5,
        "Close": [1, 2, 3, 2, 4],
    })
    cfg = {
        "data": {
            "preprocessing_steps": {
                "text_normalization": True,
                "remove_urls": False,
                "remove_user_ids": False,
                "remove_punctuation": False,
                "lemmatization": False,
            },
            "rsi_threshold": [30, 70],
            "roc_window_length": 2,
        }
    }
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    preprocessor = Preprocessor(config_path=str(cfg_path))
    out = preprocessor.preprocess(df.copy())
    assert "RSI" in out.columns
    assert "ROC" in out.columns
