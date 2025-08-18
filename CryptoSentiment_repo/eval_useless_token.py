# import glob
# import pandas as pd
# import numpy as np
# from model import Model

# # find your EA dataset on disk
# eb_files = glob.glob("data/eb_dataset_events_training_*.csv")
# eb_path = eb_files[0]
# df = pd.read_csv(eb_path)

# # init tokenizer
# tokenizer = Model("config.yaml").tokenizer

# # measure lengths
# lengths = [
#     len(tokenizer.encode(str(text), add_special_tokens=True))
#     for text in df["Tweet Content"].astype(str)
# ]

# print(f"Max tokens before truncation: {max(lengths)}")
# print(f"Mean tokens: {np.mean(lengths):.1f}")
# for p in [90, 95, 99]:
#     print(f"P{p}: {np.percentile(lengths, p):.0f}")

'''
Test for Eb dataset



'''

#!/usr/bin/env python3
import glob
import sys

import pandas as pd
import numpy as np
from model import Model

def main():
    # 1) load your tokenizer
    tokenizer = Model("config.yaml").tokenizer

    # 2) find the Eb CSV
    eb_files = glob.glob("data/eb_dataset_event_evaluation_*.csv")
    if not eb_files:
        print("❌ No Eb evaluation CSV found in data/eb_dataset_event_evaluation_*.csv")
        sys.exit(1)
    eb_path = eb_files[0]
    print(f"🔍 Loading Eb dataset: {eb_path}")

    # 3) read it
    df = pd.read_csv(eb_path)
    if "Tweet Content" not in df.columns:
        print("❌ 'Tweet Content' column not found in Eb CSV")
        sys.exit(1)

    # 4) measure pre-truncation token lengths
    lengths = [
        len(tokenizer.encode(str(txt), add_special_tokens=True))
        for txt in df["Tweet Content"].astype(str)
    ]

    max_len = int(np.max(lengths))
    mean_len = float(np.mean(lengths))
    p90 = float(np.percentile(lengths, 90))
    p95 = float(np.percentile(lengths, 95))
    p99 = float(np.percentile(lengths, 99))

    # 5) print summary
    print(f"\nEb tokens before truncation → max: {max_len}, mean: {mean_len:.1f}, "
          f"P90: {p90:.0f}, P95: {p95:.0f}, P99: {p99:.0f}\n")

    # 6) simple sanity checks
    model_max = getattr(tokenizer, "model_max_length", None)
    if model_max:
        assert max_len <= model_max, (
            f"Found Eb sequence length {max_len} > model_max_length {model_max}"
        )
    assert p99 <= 256, (
        f"99th percentile for Eb is {p99}, consider raising your max_length cap"
    )

    print("✅ Eb token‐length smoke-test passed")

if __name__ == "__main__":
    main()
