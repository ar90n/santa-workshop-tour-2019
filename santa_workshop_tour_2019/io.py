from pathlib import Path

import pandas as pd


def load_data(root: Path = Path("../input")):
    path = root / "santa-2019-workshop-scheduling" / "family_data.csv"
    return pd.read_csv(path, index_col='family_id')

def load_sample_submission(root: Path = Path("../input")):
    path = root / "santa-2019-workshop-scheduling" / "sample_submission.csv"
    return pd.read_csv(path, index_col='family_id')
