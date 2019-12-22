from typing import List
from pathlib import Path

import pandas as pd


def load_data(root: Path = Path("../input")) -> pd.DataFrame:
    path = root / "santa-workshop-tour-2019" / "family_data.csv"
    return pd.read_csv(path, index_col="family_id")


def load_sample_submission(root: Path = Path("../input")) -> pd.DataFrame:
    path = root / "santa-workshop-tour-2019" / "sample_submission.csv"
    return load_submission(path)


def load_submission(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="family_id")


def save_result(result: List[int], score: float) -> None:
    submission = load_sample_submission()
    submission["assigned_day"] = result
    submission.to_csv(f"submission_{score}.csv")
