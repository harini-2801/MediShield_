"""Tests for the shared stratified train/test split."""

import numpy as np
import pandas as pd

from src.splitting import prepare_split, split_report


def _df(n=200, pos_share=0.3, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    df["patient_id"] = range(n)
    df["y"] = np.where(rng.random(n) < pos_share, "sick", "healthy")
    return df


def test_split_is_stratified_and_disjoint():
    df = _df()
    s = prepare_split(df, "y", test_size=0.25, random_state=1)
    assert len(s["X_train"]) + len(s["X_test"]) == len(df)
    assert not set(s["X_train"].index) & set(s["X_test"].index)
    assert "patient_id" not in s["X_train"].columns
    train_share = (s["y_train"] == 1).mean()
    test_share = (s["y_test"] == 1).mean()
    assert abs(train_share - test_share) < 0.03


def test_split_is_reproducible():
    df = _df()
    a = prepare_split(df, "y", 0.2, 7)["X_test"].index
    b = prepare_split(df, "y", 0.2, 7)["X_test"].index
    assert list(a) == list(b)


def test_report_healthy():
    r = split_report(_df(), "y", 0.2, 42)
    assert r["valid"] and r["test_rows"] + r["train_rows"] == 200
    assert {c["class"] for c in r["classes"]} == {"sick", "healthy"}
    assert all(c["train"] > 0 and c["test"] > 0 for c in r["classes"])


def test_report_rejects_single_record_class():
    df = _df()
    df.loc[0, "y"] = "rare"
    r = split_report(df, "y")
    assert not r["valid"] and "rare" in r["error"]


def test_report_flags_duplicates_and_small_test():
    df = _df(n=60)
    df = pd.concat([df, df], ignore_index=True)  # every row duplicated
    r = split_report(df, "y", 0.2, 0)
    assert r["valid"] and r["duplicate_rows_across_split"] > 0
    assert any("identical" in w for w in r["warnings"])


def test_report_excludes_missing_labels():
    df = _df()
    df.loc[:9, "y"] = np.nan
    r = split_report(df, "y")
    assert r["total_rows"] == 190
    assert any("missing target" in w for w in r["warnings"])
