"""Stratified train/test splitting shared by the split check and training.

Both the preview shown at upload time and the actual training call
``prepare_split``, so the split a user inspects is exactly the split the
models are trained and evaluated on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.preprocessing import split_features_target

IDENTIFIER_COLUMNS = {
    "id",
    "patient_id",
    "record_id",
}

MIN_TEST_ROWS_WARNING = 30
MIN_CLASS_TEST_WARNING = 5
IMBALANCE_WARNING_PCT = 10.0


def prepare_split(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Validate the data and perform a stratified train/test split."""
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("Test size must be between 0.1 and 0.4.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist.")

    # Remove rows with missing target labels.
    labeled_df = df.loc[df[target_column].notna()].copy()

    if labeled_df.empty:
        raise ValueError("The selected target contains no labeled records.")

    X, y = split_features_target(labeled_df, target_column)

    # Identifiers should identify records, not predict disease.
    identifier_columns = [
        column
        for column in X.columns
        if str(column).strip().lower() in IDENTIFIER_COLUMNS
    ]
    X = X.drop(columns=identifier_columns)

    if X.empty:
        raise ValueError("No predictive features remain after excluding IDs.")

    # Encode target labels consistently for the saved pipeline.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str))

    class_counts = pd.Series(y_encoded).value_counts()
    n_classes = len(class_counts)

    if n_classes < 2:
        raise ValueError("The target must contain at least two classes.")

    if class_counts.min() < 2:
        rare = [
            f"'{label_encoder.classes_[i]}' ({n} record)"
            for i, n in class_counts.items()
            if n < 2
        ]
        raise ValueError(
            "Every target class needs at least two records for a "
            "stratified train/test split. Too small: " + ", ".join(rare)
        )

    n_test = max(int(np.ceil(len(y_encoded) * test_size)), n_classes)

    if n_test >= len(y_encoded) or len(y_encoded) - n_test < n_classes:
        raise ValueError("There are too few records for this stratified split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    return {
        "X": X,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "label_encoder": label_encoder,
        "identifier_columns": identifier_columns,
        "dropped_unlabeled": int(len(df) - len(labeled_df)),
    }


def _row_hashes(frame: pd.DataFrame) -> set:
    return set(pd.util.hash_pandas_object(frame, index=False).tolist())


def split_report(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Describe the split and flag anything that could make it misleading."""
    try:
        split = prepare_split(df, target_column, test_size, random_state)
    except ValueError as exc:
        return {"valid": False, "error": str(exc)}

    encoder = split["label_encoder"]
    y_train, y_test = split["y_train"], split["y_test"]
    n_train, n_test = len(y_train), len(y_test)
    total = n_train + n_test

    classes = []
    for index, name in enumerate(encoder.classes_):
        in_train = int((y_train == index).sum())
        in_test = int((y_test == index).sum())
        classes.append({
            "class": str(name),
            "total": in_train + in_test,
            "train": in_train,
            "test": in_test,
            "share_of_data_pct": round((in_train + in_test) / total * 100, 1),
            "train_pct": round(in_train / n_train * 100, 1),
            "test_pct": round(in_test / n_test * 100, 1),
        })

    warnings = []

    if split["dropped_unlabeled"]:
        warnings.append(
            f"{split['dropped_unlabeled']} rows with a missing target label "
            "were excluded from the split."
        )
    if split["identifier_columns"]:
        warnings.append(
            "ID columns excluded from the features: "
            + ", ".join(map(str, split["identifier_columns"]))
        )
    if n_test < MIN_TEST_ROWS_WARNING:
        warnings.append(
            f"The test set has only {n_test} rows, so evaluation metrics "
            "will be unreliable."
        )

    smallest = min(classes, key=lambda c: c["test"])
    if smallest["test"] < MIN_CLASS_TEST_WARNING:
        warnings.append(
            f"Class '{smallest['class']}' has only {smallest['test']} test "
            "records; its precision/recall will be very noisy."
        )

    rarest = min(classes, key=lambda c: c["share_of_data_pct"])
    if rarest["share_of_data_pct"] < IMBALANCE_WARNING_PCT:
        warnings.append(
            f"Class '{rarest['class']}' is only "
            f"{rarest['share_of_data_pct']}% of the data. Training uses "
            "balanced class weights; judge models by balanced accuracy and "
            "macro F1, not plain accuracy."
        )

    # Identical feature rows on both sides of the split inflate the scores.
    train_hashes = _row_hashes(split["X_train"])
    leaked = int(sum(
        h in train_hashes
        for h in pd.util.hash_pandas_object(
            split["X_test"], index=False
        ).tolist()
    ))
    if leaked:
        warnings.append(
            f"{leaked} test rows have feature values identical to a "
            "training row (duplicates). This can make test scores look "
            "better than they really are."
        )

    return {
        "valid": True,
        "total_rows": int(total),
        "train_rows": int(n_train),
        "test_rows": int(n_test),
        "test_pct": round(n_test / total * 100, 1),
        "feature_count": int(split["X"].shape[1]),
        "classes": classes,
        "duplicate_rows_across_split": leaked,
        "stratified": True,
        "random_state": int(random_state),
        "warnings": warnings,
    }
