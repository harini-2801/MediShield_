"""Hyperparameter catalogue: defaults, valid ranges, and plain-language explanations.

The Model Training page renders its controls from this catalogue, and
``build_models`` uses it to turn user-chosen values into scikit-learn
estimators. Keeping both in one place guarantees that what is explained
in the UI is exactly what is trained.
"""

from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


MODEL_INFO = {
    "Logistic Regression": {
        "icon": "📈",
        "summary": (
            "A linear model that estimates the probability of each class "
            "from a weighted sum of the features. Fast, stable and easy to "
            "interpret, but it can only draw straight decision boundaries."
        ),
    },
    "Random Forest": {
        "icon": "🌲",
        "summary": (
            "Many decision trees trained on random samples of the data; "
            "their votes are averaged. Handles non-linear patterns and mixed "
            "feature types well and is resistant to overfitting."
        ),
    },
    "Extra Trees": {
        "icon": "🌳",
        "summary": (
            "Like Random Forest, but split thresholds are chosen at random. "
            "This adds extra randomness, which often lowers variance and "
            "speeds up training."
        ),
    },
    "Support Vector Machine": {
        "icon": "🧭",
        "summary": (
            "Finds the boundary that separates classes with the widest "
            "margin, optionally in a kernel-transformed space. Strong on "
            "small/medium datasets; slower on large ones."
        ),
    },
}


_CLASS_WEIGHT = {
    "name": "class_weight",
    "label": "Class weight",
    "kind": "choice",
    "options": ["balanced", "none"],
    "default": "balanced",
    "help": (
        "'balanced' re-weights classes inversely to their frequency so rare "
        "outcomes (e.g. disease-positive cases) are not ignored. 'none' "
        "treats every record equally."
    ),
}

_N_ESTIMATORS = {
    "name": "n_estimators",
    "label": "Number of trees (n_estimators)",
    "kind": "int",
    "min": 10, "max": 1000, "step": 10, "default": 200,
    "help": (
        "How many trees are built. More trees give more stable predictions "
        "but take longer to train; gains flatten out after a few hundred."
    ),
}

_MAX_DEPTH = {
    "name": "max_depth",
    "label": "Maximum depth (0 = unlimited)",
    "kind": "int",
    "min": 0, "max": 50, "step": 1, "default": 0,
    "help": (
        "Longest path from root to leaf. Shallow trees underfit (too "
        "simple); unlimited depth can memorise noise (overfit)."
    ),
}

_MIN_SPLIT = {
    "name": "min_samples_split",
    "label": "Min samples to split a node",
    "kind": "int",
    "min": 2, "max": 50, "step": 1, "default": 2,
    "help": (
        "A node must hold at least this many records before it may be "
        "split. Larger values make the trees more conservative."
    ),
}

_MIN_LEAF = {
    "name": "min_samples_leaf",
    "label": "Min samples per leaf",
    "kind": "int",
    "min": 1, "max": 50, "step": 1, "default": 1,
    "help": (
        "Minimum records required in a final leaf. Higher values smooth "
        "the model and reduce overfitting to individual patients."
    ),
}

_MAX_FEATURES = {
    "name": "max_features",
    "label": "Features considered per split",
    "kind": "choice",
    "options": ["sqrt", "log2", "all"],
    "default": "sqrt",
    "help": (
        "How many features each split may look at. 'sqrt' and 'log2' "
        "decorrelate the trees; 'all' uses every feature at every split."
    ),
}


HYPERPARAMETERS = {
    "Logistic Regression": [
        {
            "name": "C",
            "label": "Inverse regularisation strength (C)",
            "kind": "float",
            "min": 0.001, "max": 100.0, "step": 0.001, "default": 1.0,
            "log": True,
            "help": (
                "Smaller C = stronger regularisation (simpler model, less "
                "overfitting). Larger C lets the model fit the training data "
                "more closely."
            ),
        },
        {
            "name": "max_iter",
            "label": "Maximum iterations",
            "kind": "int",
            "min": 100, "max": 10000, "step": 100, "default": 2000,
            "help": (
                "Upper limit on optimiser steps. Raise it if you see "
                "convergence warnings."
            ),
        },
        _CLASS_WEIGHT,
    ],
    "Random Forest": [
        _N_ESTIMATORS, _MAX_DEPTH, _MIN_SPLIT, _MIN_LEAF, _MAX_FEATURES,
        _CLASS_WEIGHT,
    ],
    "Extra Trees": [
        _N_ESTIMATORS, _MAX_DEPTH, _MIN_SPLIT, _MIN_LEAF, _MAX_FEATURES,
        _CLASS_WEIGHT,
    ],
    "Support Vector Machine": [
        {
            "name": "C",
            "label": "Margin penalty (C)",
            "kind": "float",
            "min": 0.01, "max": 100.0, "step": 0.01, "default": 1.0,
            "log": True,
            "help": (
                "Trade-off between a wide margin and classifying every "
                "training record correctly. High C = narrow margin, fewer "
                "training errors, higher overfitting risk."
            ),
        },
        {
            "name": "kernel",
            "label": "Kernel",
            "kind": "choice",
            "options": ["rbf", "linear", "poly", "sigmoid"],
            "default": "rbf",
            "help": (
                "Shape of the decision boundary. 'linear' = straight, "
                "'rbf' = flexible smooth curves (a good default), 'poly' = "
                "polynomial curves."
            ),
        },
        {
            "name": "gamma",
            "label": "Kernel coefficient (gamma)",
            "kind": "choice",
            "options": ["scale", "auto"],
            "default": "scale",
            "help": (
                "How far a single record's influence reaches (rbf/poly/"
                "sigmoid kernels). 'scale' adapts to the feature variance; "
                "'auto' uses 1 / n_features."
            ),
        },
        _CLASS_WEIGHT,
    ],
}


def default_params(model_name: str) -> dict:
    """Return the default hyperparameter values for a model."""
    return {
        spec["name"]: spec["default"]
        for spec in HYPERPARAMETERS[model_name]
    }


def _resolve(spec_name: str, value):
    """Translate UI-friendly values into scikit-learn arguments."""
    if spec_name == "class_weight":
        return None if value == "none" else value
    if spec_name == "max_depth":
        return None if not value else int(value)
    if spec_name == "max_features":
        return None if value == "all" else value
    return value


def build_models(
    selected: list[str],
    params: dict[str, dict] | None = None,
    random_state: int = 42,
) -> dict:
    """Create estimators for the selected models using the given params."""
    params = params or {}
    estimators = {}

    for name in selected:
        if name not in HYPERPARAMETERS:
            raise ValueError(f"Unknown model: {name}")

        merged = {**default_params(name), **params.get(name, {})}
        kwargs = {key: _resolve(key, val) for key, val in merged.items()}

        if name == "Logistic Regression":
            estimators[name] = LogisticRegression(
                random_state=random_state, **kwargs
            )
        elif name == "Random Forest":
            estimators[name] = RandomForestClassifier(
                random_state=random_state, n_jobs=-1, **kwargs
            )
        elif name == "Extra Trees":
            estimators[name] = ExtraTreesClassifier(
                random_state=random_state, n_jobs=-1, **kwargs
            )
        else:
            estimators[name] = SVC(
                probability=True, random_state=random_state, **kwargs
            )

    return estimators
