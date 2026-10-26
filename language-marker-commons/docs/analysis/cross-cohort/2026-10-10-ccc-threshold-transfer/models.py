"""Model factory functions for the CCC-threshold transfer experiment."""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_sgbt(seed: int) -> HistGradientBoostingClassifier:
    """Stochastic Gradient Boosting Tree (NaN-native)."""
    return HistGradientBoostingClassifier(
        max_iter=400,
        max_depth=4,
        min_samples_leaf=20,
        learning_rate=0.05,
        l2_regularization=0.1,
        class_weight="balanced",
        random_state=seed,
    )


def make_elasticnet(seed: int) -> Pipeline:
    """Elastic Net logistic regression (median imputation + standard scaling)."""
    return Pipeline([
        ("imp",   SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf",   LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=0.1,
            class_weight="balanced",
            max_iter=3000,
            random_state=seed,
        )),
    ])


MODELS: dict[str, callable] = {
    "sgbt":       make_sgbt,
    "elasticnet": make_elasticnet,
}
