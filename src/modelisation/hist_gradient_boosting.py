"""
Modele 3 : HIST GRADIENT BOOSTING (boosting, equivalent LightGBM natif sklearn).
=> MODELE RETENU : meilleure PR-AUC sur le test (metrique reine en fraude).
Hyperparametres choisis apres reglage (bon compromis PR-AUC / surapprentissage).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # -> src/

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

import config as cfg
from modelisation import features as F
from modelisation import evaluation as E

NAME  = "HistGradientBoosting"
SCALE = False


def build_model():
    return HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=3.0, min_samples_leaf=300,
        class_weight="balanced",
        early_stopping=True, validation_fraction=0.1, random_state=0,
    )


def train_and_eval(gold: pd.DataFrame) -> dict:
    train, test = F.temporal_split(gold)
    cat, num = F.get_feature_lists()
    pipe = Pipeline([("pre", F.make_preprocessor(cat, num, SCALE)),
                     ("clf", build_model())])
    pipe.fit(train[cat + num], train["fraude"])
    return E.full_report(NAME, pipe, train[cat + num], train["fraude"],
                         test[cat + num], test["fraude"])


if __name__ == "__main__":
    print(f">>> Modele : {NAME}")
    gold = pd.read_parquet(cfg.GOLD_PARQUET)
    row = train_and_eval(gold)
    pd.DataFrame([row]).to_csv(cfg.METRICS_DIR / f"metrics_{NAME.lower()}.csv", index=False)
