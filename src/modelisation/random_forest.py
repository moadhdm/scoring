"""
Modele 2 : RANDOM FOREST (bagging d'arbres).
Peu de preparation, robuste a la colinearite. Pas de scaling necessaire.
Reglages volontairement maitrises (profondeur + feuilles) pour rester rapide.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # -> src/

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import config as cfg
from modelisation import features as F
from modelisation import evaluation as E

NAME  = "RandomForest"
SCALE = False


def build_model():
    return RandomForestClassifier(
        n_estimators=200, max_depth=16, min_samples_leaf=100,
        max_samples=0.5,                       # bootstrap 50% -> plus rapide
        class_weight="balanced_subsample", n_jobs=-1, random_state=0,
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
