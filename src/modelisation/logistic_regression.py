"""
Modele 1 : REGRESSION LOGISTIQUE.
Baseline interpretable (coefficients = sens de l'effet). Necessite scaling.
On utilise le jeu de features REDUIT (decorrele) pour des coefficients plus nets.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # -> src/

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import config as cfg
from modelisation import features as F
from modelisation import evaluation as E

NAME  = "LogReg"
SCALE = True       # la logistique a besoin de variables centrees-reduites


def build_model():
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)


def train_and_eval(gold: pd.DataFrame) -> dict:
    train, test = F.temporal_split(gold)
    cat, num = F.get_feature_lists()
    num = F.reduced_num_cols(num)                       # version decorrelee
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
