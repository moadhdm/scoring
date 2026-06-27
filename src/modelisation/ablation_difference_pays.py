"""
Ablation : la famille FM_Difference_Pays apporte-t-elle quelque chose ?
L'IV univariee la donne quasi nulle (IV ~ 0). On verifie sur le MEILLEUR modele
(HistGB) si la retirer change la performance (un arbre peut l'utiliser en interaction).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # -> src/

import pandas as pd
from sklearn.pipeline import Pipeline

import config as cfg
from modelisation import features as F
from modelisation import evaluation as E
from modelisation.hist_gradient_boosting import build_model


def _run(gold, num_cols, tag):
    train, test = F.temporal_split(gold)
    cat, _ = F.get_feature_lists()
    pipe = Pipeline([("pre", F.make_preprocessor(cat, num_cols, scale=False)),
                     ("clf", build_model())])
    pipe.fit(train[cat + num_cols], train["fraude"])
    return E.metrics_row(tag, train["fraude"], pipe.predict_proba(train[cat + num_cols])[:, 1],
                         test["fraude"], pipe.predict_proba(test[cat + num_cols])[:, 1])


def main():
    print(">>> Ablation FM_Difference_Pays (sur HistGB)")
    gold = pd.read_parquet(cfg.GOLD_PARQUET)
    _, num_avec = F.get_feature_lists(keep_diff_pays=True)    # avec la famille
    _, num_sans = F.get_feature_lists(keep_diff_pays=False)   # sans (defaut du projet)

    rows = [_run(gold, num_avec, "avec Difference_Pays"),
            _run(gold, num_sans, "sans Difference_Pays")]
    res = pd.DataFrame(rows)[["modele", "ROC_AUC_test", "PR_AUC_test", "lift@10%_test"]]
    res.to_csv(cfg.METRICS_DIR / "ablation_difference_pays.csv", index=False)
    print(res.round(4).to_string(index=False))
    d = rows[1]["PR_AUC_test"] - rows[0]["PR_AUC_test"]
    if d > 0.003:
        verdict = "RETIRER (la version sans est meilleure)"
    elif d < -0.003:
        verdict = "GARDER (la retirer degrade)"
    else:
        verdict = "neutre"
    print(f"\nDelta PR-AUC (sans - avec) = {d:+.4f}  ->  {verdict}")


if __name__ == "__main__":
    main()
