"""
Modele : SCORECARD (regression logistique sur variables encodees en WoE).
=> Methode standard du scoring bancaire (inspiree du travail d'Amine).
Ameliore notre logistique simple (PR-AUC ~0.12 vs ~0.11) tout en restant
parfaitement interpretable, et produit une GRILLE DE POINTS.

Convention de score : points eleves = risque de fraude eleve.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # -> src/

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import config as cfg
from modelisation import features as F
from modelisation import evaluation as E
from modelisation.encoders import WoEEncoder

NAME = "Scorecard WoE"

# Variables encodees en WoE (categorielles + continues a binner) ; flags gardes en 0/1.
CAT_WOE  = ["MCC", "Pays", "CodeRep"]
CONT_WOE = ["montant_log", "FM_Velocity_Condition_3", "FM_Velocity_Condition_24",
            "FM_Sum_24", "FM_Redondance_MCC_24",
            "vel_burst_3_24", "sum_conc_3_24"]
FLAGS    = ["is_night", "is_weekend", "montant_round", "vel24_gt10",
            "code_refus", "code_lost_stolen"]

PDO = 20.0   # "points to double the odds" : echelle de la grille de score


def _build_design(enc: WoEEncoder, X: pd.DataFrame) -> pd.DataFrame:
    """Matrice = WoE(cat + cont) + flags bruts."""
    woe = enc.transform(X)
    return pd.concat([woe, X[FLAGS]], axis=1)


def build_scorecard(enc: WoEEncoder, lr: LogisticRegression, feat_order: list) -> pd.DataFrame:
    """Convertit (WoE x coefficients) en points. points = factor * coef * WoE."""
    factor = PDO / np.log(2)
    coef = dict(zip(feat_order, lr.coef_[0]))
    rows = []
    # variables continues : un point par tranche (avec bornes lisibles)
    for c in CONT_WOE:
        edges = enc.edges_[c]
        for b, woe in sorted(enc.maps_[c].items()):
            lo, hi = edges[int(b)], edges[int(b) + 1]
            rows.append({"variable": c, "modalite": f"[{lo:.2f}, {hi:.2f}]",
                         "woe": round(woe, 3), "points": round(factor * coef[c] * woe)})
    # variables categorielles : un point par modalite
    for c in CAT_WOE:
        for mod, woe in enc.maps_[c].items():
            rows.append({"variable": c, "modalite": str(mod),
                         "woe": round(woe, 3), "points": round(factor * coef[c] * woe)})
    # flags : points si flag = 1
    for c in FLAGS:
        rows.append({"variable": c, "modalite": "=1", "woe": np.nan,
                     "points": round(factor * coef[c] * 1)})
    return pd.DataFrame(rows)


def train_and_eval(gold: pd.DataFrame) -> dict:
    train, test = F.temporal_split(gold)
    ytr, yte = train["fraude"], test["fraude"]

    enc = WoEEncoder(CAT_WOE, CONT_WOE).fit(train, ytr)      # WoE ajuste sur le TRAIN seul
    Xtr, Xte = _build_design(enc, train), _build_design(enc, test)

    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    p_tr = lr.predict_proba(Xtr)[:, 1]
    p_te = lr.predict_proba(Xte)[:, 1]

    row = E.metrics_row(NAME, ytr, p_tr, yte, p_te)
    slug = NAME.lower().replace(" ", "_")
    E.threshold_table(yte, p_te).to_csv(cfg.METRICS_DIR / f"table_seuil_{slug}.csv", index=False)
    E.save_curves(NAME, yte, p_te)

    # Grille de points (livrable interpretable)
    card = build_scorecard(enc, lr, list(Xtr.columns))
    card.to_csv(cfg.METRICS_DIR / "scorecard_points.csv", index=False)

    print(f"   {NAME}: ROC_test={row['ROC_AUC_test']:.4f}  "
          f"PR_test={row['PR_AUC_test']:.4f}  lift@10%={row['lift@10%_test']:.1f}")
    return row


if __name__ == "__main__":
    print(f">>> Modele : {NAME}")
    gold = pd.read_parquet(cfg.GOLD_PARQUET)
    row = train_and_eval(gold)
    pd.DataFrame([row]).to_csv(cfg.METRICS_DIR / "metrics_scorecard.csv", index=False)
    print("   Grille de points -> scorecard_points.csv")
