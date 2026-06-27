"""
Information Value (IV) - screening univarie du pouvoir predictif.
Technique standard du scoring bancaire (contribution d'Amine), ici corrigee pour
etre calculee SUR LE TRAIN UNIQUEMENT (sinon les valeurs fuiteraient le test).

Grille de lecture (convention credit scoring) :
    IV < 0.02   inutile        0.1 - 0.3   moyen         > 0.5   tres fort (parfois suspect)
    0.02 - 0.1  faible         0.3 - 0.5   fort

A noter : l'IV revele des signaux que la correlation lineaire rate. Ici les
familles Velocity et Sum ressortent tres fortes (lien non-lineaire / a seuil)
alors que leur correlation de Pearson est faible.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # -> src/

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from modelisation.features import temporal_split

EPS = 1e-4


def information_value(feature: pd.Series, y: pd.Series, bins: int = 10) -> float:
    """IV d'une variable (binning par quantiles si continue a forte cardinalite)."""
    s = feature
    if s.dtype.kind in "if" and s.nunique() > 10:
        b = pd.qcut(s, q=bins, duplicates="drop")
    else:
        b = s.astype(str)
    d = pd.DataFrame({"bin": b, "y": y})
    g = d.groupby("bin", observed=True)["y"].agg(["sum", "count"])
    ev  = (g["sum"] / g["sum"].sum()).replace(0, EPS)                 # % d'evenements (fraude)
    nev = ((g["count"] - g["sum"]) / (g["count"] - g["sum"]).sum()).replace(0, EPS)
    return float(((ev - nev) * np.log(ev / nev)).sum())


def iv_ranking(train: pd.DataFrame, features: list, target: str = "fraude") -> pd.DataFrame:
    rows = [{"variable": f, "IV": information_value(train[f], train[target])} for f in features]
    return pd.DataFrame(rows).sort_values("IV", ascending=False).reset_index(drop=True)


def main():
    print(">>> Information Value (train-only)")
    gold = pd.read_parquet(cfg.GOLD_PARQUET)
    train, _ = temporal_split(gold)            # IV calculee sur le TRAIN seul

    features = ["MCC", "Pays", "CodeRep", "montant_log",
                *cfg.FM_FAMILIES["vel"], *cfg.FM_FAMILIES["sum"],
                *cfg.FM_FAMILIES["redond"], *cfg.FM_FAMILIES["diff"],
                "vel_burst_3_24", "sum_conc_3_24"]
    rank = iv_ranking(train, features)
    rank.to_csv(cfg.METRICS_DIR / "iv_ranking.csv", index=False)
    print(rank.round(3).to_string(index=False))

    plt.figure(figsize=(7, 8))
    plt.barh(rank["variable"][::-1], rank["IV"][::-1], color="#028090")
    for x in (0.02, 0.1, 0.3, 0.5):
        plt.axvline(x, ls="--", color="grey", lw=0.7)
    plt.xlabel("Information Value"); plt.title("Pouvoir predictif univarie (IV, train)")
    plt.tight_layout()
    plt.savefig(cfg.FIG_DIR / "iv_ranking.png", dpi=120); plt.close()
    print(f"\n-> {cfg.METRICS_DIR / 'iv_ranking.csv'}")


if __name__ == "__main__":
    main()
