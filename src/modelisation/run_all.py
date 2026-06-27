"""
Lance les 3 modeles, compile le tableau comparatif et un graphique de synthese.
Resultats ecrits dans data/results/.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # -> src/

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from modelisation import (logistic_regression, logistic_scorecard,
                          random_forest, hist_gradient_boosting)

MODELES = [logistic_regression, logistic_scorecard, random_forest, hist_gradient_boosting]


def main():
    print(">>> Lecture de la table gold...")
    gold = pd.read_parquet(cfg.GOLD_PARQUET)

    rows = []
    for mod in MODELES:
        print(f">>> Entrainement : {mod.NAME}")
        rows.append(mod.train_and_eval(gold))

    recap = pd.DataFrame(rows)
    recap.to_csv(cfg.METRICS_DIR / "recap_modeles.csv", index=False)
    print("\n=== RECAPITULATIF (test temporel) ===")
    print(recap.round(4).to_string(index=False))

    # Graphique de synthese : PR-AUC test par modele (metrique reine)
    plt.figure(figsize=(6, 4))
    plt.bar(recap["modele"], recap["PR_AUC_test"], color="#B85042")
    plt.ylabel("PR-AUC (test)"); plt.title("Comparaison des modeles - PR-AUC")
    plt.xticks(rotation=15, ha="right"); plt.tight_layout()
    plt.savefig(cfg.FIG_DIR / "comparaison_pr_auc.png", dpi=120); plt.close()
    print(f"\nFigures et metriques -> {cfg.RESULTS}")


if __name__ == "__main__":
    main()
