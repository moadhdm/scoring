"""
Genere les figures d'EDA (fonctions de viz d'Amine) dans results/figures/eda/.
Les variables continues sont tracees sur un echantillon (KDE couteux sur 1,15M lignes).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # -> src/

import pandas as pd
import config as cfg
from exploration import viz


def main():
    print(">>> Generation des figures EDA")
    gold = pd.read_parquet(cfg.GOLD_PARQUET)
    sample = gold.sample(60_000, random_state=0)        # pour les KDE (rapidite)

    viz.distrib_cat_by_target("CodeRep", gold)
    for v in ["montant_log", "FM_Velocity_Condition_24", "FM_Sum_24"]:
        viz.distrib_cont_by_target(v, sample)
    print(f"   Figures -> {cfg.FIG_DIR / 'eda'}")


if __name__ == "__main__":
    main()
