"""
COUCHE GOLD - features "row-local" pretes a modeliser.
Chaque feature n'utilise QUE la ligne courante (+ variables glissantes deja
calculees) : la table est donc identique en train et en test, sans ajustement.
Les encodages derives de la cible NE sont PAS ici -> ils vivent dans le pipeline
modele (modelisation/features.py), appris sur le train seul.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # -> src/

import numpy as np
import pandas as pd
import config as cfg


def build_gold() -> pd.DataFrame:
    print(">>> GOLD : feature engineering...")
    df = pd.read_parquet(cfg.CLEAN_PARQUET)
    g = pd.DataFrame(index=df.index)

    # --- References (non utilisees comme features : split + audit) ---
    g["datetime"] = df["datetime"]
    g["Carte"]    = df["Carte"]
    g["fraude"]   = df["fraude"]

    # --- TEMPS (row-local) ---
    g["hour"]       = df["datetime"].dt.hour
    g["dow"]        = df["datetime"].dt.dayofweek
    g["is_weekend"] = (g["dow"] >= 5).astype("int8")
    g["is_night"]   = g["hour"].between(0, 5).astype("int8")     # 00h-05h : ~x1.9 de fraude
    g["month"]      = df["datetime"].dt.month

    # --- MONTANT (row-local) : log pour ecraser l'asymetrie (max ~546k EUR) ---
    g["montant"]       = df["Montant"]
    g["montant_log"]   = np.log1p(df["Montant"].clip(lower=0))
    g["montant_round"] = (df["Montant"] % 1 == 0).astype("int8")

    # --- VARIABLES GLISSANTES : 16 brutes (les arbres gerent la colinearite) ---
    for cols in cfg.FM_FAMILIES.values():
        for c in cols:
            g[c] = df[c]

    # --- RATIOS decorreles : part de l'activite 24h concentree sur les 3 dernieres h.
    #     Denominateur plancher a 1 -> ratio borne, pas d'explosion par division par ~0. ---
    floor1 = lambda s: s.clip(lower=1.0)
    g["vel_burst_3_24"]    = (df["FM_Velocity_Condition_3"] / floor1(df["FM_Velocity_Condition_24"])).clip(0, 1)
    g["sum_conc_3_24"]     = (df["FM_Sum_3"]                / floor1(df["FM_Sum_24"])).clip(0, 1)
    g["redond_ratio_3_24"] = (df["FM_Redondance_MCC_3"]    / floor1(df["FM_Redondance_MCC_24"])).clip(0, 1)
    g["diff_ratio_3_24"]   = (df["FM_Difference_Pays_3"]   / floor1(df["FM_Difference_Pays_24"])).clip(0, 1)
    g["montant_vs_sum24"]  = (df["Montant"]                / floor1(df["FM_Sum_24"])).clip(0, 1)

    # seuil fort : >10 transactions/24h -> ~14% de fraude (x22)
    g["vel24_gt10"] = (df["FM_Velocity_Condition_24"] > 10).astype("int8")

    # --- CodeRep : flags row-local + modalite brute pour encodage en pipeline ---
    code = df["CodeRep"].astype(str)
    g["code_refus"]       = (code != "00").astype("int8")            # refus : 2.2% vs 0.5% de fraude
    g["code_lost_stolen"] = code.isin(cfg.LOST_STOLEN_CODES).astype("int8")
    g["CodeRep"] = code

    # --- Categorielles brutes (encodage cible fait dans le pipeline modele) ---
    g["MCC"]  = df["MCC"].astype(str)
    g["Pays"] = df["Pays"].astype(str)

    g.to_parquet(cfg.GOLD_PARQUET)
    n_feat = len([c for c in g.columns if c not in ("datetime", "Carte", "fraude")])
    print(f"   Parquet: {cfg.GOLD_PARQUET.name}  ({g.shape[0]:,} lignes x {n_feat} features)")
    cfg.excel_sample(g, cfg.GOLD_DIR / "gold_sample.xlsx")
    return g


if __name__ == "__main__":
    build_gold()
