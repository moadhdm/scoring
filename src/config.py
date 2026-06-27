"""
Configuration partagee du projet (chemins, date de coupure, constantes).
Un seul endroit a modifier -> tous les scripts s'y referent.
"""
from __future__ import annotations
import pathlib
import pandas as pd

# --- Racine du projet (parent du dossier src/) ---------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]

# --- Dossiers ------------------------------------------------------------------
DATA        = ROOT / "data"
RAW_DIR     = DATA / "raw"
CLEAN_DIR   = DATA / "clean"
GOLD_DIR    = DATA / "gold"
RESULTS     = DATA / "results"
FIG_DIR     = RESULTS / "figures"
METRICS_DIR = RESULTS / "metrics"

# --- Fichiers ------------------------------------------------------------------
SAS_PATH      = RAW_DIR / "autorisations.sas7bdat"   # source SAS (a placer ici)
RAW_PARQUET   = RAW_DIR / "raw.parquet"
CLEAN_PARQUET = CLEAN_DIR / "clean.parquet"
GOLD_PARQUET  = GOLD_DIR / "gold.parquet"

# --- Parametres metier / modelisation ------------------------------------------
CUTOFF = pd.Timestamp("2004-05-01")      # test = transactions >= cette date (postérieures)
SAMPLE_SIZE = 100_000                     # taille des echantillons Excel (Excel max ~1,05M lignes)
LOST_STOLEN_CODES = {"41", "43"}          # codes ISO "carte perdue / volee" (quasi-cible)
EPS = 1e-9

# Les 4 familles de variables glissantes, sur 4 fenetres (3, 6, 12, 24 heures)
FM_FAMILIES = {
    "vel":    [f"FM_Velocity_Condition_{w}" for w in (3, 6, 12, 24)],
    "sum":    [f"FM_Sum_{w}"               for w in (3, 6, 12, 24)],
    "redond": [f"FM_Redondance_MCC_{w}"    for w in (3, 6, 12, 24)],
    "diff":   [f"FM_Difference_Pays_{w}"   for w in (3, 6, 12, 24)],
}


def excel_sample(df: pd.DataFrame, path: pathlib.Path, n: int = SAMPLE_SIZE) -> None:
    """Ecrit un echantillon Excel <= n lignes.
    Si la colonne 'fraude' existe : on garde TOUTES les fraudes + un tirage
    aleatoire de non-fraudes (echantillon stratifie, representatif et leger)."""
    if "fraude" in df.columns and len(df) > n:
        pos = df[df["fraude"] == 1]
        neg = df[df["fraude"] == 0].sample(max(n - len(pos), 0), random_state=0)
        out = pd.concat([pos, neg]).sort_index()
    else:
        out = df.head(n)
    out.to_excel(path, index=False)
    print(f"   Excel  : {path.name}  ({len(out):,} lignes)")
