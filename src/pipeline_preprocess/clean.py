"""
COUCHE CLEAN - typage correct, horodatage reconstruit, validation.
>>> Aucune feature derivee de la cible ici (discipline anti-fuite). <<<
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # -> src/

import pandas as pd
import config as cfg


def build_clean() -> pd.DataFrame:
    print(">>> CLEAN : typage + horodatage...")
    df = pd.read_parquet(cfg.RAW_PARQUET)

    # 1. Reconstruire un VRAI horodatage : 'dateheure' a perdu l'heure (== Date).
    #    On recombine Date (jour) + Heure (HH:MM:SS).
    jour  = pd.to_datetime(df["Date"]).dt.normalize()
    heure = pd.to_timedelta(df["Heure"])
    df["datetime"] = jour + heure

    # 2. Typage : identifiants / qualitatives en categorie, montant en float.
    df["Carte"]   = df["Carte"].astype("int64")     # identifiant, PAS une quantite
    df["Montant"] = df["Montant"].astype("float64")
    df["fraude"]  = df["fraude"].astype("int8")
    for c in ["Pays", "CodeRep", "MCC"]:
        df[c] = df[c].astype(str).str.strip().astype("category")

    # 3. On retire les colonnes remplacees (dateheure cassee, Date/Heure -> datetime).
    df = df.drop(columns=["dateheure", "Date", "Heure"])

    # 4. Validation legere : on garde les transactions reelles, tri temporel obligatoire.
    df = df.drop_duplicates()
    df = df.sort_values("datetime").reset_index(drop=True)

    df.to_parquet(cfg.CLEAN_PARQUET)
    print(f"   Parquet: {cfg.CLEAN_PARQUET.name}  ({df.shape[0]:,} lignes x {df.shape[1]} colonnes)")
    cfg.excel_sample(df, cfg.CLEAN_DIR / "clean_sample.xlsx")
    return df


if __name__ == "__main__":
    build_clean()
