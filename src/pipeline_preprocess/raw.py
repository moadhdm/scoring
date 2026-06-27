"""
COUCHE RAW - copie fidele de la source, aucune transformation.
Lit le fichier SAS et le materialise en parquet (+ echantillon Excel).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # -> src/

import pyreadstat
import config as cfg


def build_raw() -> "pd.DataFrame":
    if not cfg.SAS_PATH.exists():
        raise FileNotFoundError(
            f"Fichier SAS introuvable : {cfg.SAS_PATH}\n"
            "Place 'autorisations.sas7bdat' dans data/raw/ puis relance."
        )
    print(">>> RAW : lecture du SAS...")
    df, _ = pyreadstat.read_sas7bdat(str(cfg.SAS_PATH))

    df.to_parquet(cfg.RAW_PARQUET)
    print(f"   Parquet: {cfg.RAW_PARQUET.name}  ({df.shape[0]:,} lignes x {df.shape[1]} colonnes)")
    cfg.excel_sample(df, cfg.RAW_DIR / "raw_sample.xlsx")
    return df


if __name__ == "__main__":
    build_raw()
