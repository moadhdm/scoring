"""
Briques communes aux 3 modeles : listes de features, split temporel, et le
preprocesseur d'encodage anti-fuite (TargetEncoder a cross-fitting + scaling).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # -> src/

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder, StandardScaler
import config as cfg


def get_feature_lists(drop_near_leak: bool = False, keep_diff_pays: bool = False):
    """Colonnes categorielles (-> TargetEncoder) et numeriques (passthrough).
    drop_near_leak=True : retire l'info 'carte deja signalee compromise' (codes 41/43
        + flags refus) pour mesurer la performance honnete.
    keep_diff_pays=False (defaut) : la famille FM_Difference_Pays est EXCLUE car
        inutile (IV ~ 0) ET nuisible -> l'ablation montre +0.04 de PR-AUC sur HistGB
        en la retirant (cf. ablation_difference_pays.py). Mettre True pour la reintegrer."""
    cat_cols = ["MCC", "Pays", "CodeRep"]
    num_cols = [
        "hour", "dow", "is_weekend", "is_night", "month",
        "montant_log", "montant_round",
        *cfg.FM_FAMILIES["vel"], *cfg.FM_FAMILIES["sum"], *cfg.FM_FAMILIES["redond"],
        "vel_burst_3_24", "sum_conc_3_24", "redond_ratio_3_24",
        "vel24_gt10", "montant_vs_sum24",
        "code_refus", "code_lost_stolen",
    ]
    if keep_diff_pays:
        num_cols += [*cfg.FM_FAMILIES["diff"], "diff_ratio_3_24"]
    if drop_near_leak:
        cat_cols = ["MCC", "Pays"]
        num_cols = [c for c in num_cols if c not in ("code_refus", "code_lost_stolen")]
    return cat_cols, num_cols


def reduced_num_cols(num_cols):
    """Variante decorrelee : retire les fenetres 6h/12h redondantes (garde 3h & 24h
    + ratios). Surtout utile pour la regression logistique (coefficients plus nets)."""
    drop = []
    for fam in cfg.FM_FAMILIES.values():
        drop += [fam[1], fam[2]]      # fenetres 6h et 12h
    return [c for c in num_cols if c not in drop]


def temporal_split(gold: pd.DataFrame):
    """Split STRICTEMENT temporel : on apprend sur le passe, on teste sur le futur."""
    train = gold[gold["datetime"] < cfg.CUTOFF].copy()
    test  = gold[gold["datetime"] >= cfg.CUTOFF].copy()
    return train, test


def make_preprocessor(cat_cols, num_cols, scale: bool) -> ColumnTransformer:
    """TargetEncoder : cross-fitting interne sur le train (anti-fuite), stats
    pleines appliquees au test. Scaling reserve aux modeles lineaires."""
    te = TargetEncoder(target_type="binary", smooth="auto", cv=5, random_state=0)
    num_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [("cat", te, cat_cols), ("num", num_step, num_cols)],
        remainder="drop",
    )
