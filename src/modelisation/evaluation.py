"""
Evaluation commune : metriques adaptees a la fraude (ROC-AUC, PR-AUC, lift),
table seuil (logique 'prioriser les dossiers') et courbes sauvegardees en PNG.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # -> src/

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # backend fichier (pas d'affichage)
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)
import config as cfg


def lift_at_k(y_true, y_score, k: float) -> float:
    """Lift sur les k% dossiers les plus risques (vs tirage aleatoire)."""
    y_true = np.asarray(y_true)
    n = max(1, int(len(y_score) * k))
    idx = np.argsort(y_score)[::-1][:n]
    base = y_true.mean()
    return (y_true[idx].mean() / base) if base > 0 else np.nan


def metrics_row(name, y_tr, p_tr, y_te, p_te) -> dict:
    return {
        "modele": name,
        "ROC_AUC_train": roc_auc_score(y_tr, p_tr),
        "ROC_AUC_test":  roc_auc_score(y_te, p_te),
        "PR_AUC_test":   average_precision_score(y_te, p_te),
        "lift@1%_test":  lift_at_k(y_te, p_te, 0.01),
        "lift@5%_test":  lift_at_k(y_te, p_te, 0.05),
        "lift@10%_test": lift_at_k(y_te, p_te, 0.10),
        "lift@20%_test": lift_at_k(y_te, p_te, 0.20),
    }


def threshold_table(y_true, y_score,
                    ks=(0.005, 0.01, 0.02, 0.05, 0.10, 0.20)) -> pd.DataFrame:
    """Table seuil : pour chaque % de dossiers cibles, precision / rappel / lift."""
    y_true = np.asarray(y_true)
    base = y_true.mean()
    order = np.argsort(y_score)[::-1]
    rows = []
    for k in ks:
        n = max(1, int(len(y_score) * k))
        sel = order[:n]
        capt = int(y_true[sel].sum())
        prec = capt / n
        rows.append({
            "pct_cible": f"{k*100:.1f}%",
            "nb_dossiers": n,
            "fraudes_captees": capt,
            "precision": round(prec, 4),
            "rappel": round(capt / y_true.sum(), 4),
            "lift": round(prec / base, 1) if base > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def save_curves(name, y_te, p_te) -> None:
    """Sauvegarde 3 courbes (ROC, precision-rappel, lift) dans results/figures/."""
    slug = name.lower().replace(" ", "_")

    # ROC
    fpr, tpr, _ = roc_curve(y_te, p_te)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_te, p_te):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("1 - specificite"); plt.ylabel("rappel"); plt.title(f"ROC - {name}")
    plt.legend(); plt.tight_layout()
    plt.savefig(cfg.FIG_DIR / f"roc_{slug}.png", dpi=120); plt.close()

    # Precision-Rappel
    prec, rec, _ = precision_recall_curve(y_te, p_te)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, label=f"PR-AUC = {average_precision_score(y_te, p_te):.3f}")
    plt.axhline(np.mean(y_te), ls="--", color="grey", label="hasard")
    plt.xlabel("rappel"); plt.ylabel("precision"); plt.title(f"Precision-Rappel - {name}")
    plt.legend(); plt.tight_layout()
    plt.savefig(cfg.FIG_DIR / f"pr_{slug}.png", dpi=120); plt.close()

    # Lift cumule
    order = np.argsort(p_te)[::-1]
    y_sorted = np.asarray(y_te)[order]
    ks = np.linspace(0.01, 1.0, 100)
    base = np.mean(y_te)
    lifts = [y_sorted[:max(1, int(len(y_sorted) * k))].mean() / base for k in ks]
    plt.figure(figsize=(5, 4))
    plt.plot(ks * 100, lifts)
    plt.axhline(1, ls="--", color="grey")
    plt.xlabel("% dossiers cibles (tries par score)"); plt.ylabel("lift")
    plt.title(f"Courbe LIFT - {name}"); plt.tight_layout()
    plt.savefig(cfg.FIG_DIR / f"lift_{slug}.png", dpi=120); plt.close()


def full_report(name, pipe, Xtr, ytr, Xte, yte) -> dict:
    """Calcule probas, metriques, table seuil et courbes pour un modele."""
    p_tr = pipe.predict_proba(Xtr)[:, 1]
    p_te = pipe.predict_proba(Xte)[:, 1]
    row = metrics_row(name, ytr, p_tr, yte, p_te)
    slug = name.lower().replace(" ", "_")
    threshold_table(yte, p_te).to_csv(cfg.METRICS_DIR / f"table_seuil_{slug}.csv", index=False)
    save_curves(name, yte, p_te)
    print(f"   {name}: ROC_test={row['ROC_AUC_test']:.4f}  "
          f"PR_test={row['PR_AUC_test']:.4f}  lift@10%={row['lift@10%_test']:.1f}")
    return row
