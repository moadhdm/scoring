"""
Fonctions de visualisation pour l'EDA.
Contribution d'Amine (legerement adaptees) : distribution d'une variable par
rapport a la cible, en categoriel et en continu.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # -> src/

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import config as cfg


def distrib_cat_by_target(var_cat: str, df: pd.DataFrame, target: str = "fraude",
                          save: bool = True):
    """Distribution normalisee d'une variable categorielle par niveau de cible,
    modalites ordonnees par frequence globale, annotee en % (effectif)."""
    cat_order = df[var_cat].value_counts().index.tolist()
    levels = sorted(df[target].unique())

    abs_t = (pd.crosstab(df[target], df[var_cat])
             .reindex(index=levels, columns=cat_order, fill_value=0)
             .reset_index().melt(id_vars=target, var_name=var_cat, value_name="Absolute"))
    norm_t = (pd.crosstab(df[target], df[var_cat], normalize="index")
              .reindex(index=levels, columns=cat_order, fill_value=0)
              .reset_index().melt(id_vars=target, var_name=var_cat, value_name="Frequency"))
    merged = norm_t.merge(abs_t, on=[target, var_cat])
    merged[var_cat] = pd.Categorical(merged[var_cat], categories=cat_order, ordered=True)
    merged = merged.sort_values([target, var_cat])

    g = sns.catplot(x=target, y="Frequency", hue=var_cat, data=merged, kind="bar",
                    height=6, aspect=1.8, hue_order=cat_order, legend=False)
    ax = g.ax
    for hue_idx, container in enumerate(ax.containers):
        cat = cat_order[hue_idx]
        for patch_idx, patch in enumerate(container):
            row = merged[(merged[target] == levels[patch_idx]) & (merged[var_cat] == cat)]
            if not row.empty:
                ax.annotate(f"{row['Frequency'].values[0]*100:.1f}% ({int(row['Absolute'].values[0])})",
                            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                            ha="center", va="bottom", fontsize=9, xytext=(0, 3),
                            textcoords="offset points")
    plt.title(f"Distribution de '{var_cat}' par cible")
    plt.tight_layout()
    if save:
        plt.savefig(cfg.FIG_DIR / "eda" / f"distrib_cat_{var_cat}.png", dpi=120)
    plt.close()


def distrib_cont_by_target(var_cont: str, df: pd.DataFrame, target: str = "fraude",
                           bins: int = 30, save: bool = True):
    """Histogramme densite + KDE d'une variable continue, un panneau par niveau
    de cible, annote moyenne / mediane."""
    levels = sorted(df[target].unique())
    _, axes = plt.subplots(1, len(levels), figsize=(6 * len(levels), 5), sharey=True)
    if len(levels) == 1:
        axes = [axes]
    for ax, lvl in zip(axes, levels):
        sub = df[df[target] == lvl]
        sns.histplot(data=sub, x=var_cont, bins=bins, stat="density", kde=True, ax=ax)
        ax.text(0.05, 0.95, f"Moy: {sub[var_cont].mean():.2f}\nMed: {sub[var_cont].median():.2f}",
                transform=ax.transAxes, va="top", fontsize=11)
        ax.set_title(f"{target} = {lvl}")
    plt.tight_layout()
    if save:
        plt.savefig(cfg.FIG_DIR / "eda" / f"distrib_cont_{var_cont}.png", dpi=120)
    plt.close()
