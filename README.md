# Scoring de fraude bancaire — M2 TIDE

Détection de transactions frauduleuses par carte. Architecture **médaillon**
(raw → clean → gold), **screening IV**, puis comparaison de modèles de scoring.

Une ligne = une transaction réalisée par une carte. **1 151 432 transactions**,
période nov. 2003 → juin 2004. Cible `fraude` (0/1), **très déséquilibrée (~0,63 %)**.

> Travail à deux mains : pipeline médaillon + modélisation anti-fuite d'un côté ;
> EDA et **Information Value / WoE** apportés par Amine, intégrés ici de façon
> *leakage-safe* (calculés sur le train uniquement). Voir `src/exploration/`.

---

## Arborescence

```
projet-scoring-fraude/
├── environment.yml
├── env/                              # environnement conda (à créer)
├── data/
│   ├── raw/ clean/ gold/             # parquet (complet) + échantillon Excel par couche
│   └── results/
│       ├── figures/  (+ figures/eda/) # ROC / PR / lift, IV, comparaison, EDA
│       └── metrics/                  # recap, tables seuil, iv_ranking, scorecard, ablation
└── src/
    ├── config.py                     # chemins + date de coupure + constantes
    ├── exploration/                  # CONTRIBUTION AMINE (anti-fuite)
    │   ├── viz.py                    # distributions par cible (cat & continu)
    │   ├── information_value.py      # IV train-only + ranking
    │   └── run_eda.py                # génère les figures EDA
    ├── pipeline_preprocess/          # raw.py, clean.py, gold.py
    └── modelisation/
        ├── features.py               # listes features, split temporel, préprocesseur
        ├── encoders.py               # WoEEncoder (train-only)
        ├── evaluation.py             # métriques, table seuil, courbes
        ├── logistic_regression.py    # logistique simple
        ├── logistic_scorecard.py     # SCORECARD WoE + grille de points
        ├── random_forest.py
        ├── hist_gradient_boosting.py # MODELE RETENU
        ├── ablation_difference_pays.py
        └── run_all.py                # 4 modèles + comparaison
```

> Les `.xlsx` sont des **échantillons** (Excel plafonne à ~1,05 M lignes ; on a 1,15 M).
> Le `.parquet` reste **complet** et fait foi.

---

## Installation & lancement

```bash
conda env create -f environment.yml -p ./env
conda activate ./env
```

Données pré-générées incluses. Pour tout régénérer depuis la source, placer
`autorisations.sas7bdat` dans `data/raw/`, puis :

```bash
python src/pipeline_preprocess/raw.py        # SAS  -> raw.parquet
python src/pipeline_preprocess/clean.py      # raw  -> clean.parquet
python src/pipeline_preprocess/gold.py       # clean-> gold.parquet
python src/exploration/information_value.py  # screening IV (train-only)
python src/exploration/run_eda.py            # figures EDA
python src/modelisation/run_all.py           # 4 modèles + récap
python src/modelisation/ablation_difference_pays.py   # test feature inutile
```

---

## Résultats (test temporel mai–juin 2004, postérieur au train)

| Modèle | ROC test | **PR-AUC test** | lift@1% | lift@10% |
|---|---|---|---|---|
| Logistique simple | 0,827 | 0,112 | 27 | 6,0 |
| **Scorecard WoE** (interprétable) | 0,829 | 0,120 | 26 | 6,2 |
| Random Forest | 0,836 | 0,181 | 31 | 6,5 |
| **HistGradientBoosting** (retenu) | 0,831 | **0,219** | **37** | 6,4 |

Métrique reine en fraude déséquilibrée : **PR-AUC** (+ lift / table seuil pour la
priorisation des dossiers). L'accuracy n'a pas de sens ici.

### Apports de l'intégration (mesurés)

1. **Screening IV (train-only)** révèle ce que la corrélation linéaire ratait :
   les familles **Sum** et **Velocity** sont très prédictives (IV 0,47–0,79) malgré
   une corrélation de Pearson faible (lien non-linéaire / à seuil). Voir `iv_ranking.csv`.
2. **`FM_Difference_Pays` est inutile (IV ≈ 0) ET nuisible** : l'ablation montre
   **+0,04 de PR-AUC sur HistGB** (0,178 → 0,219) en la retirant. Elle est donc
   **exclue par défaut** des features (colonnes conservées dans gold ;
   `get_feature_lists(keep_diff_pays=True)` pour la réintégrer).
3. **Scorecard WoE** : améliore le modèle interprétable (PR 0,112 → 0,120) et produit
   une **grille de points** (`scorecard_points.csv`, convention : points élevés =
   risque élevé). Détail notable : `code_refus` reçoit des points négatifs car, à
   comportement égal, la majorité des fraudes passent en *accepté*.

---

## Principes anti-fuite (réflexes fraude bancaire)

- **Split strictement temporel** : apprentissage sur le passé, test sur le futur.
- **Tout ce qui dérive de la cible** (TargetEncoder, WoE, IV) est appris **sur le
  train seul** ; rien n'est figé dans la table gold.
- **Codes 41/43 ("carte perdue/volée")** quasi-circulaires : testés en retrait
  (`get_feature_lists(drop_near_leak=True)`) pour mesurer la performance honnête.

---

## Dictionnaire de données (couche raw / clean)

| Colonne | Description |
|---|---|
| `Carte` | numéro de carte (identifiant, ~198k cartes) |
| `Pays` | code pays de la transaction (167 modalités) |
| `Date` / `Heure` | jour et heure → recombinés en `datetime` au clean |
| `CodeRep` | code réponse autorisation (`00` = accepté, sinon refus) |
| `MCC` | code commerçant (661 modalités) |
| `Montant` | montant de la transaction (€) |
| `fraude` | **cible** : 1 = frauduleuse, 0 = saine |
| `FM_Velocity_Condition_{3,6,12,24}` | nb transactions acceptées sur les X dernières h |
| `FM_Sum_{3,6,12,24}` | montant cumulé sur les X dernières h |
| `FM_Redondance_MCC_{3,6,12,24}` | nb transactions chez le même commerçant sur X h |
| `FM_Difference_Pays_{3,6,12,24}` | nb pays différents sur X h *(exclue du modèle : IV≈0)* |
