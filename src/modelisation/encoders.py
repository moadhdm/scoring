"""
WoEEncoder - encodage Weight of Evidence, AJUSTE SUR LE TRAIN UNIQUEMENT.

WoE(bin) = ln( (% d'evenements dans le bin) / (% de non-evenements dans le bin) ).
Pour les variables continues : binning par quantiles appris sur le train.
Pour les categorielles : un WoE par modalite (avec lissage).
Les modalites/valeurs inconnues au test recoivent le log-odds global (neutre).

C'est la brique du scorecard bancaire : variables transformees en WoE puis
regression logistique -> coefficients interpretables et grille de points.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class WoEEncoder:
    def __init__(self, cat_cols, cont_cols, bins: int = 10, smooth: float = 0.5):
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.bins = bins
        self.smooth = smooth

    def _woe_map(self, binned: pd.Series, y: pd.Series):
        """Retourne {modalite: woe} + la valeur par defaut (log-odds global)."""
        d = pd.DataFrame({"b": binned, "y": y})
        g = d.groupby("b", observed=True)["y"].agg(["sum", "count"])
        ev = g["sum"]; nev = g["count"] - g["sum"]
        E, N = ev.sum(), nev.sum()
        woe = np.log(((ev + self.smooth) / E) / ((nev + self.smooth) / N))
        return woe.to_dict(), float(np.log(E / N))

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.maps_, self.edges_, self.default_ = {}, {}, {}
        for c in self.cat_cols:
            self.maps_[c], self.default_[c] = self._woe_map(X[c].astype(str), y)
        for c in self.cont_cols:
            b, edges = pd.qcut(X[c], self.bins, duplicates="drop", retbins=True, labels=False)
            self.edges_[c] = edges
            self.maps_[c], self.default_[c] = self._woe_map(pd.Series(b, index=X.index), y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for c in self.cat_cols:
            out[c] = X[c].astype(str).map(self.maps_[c]).fillna(self.default_[c])
        for c in self.cont_cols:
            b = pd.cut(X[c], bins=self.edges_[c], labels=False, include_lowest=True)
            out[c] = pd.Series(b, index=X.index).map(self.maps_[c]).fillna(self.default_[c])
        return out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
