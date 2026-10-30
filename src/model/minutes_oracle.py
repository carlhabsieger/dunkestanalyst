"""Diagnostic : quelle part de l'erreur vient des minutes ?"""
import pandas as pd
from src.model.baseline import load_all, MIN_PRIOR_GAMES, MIN_PRIOR_MINUTES

df = load_all()
g = df.groupby("PLAYER_ID")
df["n_prior"] = g.cumcount()
df["prior_min"] = g["MIN"].transform(
    lambda s: s.shift(1).rolling(10, min_periods=1).mean()
)
# cumuls AVANT le match courant (pas de fuite)
cum_d = g["DUNKEST"].cumsum() - df["DUNKEST"]
cum_m = g["MIN"].cumsum() - df["MIN"]
df["rate"] = cum_d / cum_m                      # points Dunkest par minute
df["pred_moy"] = cum_d / df["n_prior"]          # baseline (moyenne cumulée)
df["pred_taux_x_min_naive"] = df["rate"] * df["prior_min"]
df["pred_taux_x_min_reelles"] = df["rate"] * df["MIN"]   # ORACLE (plafond)

m = (df["n_prior"] >= MIN_PRIOR_GAMES) & (df["prior_min"] >= MIN_PRIOR_MINUTES)
y = df.loc[m, "DUNKEST"]
rows = []
for c in ["pred_moy", "pred_taux_x_min_naive", "pred_taux_x_min_reelles"]:
    e = (df.loc[m, c] - y).abs()
    rows.append({"modele": c, "MAE": round(e.mean(), 2)})
print(pd.DataFrame(rows).to_string(index=False))

# --- Analyse par écart de minutes (minutes réelles - minutes habituelles) ---
sub = df[m].copy()
sub["ecart_min"] = sub["MIN"] - sub["prior_min"]
sub["err_base"] = (sub["pred_moy"] - sub["DUNKEST"]).abs()
sub["err_oracle"] = (sub["pred_taux_x_min_reelles"] - sub["DUNKEST"]).abs()
sub["biais_base"] = sub["pred_moy"] - sub["DUNKEST"]
sub["tranche"] = pd.cut(
    sub["ecart_min"], bins=[-100, -12, -6, -3, 3, 6, 12, 100]
)
res = sub.groupby("tranche", observed=True).agg(
    n=("DUNKEST", "size"),
    MAE_baseline=("err_base", "mean"),
    MAE_oracle=("err_oracle", "mean"),
    biais_baseline=("biais_base", "mean"),
).round(2)
res["part_%"] = (100 * res["n"] / res["n"].sum()).round(1)
print()
print(res.to_string())
