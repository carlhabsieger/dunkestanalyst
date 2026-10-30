"""Modèle de minutes (LightGBM) puis score Dunkest = taux par minute x minutes."""
import pandas as pd
import lightgbm as lgb
from src.model.baseline import load_all, MIN_PRIOR_GAMES, MIN_PRIOR_MINUTES

df = load_all()
g = df.groupby("PLAYER_ID")
df["n_prior"] = g.cumcount()
df["prior_min"] = g["MIN"].transform(
    lambda s: s.shift(1).rolling(10, min_periods=1).mean())
df["prior_min3"] = g["MIN"].transform(
    lambda s: s.shift(1).rolling(3, min_periods=1).mean())
df["days_rest"] = g["GAME_DATE"].diff().dt.days.clip(upper=7)
df["home"] = df["MATCHUP"].str.contains("vs.").astype(int)
cum_d = g["DUNKEST"].cumsum() - df["DUNKEST"]
cum_m = g["MIN"].cumsum() - df["MIN"]
df["rate"] = cum_d / cum_m
df["pred_moy"] = cum_d / df["n_prior"]

rows = []
for team, tdf in df.groupby("TEAM_ID"):
    games = list(tdf.sort_values("GAME_DATE")["GAME_ID"].unique())
    by_game = {gid: grp.set_index("PLAYER_ID")["MIN"]
               for gid, grp in tdf.groupby("GAME_ID")}
    for i, gid in enumerate(games):
        if i < 10:
            continue
        prev = pd.concat([by_game[x] for x in games[i - 10:i]])
        habit = prev.groupby(level=0).mean()
        habit = habit[habit >= 15]
        now = by_game[gid].index
        last = by_game[games[i - 1]].index
        absents = habit.index.difference(now)
        new_abs = absents.intersection(last)
        old_abs = absents.difference(last)
        rows.append({"TEAM_ID": team, "GAME_ID": gid,
                     "vac_new": habit[new_abs].sum(),
                     "vac_old": habit[old_abs].sum(),
                     "n_new_abs": len(new_abs)})
df = df.merge(pd.DataFrame(rows), on=["TEAM_ID", "GAME_ID"], how="inner")

feats = ["prior_min", "prior_min3", "vac_new", "vac_old",
         "n_new_abs", "days_rest", "home"]
ok = df[feats + ["MIN", "rate"]].notna().all(axis=1)
m = (df["n_prior"] >= MIN_PRIOR_GAMES) & (df["prior_min"] >= MIN_PRIOR_MINUTES) & ok
d = df[m].sort_values("GAME_DATE").reset_index(drop=True)
cut = int(len(d) * 0.7)
tr, te = d.iloc[:cut], d.iloc[cut:].copy()
print(f"train {len(tr)} | test {len(te)} (à partir du {te['GAME_DATE'].min().date()})")

model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                          min_child_samples=50, subsample=0.8,
                          subsample_freq=1, verbose=-1)
model.fit(tr[feats], tr["MIN"])
te["pred_min"] = model.predict(te[feats])

print("\nMAE minutes : baseline (prior_min) =",
      round((te["prior_min"] - te["MIN"]).abs().mean(), 2),
      "| modèle =", round((te["pred_min"] - te["MIN"]).abs().mean(), 2))

te["p_modele"] = te["rate"] * te["pred_min"]
te["p_naive"] = te["rate"] * te["prior_min"]
te["p_oracle"] = te["rate"] * te["MIN"]
print("\nMAE Dunkest (test) :")
for c in ["pred_moy", "p_naive", "p_modele", "p_oracle"]:
    print(f"  {c:10s} {(te[c] - te['DUNKEST']).abs().mean():.2f}")

print("\nImportance des variables :")
print(pd.Series(model.feature_importances_, index=feats)
      .sort_values(ascending=False).to_string())

# --- Évaluation sur les cas où le modèle prévoit un changement de minutes ---
te["delta_pred"] = te["pred_min"] - te["prior_min"]
te["delta_reel"] = te["MIN"] - te["prior_min"]
te["err_naive"] = (te["p_naive"] - te["DUNKEST"]).abs()
te["err_modele"] = (te["p_modele"] - te["DUNKEST"]).abs()
te["biais_naive"] = te["p_naive"] - te["DUNKEST"]
te["biais_modele"] = te["p_modele"] - te["DUNKEST"]
te["tranche"] = pd.cut(te["delta_pred"], bins=[-100, -3, -1, 1, 3, 100])
res = te.groupby("tranche", observed=True).agg(
    n=("DUNKEST", "size"),
    delta_pred=("delta_pred", "mean"),
    delta_reel=("delta_reel", "mean"),
    err_naive=("err_naive", "mean"),
    err_modele=("err_modele", "mean"),
    biais_naive=("biais_naive", "mean"),
    biais_modele=("biais_modele", "mean"),
).round(2)
print()
print(res.to_string())
