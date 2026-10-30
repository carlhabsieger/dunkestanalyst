"""Absences récentes vs anciennes, selon le rôle du joueur qui en profite."""
import pandas as pd
from src.model.baseline import load_all, MIN_PRIOR_GAMES, MIN_PRIOR_MINUTES

df = load_all()
g = df.groupby("PLAYER_ID")
df["n_prior"] = g.cumcount()
df["prior_min"] = g["MIN"].transform(
    lambda s: s.shift(1).rolling(10, min_periods=1).mean()
)
df["pred_moy"] = (g["DUNKEST"].cumsum() - df["DUNKEST"]) / df["n_prior"]

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
        new_abs = absents.intersection(last)   # jouait au match précédent
        old_abs = absents.difference(last)     # déjà absent au match précédent
        rows.append({"TEAM_ID": team, "GAME_ID": gid,
                     "vac_new": habit[new_abs].sum(),
                     "vac_old": habit[old_abs].sum()})
df = df.merge(pd.DataFrame(rows), on=["TEAM_ID", "GAME_ID"], how="inner")

m = (df["n_prior"] >= MIN_PRIOR_GAMES) & (df["prior_min"] >= MIN_PRIOR_MINUTES)
sub = df[m].copy()
sub["ecart_min"] = sub["MIN"] - sub["prior_min"]
sub["gain_vs_base"] = sub["DUNKEST"] - sub["pred_moy"]
sub["role"] = pd.cut(sub["prior_min"], bins=[0, 22, 30, 60],
                     labels=["15-22 min", "22-30 min", "30+ min"])
sub["vac_new_tr"] = pd.cut(sub["vac_new"], bins=[-1, 0.5, 15, 30, 500],
                           labels=["0", "0-15", "15-30", "30+"])

print("Part des matchs avec une absence récente :",
      round(100 * (sub["vac_new"] > 0.5).mean(), 1), "%\n")
res = sub.groupby(["role", "vac_new_tr"], observed=True).agg(
    n=("DUNKEST", "size"),
    ecart_min=("ecart_min", "mean"),
    gain_vs_base=("gain_vs_base", "mean"),
).round(2)
print(res.to_string())
print("\nCorrélation vac_new vs ecart_min :",
      round(sub["vac_new"].corr(sub["ecart_min"]), 3))
print("Corrélation vac_old vs ecart_min :",
      round(sub["vac_old"].corr(sub["ecart_min"]), 3))
