"""Les minutes libérées par les absents expliquent-elles les hausses de minutes ?"""
import pandas as pd
from src.model.baseline import load_all, MIN_PRIOR_GAMES, MIN_PRIOR_MINUTES

df = load_all()
g = df.groupby("PLAYER_ID")
df["n_prior"] = g.cumcount()
df["prior_min"] = g["MIN"].transform(
    lambda s: s.shift(1).rolling(10, min_periods=1).mean()
)
df["pred_moy"] = (g["DUNKEST"].cumsum() - df["DUNKEST"]) / df["n_prior"]

# Minutes libérées par les absents, calculées avec les 10 matchs précédents de l'équipe
rows = []
for team, tdf in df.groupby("TEAM_ID"):
    games = list(tdf.sort_values("GAME_DATE")["GAME_ID"].unique())
    by_game = {gid: grp.set_index("PLAYER_ID")["MIN"]
               for gid, grp in tdf.groupby("GAME_ID")}
    for i, gid in enumerate(games):
        if i < 10:
            continue
        prev = pd.concat([by_game[x] for x in games[i - 10:i]])
        habit = prev.groupby(level=0).mean()   # minutes moyennes quand il joue
        habit = habit[habit >= 15]
        absents = habit.index.difference(by_game[gid].index)
        rows.append({"TEAM_ID": team, "GAME_ID": gid,
                     "vacated": habit[absents].sum(),
                     "n_absents": len(absents)})
vac = pd.DataFrame(rows)
df = df.merge(vac, on=["TEAM_ID", "GAME_ID"], how="inner")

m = (df["n_prior"] >= MIN_PRIOR_GAMES) & (df["prior_min"] >= MIN_PRIOR_MINUTES)
sub = df[m].copy()
sub["ecart_min"] = sub["MIN"] - sub["prior_min"]
sub["gain_vs_base"] = sub["DUNKEST"] - sub["pred_moy"]
sub["tranche"] = pd.cut(sub["vacated"], bins=[-1, 0.5, 15, 30, 50, 500])

res = sub.groupby("tranche", observed=True).agg(
    n=("DUNKEST", "size"),
    ecart_min_moyen=("ecart_min", "mean"),
    gain_moyen_vs_base=("gain_vs_base", "mean"),
).round(2)
res["part_%"] = (100 * res["n"] / res["n"].sum()).round(1)
print(res.to_string())
print("\nCorrélation vacated vs ecart_min :",
      round(sub["vacated"].corr(sub["ecart_min"]), 3))
