"""Début de saison : que valent les 4 premiers matchs pour prédire les 20 suivants ?"""
import numpy as np
import pandas as pd
from src.model.baseline import load_all

N_EARLY, N_NEXT = 4, 20
df = load_all("data/player_logs_2025-26.csv")
rows = []
for pid, g in df.groupby("PLAYER_ID"):
    g = g.sort_values("GAME_DATE")
    if len(g) < N_EARLY + N_NEXT:
        continue
    e, n = g.iloc[:N_EARLY], g.iloc[N_EARLY:N_EARLY + N_NEXT]
    rows.append({"pid": pid, "x": e.DUNKEST.mean(), "m": e.MIN.mean(),
                 "y": n.DUNKEST.mean()})
d = pd.DataFrame(rows)
train, test = d[d.pid % 2 == 0], d[d.pid % 2 == 1]


def A(t):
    return np.column_stack([np.ones(len(t)), t.x, t.m])


coef, *_ = np.linalg.lstsq(A(train), train.y, rcond=None)
pred = A(test) @ coef
mae = lambda p: float(np.abs(p - test.y).mean())
print(f"{len(train)} joueurs en apprentissage, {len(test)} en test")
print(f"y = {coef[0]:.2f} + {coef[1]:.2f} * score_4_matchs + {coef[2]:.2f} * minutes_4_matchs")
print(f"MAE moyenne des 4 premiers matchs : {mae(test.x):.2f}")
print(f"MAE moyenne de la ligue           : {mae(train.y.mean()):.2f}")
print(f"MAE régression (score + minutes)  : {mae(pred):.2f}")