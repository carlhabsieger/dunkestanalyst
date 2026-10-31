"""Début de saison : que valent les 4 premiers matchs pour prédire les 20 suivants ?"""
import os
import sys

import numpy as np
import pandas as pd
from src.model.baseline import load_all

N_EARLY, N_NEXT = 4, 20
LOGS_PATH = "data/player_logs_2025-26.csv"
# Hypothèse : fichier produit par les fonctions de src/data/box_scores.py avec season="2025-26"
# (le __main__ de box_scores n'écrit que data/player_logs.csv pour la saison courante).
GEN_CMD = ('python -c "from src.data.box_scores import fetch_player_logs, add_dunkest; '
           "add_dunkest(fetch_player_logs(season='2025-26')).to_csv('" + LOGS_PATH + "', index=False)\"")


def A(t):
    return np.column_stack([np.ones(len(t)), t.x, t.m])


def main():
    if not os.path.exists(LOGS_PATH):
        print(f"Erreur : {LOGS_PATH} introuvable. Génère-le avec :\n  {GEN_CMD}", file=sys.stderr)
        return 1
    df = load_all(LOGS_PATH)
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

    coef, *_ = np.linalg.lstsq(A(train), train.y, rcond=None)
    pred = A(test) @ coef
    mae = lambda p: float(np.abs(p - test.y).mean())
    print(f"{len(train)} joueurs en apprentissage, {len(test)} en test")
    print(f"y = {coef[0]:.2f} + {coef[1]:.2f} * score_4_matchs + {coef[2]:.2f} * minutes_4_matchs")
    print(f"MAE moyenne des 4 premiers matchs : {mae(test.x):.2f}")
    print(f"MAE moyenne de la ligue           : {mae(train.y.mean()):.2f}")
    print(f"MAE régression (score + minutes)  : {mae(pred):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
