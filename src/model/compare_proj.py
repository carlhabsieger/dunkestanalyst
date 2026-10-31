"""Compare les projections tous matchs NBA (nba) et snapshot Dunkest (dunkest) sur le dernier snapshot.

    python -m src.model.compare_proj [--snapshot CSV] [--logs CSV] [--top 15] [--prior price|price_min]

--prior price_min (expérience, non branchée dans les outils) : compare l'a priori prix seul à un a
priori avg ~ prix + minutes moyennes par match.
"""
import argparse
import sys

import numpy as np

from src.model.projections import (LOGS_PATH, MIN_FIT_GAMES, MIN_FIT_PLAYERS, SHRINK_K,
                                   LogsNotFoundError, load_logs, match_report, project_all_games,
                                   project_dunkest)
from src.optimizer import load_snapshot

PRIORS = ("price", "price_min")
WARNING = ("ATTENTION : l'a priori price_min est ajusté sur les mêmes joueurs que ceux projetés "
           "(échantillon non indépendant). Cette comparaison montre l'effet du changement, elle ne "
           "prouve pas que price_min prédit mieux que price.")


def compare(df, logs):
    """Une ligne par joueur présent dans les deux projections (joué côté NBA et côté Dunkest)."""
    nba = project_all_games(df, logs)
    dk = project_dunkest(df)
    m = nba[nba.n > 0][["id", "name", "team_code", "position", "cr", "n", "proj"]].merge(
        dk[["id", "gp", "proj"]], on="id", suffixes=("_nba", "_dunkest"))
    m["ecart"] = m.proj_nba - m.proj_dunkest
    return nba, m


def prior_price_min(nba, k=SHRINK_K):
    """A priori avg ~ cr + mpg, ajusté comme fit_prior (n >= 3, tous les joueurs ayant joué si < 20).

    Retourne (joueurs avec n > 0 et colonnes prior_min / proj_min, coefficients [constante, cr, mpg]).
    Les joueurs sans match (n = 0, minutes inconnues) sont écartés : ils ne sont pas proposés.
    """
    p = nba[nba.n > 0].copy()
    base = p[p.n >= MIN_FIT_GAMES]
    if len(base) < MIN_FIT_PLAYERS:
        base = p
    X = lambda t: np.column_stack([np.ones(len(t)), t.cr.to_numpy(float), t.mpg.to_numpy(float)])
    coef, *_ = np.linalg.lstsq(X(base), base.avg.to_numpy(float), rcond=None)
    p["prior_min"] = X(p) @ coef
    p["proj_min"] = (p.n * p.avg + k * p.prior_min) / (p.n + k)
    p["ecart"] = p.proj_min - p.proj
    return p, coef


def show_price_min(nba, top):
    p, coef = prior_price_min(nba)
    print(f"\nA priori price_min : avg = {coef[0]:.2f} + {coef[1]:.2f} x prix + {coef[2]:.2f} x min/match "
          f"(k = {SHRINK_K})")
    print(WARNING)
    print(f"\n{top} joueurs dont la projection change le plus (price -> price_min, {len(p)} joueurs) :")
    t = p.reindex(p.ecart.abs().sort_values(ascending=False).index).head(top)
    print(t.rename(columns={"name": "nom", "team_code": "équipe", "cr": "prix", "n": "matchs NBA",
                            "mpg": "min/match", "prior": "prior_price", "proj": "proj_price",
                            "ecart": "écart"})
          [["nom", "équipe", "position", "prix", "matchs NBA", "min/match", "avg", "prior_price",
            "prior_min", "proj_price", "proj_min", "écart"]].round(2).to_string(index=False))
    print(f"\nÉcart absolu moyen : {p.ecart.abs().mean():.2f} | écart moyen : {p.ecart.mean():+.2f}")
    print(WARNING)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--logs", default=LOGS_PATH)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--prior", choices=PRIORS, default="price",
                    help="price : nba vs dunkest (défaut) ; price_min : expérience a priori prix + minutes")
    a = ap.parse_args(argv)
    df, path = load_snapshot(a.snapshot)
    try:
        logs = load_logs(a.logs)
    except LogsNotFoundError as e:
        print(f"Erreur : {e}", file=sys.stderr)
        return 1
    nba, m = compare(df, logs)
    print(f"Snapshot : {path} | logs : {a.logs} (jusqu'au {logs.GAME_DATE.max():%Y-%m-%d})")
    print(match_report(nba))
    if a.prior == "price_min":
        show_price_min(nba, a.top)
        return 0
    print(f"\n{a.top} plus grands écarts |proj_nba - proj_dunkest| ({len(m)} joueurs comparés) :")
    top = m.reindex(m.ecart.abs().sort_values(ascending=False).index).head(a.top)
    print(top.rename(columns={"name": "nom", "team_code": "équipe", "cr": "prix", "n": "matchs NBA",
                              "gp": "mj Dunkest", "ecart": "écart"})
          [["nom", "équipe", "position", "prix", "matchs NBA", "mj Dunkest", "proj_nba",
            "proj_dunkest", "écart"]].round(2).to_string(index=False))
    print("\nÉcart moyen par poste (proj_nba - proj_dunkest) :")
    by = m.groupby("position").agg(joueurs=("ecart", "size"), ecart_moyen=("ecart", "mean"),
                                   ecart_abs_moyen=("ecart", lambda s: s.abs().mean()),
                                   matchs_nba=("n", "mean"), mj_dunkest=("gp", "mean"))
    print(by.round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
