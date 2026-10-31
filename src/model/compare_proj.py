"""Compare les projections tous matchs NBA (nba) et snapshot Dunkest (dunkest) sur le dernier snapshot.

    python -m src.model.compare_proj [--snapshot CSV] [--logs CSV] [--top 15]
"""
import argparse
import sys

from src.model.projections import (LOGS_PATH, LogsNotFoundError, load_logs, match_report,
                                   project_all_games, project_dunkest)
from src.optimizer import load_snapshot


def compare(df, logs):
    """Une ligne par joueur présent dans les deux projections (joué côté NBA et côté Dunkest)."""
    nba = project_all_games(df, logs)
    dk = project_dunkest(df)
    m = nba[nba.n > 0][["id", "name", "team_code", "position", "cr", "n", "proj"]].merge(
        dk[["id", "gp", "proj"]], on="id", suffixes=("_nba", "_dunkest"))
    m["ecart"] = m.proj_nba - m.proj_dunkest
    return nba, m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--logs", default=LOGS_PATH)
    ap.add_argument("--top", type=int, default=15)
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
