"""Optimiseur d'équipe Dunkest v0 : projections simples, une journée (scipy MILP)."""
import argparse
import glob
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from src.model.projections import (LOGS_PATH, PROJ_SOURCES, SHRINK_K, LogsNotFoundError,
                                   build_projections, project_dunkest)

# (guards, forwards, centers) parmi les 5 titulaires : hypothèse G-F-C, à confirmer dans l'app
FORMATIONS = {"2-2-1": (2, 2, 1), "1-2-2": (1, 2, 2), "2-1-2": (2, 1, 2),
              "1-3-1": (1, 3, 1), "3-1-1": (3, 1, 1)}
BUDGET = 95.0
MAX_PER_TEAM = 3


def load_snapshot(path=None):
    path = path or sorted(glob.glob("snapshots/prices_*.csv"))[-1]
    df = pd.read_csv(path)
    for c in ["cr", "gp", "pdk", "min"]:
        df[c] = pd.to_numeric(df[c])
    df["name"] = df.first_name + " " + df.last_name
    return df, path


def project(df):
    """Ancienne projection (source dunkest : pdk/gp du snapshot), voir src/model/projections.py."""
    return project_dunkest(df, SHRINK_K)


def solve(d, formation, budget, exclude):
    g, f, c = FORMATIONS[formation]
    pool = d[~d.name.str.lower().isin(exclude)].reset_index(drop=True)
    n = len(pool)
    X, S, M, K = 0, n, 2 * n, 3 * n          # décalages : effectif, titulaire, 6e homme, capitaine
    N = 4 * n
    pr = pool.proj.to_numpy()
    cr = pool.cr.to_numpy()
    pos = pool.position.to_numpy()
    teams = pool.team_code.to_numpy()

    # Score = pr * (s + m + 0.5*(x - s - m) + 0.5*k) = 0.5 * pr * (x + s + m + k)
    cost = np.zeros(N)
    for off in (X, S, M, K):
        cost[off:off + n] = -0.5 * pr        # on minimise l'opposé

    rows, lo, hi = [], [], []

    def add(r, l, u):
        rows.append(r)
        lo.append(l)
        hi.append(u)

    def row():
        return np.zeros(N)

    r = row(); r[X:X + n] = 1; add(r, 10, 10)            # 10 joueurs
    r = row(); r[M:M + n] = 1; add(r, 1, 1)              # un 6e homme
    r = row(); r[K:K + n] = 1; add(r, 1, 1)              # un capitaine
    r = row(); r[X:X + n] = cr; add(r, 0, budget)        # budget
    for i in range(n):
        r = row(); r[S + i] = 1; r[M + i] = 1; r[X + i] = -1; add(r, -np.inf, 0)   # rôle => dans l'effectif
        r = row(); r[K + i] = 1; r[S + i] = -1; add(r, -np.inf, 0)                  # capitaine => titulaire
    for p, n_squad, n_start in (("G", 4, g), ("F", 4, f), ("C", 2, c)):
        mask = (pos == p).astype(float)
        r = row(); r[X:X + n] = mask; add(r, n_squad, n_squad)
        r = row(); r[S:S + n] = mask; add(r, n_start, n_start)
    for t in np.unique(teams):
        r = row(); r[X:X + n] = (teams == t).astype(float); add(r, -np.inf, MAX_PER_TEAM)

    res = milp(c=cost,
               constraints=LinearConstraint(np.vstack(rows), lo, hi),
               integrality=np.ones(N), bounds=Bounds(0, 1))
    if not res.success:
        return None
    z = np.round(res.x).astype(int)
    xs, ss, ms, ks = z[X:X + n], z[S:S + n], z[M:M + n], z[K:K + n]
    out = pool[xs == 1].copy()
    out["role"] = ["capitaine" if ks[i] else "titulaire" if ss[i]
                   else "6e homme" if ms[i] else "banc" for i in out.index]
    return -res.fun, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coach-cost", type=float, default=10.0,
                    help="prix du coach (placeholder, à régler)")
    ap.add_argument("--exclude", default="", help='joueurs déjà pris : "Nom A,Nom B"')
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--proj-source", choices=PROJ_SOURCES, default="nba",
                    help="nba : tous les matchs NBA (logs) ; dunkest : pdk/gp du snapshot (défaut nba)")
    ap.add_argument("--logs", default=LOGS_PATH, help=f"logs NBA (défaut {LOGS_PATH})")
    a = ap.parse_args()
    exclude = {x.strip().lower() for x in a.exclude.split(",") if x.strip()}
    df, path = load_snapshot(a.snapshot)
    try:
        d, gp_label = build_projections(df, a.proj_source, a.logs)
    except LogsNotFoundError as e:
        raise SystemExit(str(e))
    budget = BUDGET - a.coach_cost
    print(f"Snapshot : {path} | budget joueurs : {budget:.1f}")
    best = None
    for fm in FORMATIONS:
        r = solve(d, fm, budget, exclude)
        if r:
            print(f"  {fm} : {r[0]:.1f} pts attendus (hors coach)")
            if best is None or r[0] > best[0]:
                best = (r[0], fm, r[1])
        results = {fm: solve(d, fm, budget, exclude) for fm in FORMATIONS}
    results = {k: v for k, v in results.items() if v}
    if not results:
        raise SystemExit("Aucune équipe valide : budget trop bas ou trop de joueurs exclus.")
    best_fm = max(results, key=lambda k: results[k][0])
    top = results[best_fm][0]
    print("Formation : pts attendus (écart) | titulaires")
    for fm, (v, t) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        starters = ", ".join(t[t.role.isin(["capitaine", "titulaire"])].last_name)
        print(f"  {fm} : {v:.1f} ({v - top:+.1f}) | {starters}")
    val, team = results[best_fm]
    fm = best_fm
    order = {"capitaine": 0, "titulaire": 1, "6e homme": 2, "banc": 3}
    team = team.sort_values("role", key=lambda s: s.map(order))
    print(f"\nMeilleure formation : {fm} ({val:.1f} pts)")
    print(f"Projections : source {a.proj_source}")
    print(team[["role", "name", "team_code", "position", "cr", "gp", "proj"]]
          .rename(columns={"gp": gp_label}).round(2).to_string(index=False))
    print(f"Coût total : {team.cr.sum():.1f}")