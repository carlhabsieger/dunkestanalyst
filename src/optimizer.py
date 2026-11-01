"""Optimiseur d'équipe Dunkest v0 : projections simples, une journée (scipy MILP)."""
import argparse
import glob
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

import warnings

from src.data.injuries import InjuryReportError, get_report
from src.data.schedule import ScheduleError, fetch_schedule, match_labels, parse_dates
from src.model.injury_probs import DEFAULT_UNKNOWN_P, is_uncertain
from src.model.projections import (LOGS_PATH, PROJ_SOURCES, SHRINK_K, LogsNotFoundError, attach_matches,
                                   build_projections, filter_available, project_dunkest, uncertain_mark)

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
    ap.add_argument("--injuries", choices=("on", "off"), default="on",
                    help="rapport officiel des blessures : proj = p_play x proj, Out exclus (défaut on)")
    ap.add_argument("--min-p-play", type=float, default=0.0,
                    help="exclut les candidats dont la probabilité de jouer est sous ce seuil (défaut 0)")
    ap.add_argument("--unknown-p", type=float, default=DEFAULT_UNKNOWN_P,
                    help="probabilité de jouer si le statut est inconnu : rapport non soumis ou hors rapport (défaut 1)")
    ap.add_argument("--dates", default=None,
                    help="dates de match AAAA-MM-JJ séparées par des virgules : équipes sans match exclues")
    a = ap.parse_args()
    exclude = {x.strip().lower() for x in a.exclude.split(",") if x.strip()}
    df, path = load_snapshot(a.snapshot)
    warnings.formatwarning = lambda msg, *args, **kw: f"Attention : {msg}\n"
    report = (lambda: get_report()[1]) if a.injuries == "on" else None
    try:
        dates = parse_dates(a.dates) if a.dates else None
        d, gp_label = build_projections(df, a.proj_source, a.logs, injuries=report, dates=dates,
                                        unknown_p=a.unknown_p)
        labels = match_labels(fetch_schedule(), dates) if dates else None
    except LogsNotFoundError as e:
        raise SystemExit(str(e))
    except InjuryReportError as e:
        raise SystemExit(f"Rapport des blessures introuvable : {e}\n"
                         f"Relance avec --injuries off pour ignorer les blessures.")
    except ScheduleError as e:
        raise SystemExit(f"{e}\nRelance sans --dates pour ignorer le calendrier.")
    d = filter_available(d, a.min_p_play)
    if dates:
        print(f"Dates : {', '.join(map(str, dates))} | {len(labels)} équipes jouent")
        d = attach_matches(d, labels)
    budget = BUDGET - a.coach_cost
    print(f"Snapshot : {path} | budget joueurs : {budget:.1f}")
    results = {}
    for fm in FORMATIONS:                    # une seule résolution par formation
        r = solve(d, fm, budget, exclude)
        if r:
            print(f"  {fm} : {r[0]:.1f} pts attendus (hors coach)")
            results[fm] = r
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
    print(f"Projections : source {a.proj_source} | blessures : {a.injuries}"
          + (" (proj = p_play x proj si joue)" if a.injuries == "on" else ""))
    if "status" in team:
        team = team.assign(name=[f"{n} {uncertain_mark(s)}".strip() for n, s in zip(team.name, team.status)])
    cols = ["role", "name"] + (["status", "p_play"] if "status" in team else []) + \
        ["team_code"] + (["match"] if "match" in team else []) + ["position", "cr", "gp", "proj"]
    print(team[cols].rename(columns={"gp": gp_label, "status": "statut"}).round(2).to_string(index=False))
    print(f"Coût total : {team.cr.sum():.1f}")
    if "status" in team:
        n_unk = int(team.status.map(is_uncertain).sum())
        print(f"Joueurs proposés au statut inconnu (?) : {n_unk} / {len(team)}"
              + (f" (p_play = {a.unknown_p} appliqué)" if n_unk else ""))