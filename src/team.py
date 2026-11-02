"""Gestion des équipes de l'utilisateur : stockage, visualisation, composition et transferts.

    python -m src.team list
    python -m src.team init NOM [--players "A,B,..."] [--cash 1.4]
    python -m src.team show NOM [--dates a,b] [--html rapport.html]
    python -m src.team add NOM "Joueur" | remove NOM "Joueur" | swap NOM "Ancien" "Nouveau"
    python -m src.team set NOM [--cash 1.4] [--coach "..."] [--exclude "A,B"] [--exclude-teams "autre"]
                              [--price "Nom=9.5"]
    python -m src.team lineup NOM --dates a,b
    python -m src.team suggest NOM --dates a,b [--max-transfers 3] [--horizon 1] [--min-net-gain 1.0]

Un fichier par équipe : teams/<nom>.json (lisible à la main). Chaque écriture sauvegarde l'ancienne
version dans teams/.backup/.
Hypothèse NON vérifiée : un joueur vendu rapporte son prix actuel du snapshot.
"""
import argparse
import difflib
import html
import json
import os
import re
import shutil
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_array

from src.model.name_aliases import NAME_ALIASES
from src.model.projections import normalize_name
from src.optimizer import FORMATIONS, MAX_PER_TEAM, solve

TEAMS_DIR = "teams"
BACKUP_DIR = os.path.join(TEAMS_DIR, ".backup")
SQUAD_SIZE = 10
QUOTAS = {"G": 4, "F": 4, "C": 2}
POSITIONS = ("G", "F", "C")
TRANSFER_COST = 5.0            # points retranchés du score de la journée suivante, par joueur changé
NOISE_PTS = 2.0                # écart sous lequel deux options sont dans le bruit
MARKET_STATUS = {1: "trades_open", 2: "matches_in_progress", 3: "substitutions_open",
                 4: "calculation_in_progress", 5: "matchday_ended", 6: "inactive"}   # cf. snapshot_check.py
SELL_PRICE_NOTE = ("Hypothèse NON vérifiée : un joueur vendu rapporte son prix actuel du snapshot "
                   "(budget = prix actuels des joueurs possédés + cash).")
NOISE_NOTE = (f"Les projections reposent sur peu de matchs : un écart de moins de {NOISE_PTS:.0f} points "
              "entre deux options est dans le bruit.")
EMPTY_TEAM = {"players": [], "cash": 0.0, "coach": "", "excluded": [], "excluded_from_teams": [],
              "price_overrides": {}}


class TeamError(ValueError):
    """Erreur lisible (nom ambigu, effectif invalide, prix inconnu...), affichée sans traceback."""


# ---------------------------------------------------------------- stockage

def team_path(name, root=TEAMS_DIR):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise TeamError(f"Nom d'équipe invalide « {name} » (lettres, chiffres, _ et - seulement).")
    return os.path.join(root, f"{name}.json")


def list_teams(root=TEAMS_DIR):
    if not os.path.isdir(root):
        return []
    return sorted(f[:-5] for f in os.listdir(root) if f.endswith(".json"))


def load_team(name, root=TEAMS_DIR):
    path = team_path(name, root)
    if not os.path.exists(path):
        raise TeamError(f"Équipe « {name} » introuvable ({path}). Crée-la avec : python -m src.team init {name}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    team = {**EMPTY_TEAM, **data}
    team["players"] = list(team["players"])
    return team


def save_team(name, team, root=TEAMS_DIR, backup=True):
    """Écrit teams/<nom>.json ; l'ancienne version est copiée dans teams/.backup/ avant."""
    path = team_path(name, root)
    os.makedirs(root, exist_ok=True)
    if backup and os.path.exists(path):
        bdir = os.path.join(root, ".backup")
        os.makedirs(bdir, exist_ok=True)
        shutil.copy2(path, os.path.join(bdir, f"{name}-{datetime.now():%Y%m%d-%H%M%S-%f}.json"))
    ordered = {k: team.get(k, EMPTY_TEAM[k]) for k in EMPTY_TEAM}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


# ---------------------------------------------------------------- noms

def resolve_name(query, names, aliases=NAME_ALIASES, cutoff=0.8):
    """Nom Dunkest correspondant à query parmi names, sans deviner.

    Ordre : égalité normalisée (accents, casse, ponctuation), alias dans les deux sens (nom Dunkest ou
    nom NBA), sous-chaîne unique, puis correspondance approchée (difflib) unique. Ambigu ou introuvable :
    TeamError avec la liste des candidats.
    """
    names = list(dict.fromkeys(names))
    by_key = {}
    for n in names:
        by_key.setdefault(normalize_name(n), []).append(n)
    q = normalize_name(query)
    nba_to_dunkest = {normalize_name(v): k for k, v in aliases.items()}
    for key in (q, normalize_name(nba_to_dunkest.get(q, "")), normalize_name(aliases.get(query, ""))):
        if key and key in by_key:
            hits = by_key[key]
            if len(hits) > 1:
                raise TeamError(f"« {query} » est ambigu : {', '.join(hits)}.")
            return hits[0]
    sub = [n for k, ns in by_key.items() if q and q in k for n in ns]
    if len(sub) == 1:
        return sub[0]
    if len(sub) > 1:
        raise TeamError(f"« {query} » est ambigu ({len(sub)} joueurs) : {', '.join(sorted(sub)[:10])}. "
                        f"Précise le nom.")
    close = difflib.get_close_matches(q, list(by_key), n=5, cutoff=cutoff)
    if len(close) == 1:
        return by_key[close[0]][0]
    if len(close) > 1:
        raise TeamError(f"« {query} » est ambigu : {', '.join(by_key[c][0] for c in close)}. Précise le nom.")
    hint = difflib.get_close_matches(q, list(by_key), n=3, cutoff=0.5)
    sugg = f" Proches : {', '.join(by_key[c][0] for c in hint)}." if hint else ""
    raise TeamError(f"Joueur « {query} » introuvable dans le snapshot.{sugg} "
                    f"S'il n'y figure pas, ajoute un price_override (set --price).")


# ---------------------------------------------------------------- effectif

def coach_team(coach):
    """Code équipe du coach s'il est écrit « Nom (BOS) », sinon None (coach non modélisé)."""
    m = re.search(r"\(([A-Z]{3})\)\s*$", str(coach or ""))
    return m.group(1) if m else None


def player_info(team, snap):
    """Une ligne par joueur possédé : name, position, team_code, cr, in_snapshot.

    Joueur absent du snapshot : price_overrides[nom] = prix, ou {"cr", "position", "team_code"}.
    TeamError si le prix est inconnu.
    """
    rows = []
    by_name = snap.set_index("name")
    over = team.get("price_overrides", {}) or {}
    for nm in team["players"]:
        o = over.get(nm)
        if nm in by_name.index:
            r = by_name.loc[nm]
            r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
            cr = float(o["cr"] if isinstance(o, dict) else o) if o is not None else float(r.cr)
            rows.append({"name": nm, "id": r.get("id"), "position": r.position, "team_code": r.team_code,
                         "cr": cr, "in_snapshot": True})
        elif o is not None:
            o = o if isinstance(o, dict) else {"cr": o}
            rows.append({"name": nm, "id": None, "position": o.get("position"), "team_code": o.get("team_code"),
                         "cr": float(o["cr"]), "in_snapshot": False})
        else:
            raise TeamError(f"« {nm} » est absent du snapshot et n'a pas de prix : ajoute un price_override, "
                            f'par exemple : python -m src.team set NOM --price "{nm}=9.5" '
                            f"(ou un objet {{\"cr\": 9.5, \"position\": \"F\", \"team_code\": \"BOS\"}} dans le JSON).")
    return pd.DataFrame(rows, columns=["name", "id", "position", "team_code", "cr", "in_snapshot"])


def check_squad(info, coach=""):
    """Lève TeamError si l'effectif dépasse 10 joueurs, un quota de poste ou 3 joueurs par équipe NBA."""
    if len(info) > SQUAD_SIZE:
        raise TeamError(f"{len(info)} joueurs : l'effectif est limité à {SQUAD_SIZE}.")
    if info.position.isna().any():
        miss = ", ".join(info[info.position.isna()].name)
        raise TeamError(f"Poste inconnu pour : {miss} (absent du snapshot) ; indique-le dans price_overrides "
                        f'sous la forme {{"cr": 9.5, "position": "F", "team_code": "BOS"}}.')
    for p, q in QUOTAS.items():
        n = int((info.position == p).sum())
        if n > q:
            raise TeamError(f"Trop de {p} : {n} (maximum {q}).")
    counts = info.team_code.dropna().value_counts()
    ct = coach_team(coach)
    if ct:
        counts[ct] = counts.get(ct, 0) + 1
    over = counts[counts > MAX_PER_TEAM]
    if len(over):
        raise TeamError(f"Plus de {MAX_PER_TEAM} joueurs de {', '.join(over.index)} (coach compris s'il est noté "
                        f"« Nom (EQU) »).")


def add_player(team, query, snap):
    names = list(snap.name) + list((team.get("price_overrides") or {}).keys())
    nm = resolve_name(query, names)
    if nm in team["players"]:
        raise TeamError(f"{nm} est déjà dans l'équipe.")
    new = {**team, "players": team["players"] + [nm]}
    check_squad(player_info(new, snap), new.get("coach"))
    return new, nm


def remove_player(team, query):
    nm = resolve_name(query, team["players"])
    return {**team, "players": [p for p in team["players"] if p != nm]}, nm


def swap_player(team, old_query, new_query, snap):
    t, out = remove_player(team, old_query)
    t, inn = add_player(t, new_query, snap)
    return t, out, inn


# ---------------------------------------------------------------- projections de l'effectif

def owned_frame(team, snap, d, labels=None):
    """Joueurs possédés avec prix, projection, statut et match.

    d : projections (build_projections, blessures comprises, non filtrées). Un joueur absent de d (aucun
    match NBA), Out, ou dont l'équipe ne joue pas dans la fenêtre (labels) compte pour 0.
    """
    info = player_info(team, snap)
    cols = [c for c in ["id", "proj", "status", "p_play", "reason", "gp"] if c in d]
    o = info.merge(d[cols], on="id", how="left") if "id" in d else info.assign(proj=np.nan)
    notes = [[] for _ in range(len(o))]

    def note(mask, text):
        for i in np.flatnonzero(np.asarray(mask, dtype=bool)):
            notes[i].append(text)

    note(~o.in_snapshot, "absent du snapshot (prix manuel), projection 0")
    note(o.in_snapshot & o.proj.isna(), "aucune projection (aucun match NBA), compte 0")
    if "status" in o:
        note(o.status == "Out", "Out, compte 0")
    if labels is not None:
        o["match"] = o.team_code.map(labels)
        note(o.match.isna(), "sans match dans la fenêtre, compte 0")
        o.loc[o.match.isna(), "proj"] = 0.0
    o["note"] = [" ; ".join(n) for n in notes]
    o["proj"] = o.proj.fillna(0.0)
    return o


# ---------------------------------------------------------------- composition sans transfert

def lineup(owned):
    """Meilleure composition des 10 joueurs possédés (optimizer.solve restreint à l'effectif).

    Retourne (valeur, formation, effectif avec la colonne role).
    """
    if len(owned) != SQUAD_SIZE:
        raise TeamError(f"Effectif incomplet ({len(owned)}/{SQUAD_SIZE}) : complète-le (add, ou python -m src.buy).")
    check_squad(owned)
    best = None
    for fm in FORMATIONS:
        r = solve(owned.reset_index(drop=True), fm, budget=np.inf, exclude=set())
        if r and (best is None or r[0] > best[0] + 1e-9):
            best = (r[0], fm, r[1])
    if best is None:
        raise TeamError("Aucune composition valide (quotas de postes ?).")
    return best


# ---------------------------------------------------------------- transferts

@dataclass
class Plan:
    k_max: int
    value: float                      # score pondéré de la composition (achats x horizon)
    squad: pd.DataFrame               # 10 joueurs, colonnes role, owned
    formation: str
    out: list = field(default_factory=list)
    inn: list = field(default_factory=list)
    credits_left: float = 0.0

    @property
    def n_transfers(self):
        return len(self.inn)

    def moves(self, owned):
        """Paires (sortant, entrant) associées par poste (les quotas imposent autant de sorties que
        d'entrées à chaque poste)."""
        pos_out = owned.set_index("name").position
        pos_in = self.squad.set_index("name").position
        left = list(self.out)
        pairs = []
        for i in self.inn:
            o = next((x for x in left if pos_out[x] == pos_in[i]), left[0])
            left.remove(o)
            pairs.append((o, i))
        return pairs


def solve_transfers(owned, candidates, budget, max_new, horizon=1, coach="", exact=False):
    """MILP : meilleur effectif de 10 avec au plus max_new joueurs non possédés (exactement si exact).

    Objectif : somme pondérée des projections (titulaires et 6e homme 1,0, banc 0,5, capitaine +0,5),
    les joueurs achetés comptant horizon fois, MOINS 5 points par joueur acheté.
    """
    pool = pd.concat([owned.assign(owned=True),
                      candidates[~candidates.name.isin(set(owned.name))].assign(owned=False)],
                     ignore_index=True)
    n = len(pool)
    pr = pool.proj.to_numpy(float)
    cr = pool.cr.to_numpy(float)
    pos = pool.position.to_numpy()
    teams = pool.team_code.to_numpy()
    new = (~pool.owned).to_numpy()
    w = np.where(new, float(horizon), 1.0)
    X, S, M, K, Y = 0, n, 2 * n, 3 * n, 4 * n
    fms = list(FORMATIONS.items())
    N = Y + len(fms)
    c = np.zeros(N)
    for off in (X, S, M, K):                 # score = 0,5 x pr x (x + s + m + k), cf. optimizer.solve
        c[off:off + n] = -0.5 * w * pr
    c[X:X + n] += TRANSFER_COST * new

    rows, cols, vals, lo, hi = [], [], [], [], []

    def add(cc, vv, l, u):
        cc = np.asarray(cc).ravel()
        rows.extend([len(lo)] * len(cc))
        cols.extend(cc)
        vals.extend(np.broadcast_to(vv, cc.shape))
        lo.append(l)
        hi.append(u)

    idx = np.arange(n)
    add(X + idx, 1.0, SQUAD_SIZE, SQUAD_SIZE)
    add(M + idx, 1.0, 1, 1)
    add(K + idx, 1.0, 1, 1)
    add(X + idx, cr, -np.inf, budget + 1e-6)
    add(X + idx[new], 1.0, max_new if exact else -np.inf, max_new)
    for i in idx:
        add([S + i, M + i, X + i], [1.0, 1.0, -1.0], -np.inf, 0)    # rôle => dans l'effectif
        add([K + i, S + i], [1.0, -1.0], -np.inf, 0)                 # capitaine => titulaire
    add(np.arange(Y, N), 1.0, 1, 1)                                  # une formation
    for j, p in enumerate(POSITIONS):
        mask = idx[pos == p]
        add(X + mask, 1.0, QUOTAS[p], QUOTAS[p])
        add(np.concatenate([S + mask, np.arange(Y, N)]),
            np.concatenate([np.ones(len(mask)), [-f[j] for _, f in fms]]), 0, 0)
    ct = coach_team(coach)
    for t in pd.unique(teams):
        add(X + idx[teams == t], 1.0, -np.inf, MAX_PER_TEAM - (1 if t == ct else 0))

    A = csr_array((vals, (rows, cols)), shape=(len(lo), N))
    res = milp(c=c, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(N), bounds=Bounds(0, 1))
    if not res.success:
        return None
    z = np.round(res.x).astype(int)
    sel = z[X:X + n] == 1
    squad = pool[sel].copy()
    squad["role"] = ["capitaine" if z[K + i] else "titulaire" if z[S + i] else "6e homme" if z[M + i]
                     else "banc" for i in np.flatnonzero(sel)]
    weight = squad.role.map({"capitaine": 1.5, "titulaire": 1.0, "6e homme": 1.0, "banc": 0.5})
    value = float((squad.proj * weight * np.where(squad.owned, 1.0, horizon)).sum())
    fm = fms[int(np.argmax(z[Y:N]))][0]
    plan = Plan(max_new, value, squad, fm,
                out=sorted(set(owned.name) - set(squad.name)),
                inn=list(squad[~squad.owned].name),
                credits_left=round(budget - float(squad.cr.sum()), 2))
    return plan


@dataclass
class Option:
    k: int                 # limite --max-transfers testée
    plan: Plan             # meilleur plan avec au plus k transferts (objectif net)
    gross: float           # gain brut par rapport à la baseline
    cost: float            # 5 x nombre de transferts
    net: float
    forced: Plan = None    # meilleur plan à exactement k transferts, si le plan ci-dessus en fait moins


def suggest(owned, candidates, cash, max_transfers=3, horizon=1, min_net_gain=1.0, coach=""):
    """Pour k = 0..max_transfers : meilleur plan (au plus k transferts) et, s'il en fait moins, le meilleur
    plan à exactement k transferts (pour montrer ce qu'il rapporterait). Retourne (baseline, options,
    recommandée ou None). Une option n'est recommandée que si son gain net >= min_net_gain.
    """
    if len(owned) != SQUAD_SIZE:
        raise TeamError(f"Effectif incomplet ({len(owned)}/{SQUAD_SIZE}) : suggest compare des transferts "
                        f"un pour un ; complète d'abord l'équipe (add, ou python -m src.buy).")
    budget = float(owned.cr.sum()) + float(cash)
    base = solve_transfers(owned, candidates.iloc[0:0], budget, 0, horizon, coach)
    if base is None:
        raise TeamError("L'effectif actuel ne respecte pas les règles (postes, 3 par équipe, budget).")
    options, best = [], None
    for k in range(0, max_transfers + 1):
        p = solve_transfers(owned, candidates, budget, k, horizon, coach)
        if p is None:
            continue
        gross = p.value - base.value
        cost = TRANSFER_COST * p.n_transfers
        opt = Option(k, p, gross, cost, gross - cost)
        if k and p.n_transfers < k:
            opt.forced = solve_transfers(owned, candidates, budget, k, horizon, coach, exact=True)
        options.append(opt)
        if p.n_transfers and opt.net >= min_net_gain and (best is None or opt.net > best.net + 1e-9):
            best = opt
    return base, options, best


# ---------------------------------------------------------------- contexte (réseau) et affichage

def market_warning(get_state=None):
    """Avertissement si le marché n'est pas ouvert (statut 1), ou si l'état est inconnu. Ne bloque rien."""
    try:
        if get_state is None:
            from src.data.snapshot_prices import get_state
        status = get_state()[2]
    except Exception as e:
        return f"État du marché inconnu ({type(e).__name__}) : vérifie dans l'app avant de transférer."
    if status != 1:
        return (f"Marché fermé (statut {status} : {MARKET_STATUS.get(status, '?')}) : les transferts ne sont pas "
                f"possibles maintenant ; la composition (lineup) reste utile.")
    return None


def load_context(a):
    """Snapshot, projections (blessures et calendrier comme optimizer.py) et étiquettes de match."""
    from src.data.injuries import get_report
    from src.data.schedule import fetch_schedule, first_game_dates, match_labels, parse_dates
    from src.model.projections import build_projections
    from src.optimizer import load_snapshot
    warnings.formatwarning = lambda msg, *args, **kw: f"Attention : {msg}\n"
    snap, path = load_snapshot(a.snapshot)
    dates = parse_dates(a.dates) if a.dates else None
    schedule = fetch_schedule() if dates else None
    fg = first_game_dates(schedule, dates) if dates else None
    report = (lambda: get_report()[1]) if a.injuries == "on" else None
    d, gp_label = build_projections(snap, a.proj_source, a.logs, injuries=report, dates=dates,
                                    first_games=fg, unknown_p=a.unknown_p, stale_days=a.stale_days)
    labels = match_labels(schedule, dates) if dates else None
    print(f"Snapshot : {path}" + (f" | dates : {', '.join(map(str, dates))}" if dates else
                                  " | sans --dates : calendrier non pris en compte"))
    return snap, d, labels, gp_label


def candidates_for(d, team, labels, min_gp=1, allow_out=False, other_players=()):
    """Candidats à l'achat : filtre --min-gp, non Out (sauf allow_out), équipe qui joue dans la fenêtre,
    hors excluded et hors joueurs des équipes de excluded_from_teams."""
    from src.model.projections import attach_matches
    c = d[d.gp >= min_gp]
    if "status" in c and not allow_out:
        c = c[c.status != "Out"]
    if labels is not None:
        c = attach_matches(c, labels)
    banned = {normalize_name(x) for x in list(team.get("excluded", [])) + list(other_players)}
    return c[~c.name.map(normalize_name).isin(banned)].copy()


def team_warnings(owned):
    msgs = []
    for _, r in owned.iterrows():
        if r.note:
            msgs.append(f"{r['name']} : {r.note}")
    return msgs


def fmt_table(df, gp_label="matchs NBA"):
    from src.model.projections import short_reason, uncertain_mark
    t = pd.DataFrame({
        "rôle": df.get("role", ""),
        "joueur": [f"{n} {uncertain_mark(s)}".strip() for n, s in zip(df.name, df.get("status", [""] * len(df)))],
        "poste": df.position, "équipe": df.team_code, "prix": df.cr.map(lambda v: f"{v:.1f}"),
    })
    if "match" in df:
        t["match"] = df.match.fillna("aucun")
    if "status" in df:
        t["statut"] = df.status.fillna("")
        t["p_play"] = df.p_play.map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
        t["motif"] = df.reason.map(short_reason, na_action="ignore").fillna("")
    t["proj"] = df.proj.map(lambda v: f"{v:.1f}")
    return t


ROLE_ORDER = {"capitaine": 0, "titulaire": 1, "6e homme": 2, "banc": 3}


def render_html(title, summary_lines, table, warnings_list=()):
    """Rapport HTML autonome (un fichier, CSS en ligne, rien d'externe)."""
    def row_class(r):
        s = str(r.get("statut", ""))
        if s == "Out" or r.get("match") == "aucun":
            return "bad"
        if "?" in str(r.get("joueur", "")) or s in ("Questionable", "Doubtful"):
            return "warn"
        return ""
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in table.columns)
    body = "".join(
        f"<tr class='{row_class(r)}'>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r.values) + "</tr>"
        for _, r in table.iterrows())
    warn = "".join(f"<li>{html.escape(w)}</li>" for w in warnings_list)
    summ = "".join(f"<p>{html.escape(s)}</p>" for s in summary_lines)
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#1b1b1b;background:#fafafa}}
h1{{font-size:20px}} table{{border-collapse:collapse;background:#fff;font-size:14px}}
th,td{{border:1px solid #ccc;padding:4px 8px;text-align:left}} th{{background:#e8eef7}}
tr.bad td{{background:#f8d7da}} tr.warn td{{background:#fff3cd}}
.w{{border-left:4px solid #b02a37;padding:4px 12px;background:#fff}}
</style></head><body><h1>{html.escape(title)}</h1>
{f"<ul class='w'>{warn}</ul>" if warn else ""}{summ}<table><tr>{head}</tr>{body}</table></body></html>
"""


# ---------------------------------------------------------------- CLI

def _common(p, dates_help="dates de match AAAA-MM-JJ (premier match de chaque équipe dans la fenêtre)"):
    from src.model.injury_probs import DEFAULT_UNKNOWN_P
    from src.model.projections import LOGS_PATH, PROJ_SOURCES
    p.add_argument("--dates", default=None, help=dates_help)
    p.add_argument("--injuries", choices=("on", "off"), default="on")
    p.add_argument("--unknown-p", type=float, default=DEFAULT_UNKNOWN_P)
    p.add_argument("--stale-days", type=int, default=7)
    p.add_argument("--proj-source", choices=PROJ_SOURCES, default="nba")
    p.add_argument("--logs", default=LOGS_PATH)
    p.add_argument("--snapshot", default=None)


def _owned_with_projection(a):
    team = load_team(a.name)
    snap, d, labels, gp_label = load_context(a)
    return team, snap, d, labels, owned_frame(team, snap, d, labels)


def cmd_show(a):
    team, snap, d, labels, owned = _owned_with_projection(a)
    warns = team_warnings(owned)
    for w in warns:
        print(f"ATTENTION : {w}")
    value = formation = None
    if len(owned) == SQUAD_SIZE:
        value, formation, lu = lineup(owned)
        owned = owned.merge(lu[["name", "role"]], on="name", how="left")
        owned = owned.sort_values("role", key=lambda s: s.map(ROLE_ORDER))
    else:
        print(f"Effectif incomplet ({len(owned)}/{SQUAD_SIZE}) : pas de composition calculée.")
    table = fmt_table(owned)
    print(table.to_string(index=False))
    cost = float(owned.cr.sum())
    lines = []
    if value is not None:
        lines.append(f"Formation recommandée : {formation} | points attendus : {value:.1f} (hors coach)")
    lines.append(f"Coût des joueurs : {cost:.1f} | cash : {team['cash']:.1f} | budget : {cost + team['cash']:.1f} "
                 f"| crédits restants : {team['cash']:.1f}")
    lines.append(f"Coach : {team.get('coach') or 'non renseigné'} (non modélisé)")
    lines.append(NOISE_NOTE)
    for l in lines:
        print(l)
    if a.html:
        with open(a.html, "w", encoding="utf-8") as f:
            f.write(render_html(f"Équipe {a.name}", lines, table, warns))
        print(f"Rapport HTML : {a.html}")
    return 0


def cmd_lineup(a):
    w = market_warning()
    if w:
        print(f"Info : {w}")
    team, snap, d, labels, owned = _owned_with_projection(a)
    for x in team_warnings(owned):
        print(f"ATTENTION : {x}")
    value, fm, lu = lineup(owned)
    lu = owned.merge(lu[["name", "role"]], on="name").sort_values("role", key=lambda s: s.map(ROLE_ORDER))
    print(f"\nMeilleure composition sans transfert : {fm} | {value:.1f} points attendus (hors coach)")
    print(fmt_table(lu).to_string(index=False))
    print(NOISE_NOTE)
    return 0


def cmd_suggest(a):
    w = market_warning()
    if w:
        print(f"ATTENTION : {w}")
    team, snap, d, labels, owned = _owned_with_projection(a)
    for x in team_warnings(owned):
        print(f"ATTENTION : {x}")
    others = []
    for t in team.get("excluded_from_teams", []):
        others += load_team(t)["players"]
    cands = candidates_for(d, team, labels, a.min_gp, a.allow_out, others)
    print(f"{len(cands)} candidats (filtre --min-gp {a.min_gp}" + ("" if a.allow_out else ", sans les Out")
          + (", équipe qui joue dans la fenêtre" if labels is not None else "") + ")")
    base, plans, best = suggest(owned, cands, team["cash"], a.max_transfers, a.horizon, a.min_net_gain,
                                team.get("coach"))
    print(f"\nBudget : {owned.cr.sum():.1f} (joueurs) + {team['cash']:.1f} (cash) = {owned.cr.sum() + team['cash']:.1f}")
    print(SELL_PRICE_NOTE)
    if a.horizon > 1:
        print(f"Horizon {a.horizon} : les projections des joueurs achetés comptent {a.horizon} fois, le coût de "
              f"5 points une seule fois (option optimiste).")
    lu_value, lu_fm, _ = lineup(owned)          # même valeur ; formation identique à show / lineup en cas d'égalité
    print(f"\nBaseline (aucun transfert) : {base.value:.1f} points attendus, formation {lu_fm}"
          + ("" if abs(lu_value - base.value) < 1e-6 else f" (lineup : {lu_value:.1f}, écart à vérifier)"))
    for o in plans:
        if o.k == 0:
            continue
        p = o.plan
        print(f"\nk <= {o.k} : {p.n_transfers} transfert(s) | gain brut {o.gross:+.1f} | coût {o.cost:.0f} | "
              f"gain NET {o.net:+.1f} | crédits restants {p.credits_left:.1f} | formation {p.formation}")
        _print_moves(p)
        if o.forced is not None:
            fg = o.forced.value - base.value
            print(f"  (meilleure option à exactement {o.k} transfert(s) : brut {fg:+.1f}, coût "
                  f"{TRANSFER_COST * o.k:.0f}, net {fg - TRANSFER_COST * o.k:+.1f} : non rentable)")
            _print_moves(o.forced, indent="    ")
    if best:
        p = best.plan
        print(f"\nRecommandation : {p.n_transfers} transfert(s), gain net {best.net:+.1f} "
              f"(seuil --min-net-gain {a.min_net_gain}) : " + ", ".join(f"{o} -> {i}" for o, i in p.moves(owned)))
        if best.net < NOISE_PTS:
            print(f"  Gain net inférieur à {NOISE_PTS:.0f} points : dans le bruit, à prendre avec prudence.")
    else:
        print(f"\nAucun transfert recommandé (aucun gain net >= {a.min_net_gain}).")
    print(NOISE_NOTE)
    return 0


def _print_moves(p, indent="  "):
    if p.n_transfers == 0:
        print(f"{indent}aucun transfert retenu avec cette limite")
        return
    print(f"{indent}sortants : " + ", ".join(p.out))
    inn = p.squad[~p.squad.owned]
    for _, r in inn.iterrows():
        st = f"{r.status} p={r.p_play:.2f}, " if "status" in inn and pd.notna(r.get("status")) else ""
        m = f", {r.match}" if "match" in inn and pd.notna(r.get("match")) else ""
        print(f"{indent}entrant  : {r['name']} ({r.position}, {r.team_code}, {r.cr:.1f} cr, {st}proj {r.proj:.1f}"
              f"{m}, {r.role})")


def _snap(a):
    from src.optimizer import load_snapshot
    return load_snapshot(getattr(a, "snapshot", None))[0]


def cmd_list(a):
    names = list_teams()
    if not names:
        print("Aucune équipe : python -m src.team init NOM")
    snap = _snap(a) if names else None
    for n in names:
        t = load_team(n)
        try:
            v = f"{player_info(t, snap).cr.sum():.1f}"
        except TeamError:
            v = "?"
        print(f"{n:15s} {len(t['players']):2d}/10 joueurs | valeur {v} | cash {t['cash']:.1f} | coach {t['coach'] or '-'}")
    return 0


def cmd_init(a):
    if os.path.exists(team_path(a.name)):
        raise TeamError(f"L'équipe « {a.name} » existe déjà ({team_path(a.name)}).")
    team = {**EMPTY_TEAM, "cash": a.cash or 0.0}
    if a.players:
        snap = _snap(a)
        for q in [x.strip() for x in a.players.split(",") if x.strip()]:
            team, _ = add_player(team, q, snap)
    print(f"Créée : {save_team(a.name, team)} ({len(team['players'])} joueurs)")
    return 0


def cmd_add(a):
    team = load_team(a.name)
    team, nm = add_player(team, a.player, _snap(a))
    save_team(a.name, team)
    print(f"Ajouté : {nm} ({len(team['players'])}/10)")
    return 0


def cmd_remove(a):
    team, nm = remove_player(load_team(a.name), a.player)
    save_team(a.name, team)
    print(f"Retiré : {nm} ({len(team['players'])}/10)")
    return 0


def cmd_swap(a):
    team, out, inn = swap_player(load_team(a.name), a.old, a.new, _snap(a))
    save_team(a.name, team)
    print(f"Échange : {out} -> {inn}")
    return 0


def cmd_set(a):
    team = load_team(a.name)
    if a.cash is not None:
        team["cash"] = a.cash
    if a.coach is not None:
        team["coach"] = a.coach
    if a.exclude is not None:
        team["excluded"] = [x.strip() for x in a.exclude.split(",") if x.strip()]
    if a.exclude_teams is not None:
        others = [x.strip() for x in a.exclude_teams.split(",") if x.strip()]
        for o in others:
            load_team(o)
        team["excluded_from_teams"] = others
    for item in a.price or []:
        nm, sep, val = item.rpartition("=")
        if not sep:
            raise TeamError(f"--price attend Nom=prix, reçu « {item} ».")
        team.setdefault("price_overrides", {})[nm.strip()] = float(val)
    save_team(a.name, team)
    print(f"Équipe {a.name} mise à jour.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Gestion des équipes Dunkest de l'utilisateur.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").add_argument("--snapshot", default=None)
    p = sub.add_parser("init")
    p.add_argument("name"); p.add_argument("--players", default=None); p.add_argument("--cash", type=float)
    p.add_argument("--snapshot", default=None)
    p = sub.add_parser("show"); p.add_argument("name"); _common(p); p.add_argument("--html", default=None)
    p = sub.add_parser("lineup"); p.add_argument("name"); _common(p)
    p = sub.add_parser("suggest"); p.add_argument("name"); _common(p)
    p.add_argument("--max-transfers", type=int, default=3)
    p.add_argument("--horizon", type=int, default=1, help="optimiste : achats comptés H fois, coût une fois")
    p.add_argument("--min-net-gain", type=float, default=1.0)
    p.add_argument("--min-gp", type=int, default=1)
    p.add_argument("--allow-out", action="store_true")
    for c in ("add", "remove"):
        p = sub.add_parser(c); p.add_argument("name"); p.add_argument("player"); p.add_argument("--snapshot", default=None)
    p = sub.add_parser("swap"); p.add_argument("name"); p.add_argument("old"); p.add_argument("new")
    p.add_argument("--snapshot", default=None)
    p = sub.add_parser("set"); p.add_argument("name")
    p.add_argument("--cash", type=float); p.add_argument("--coach"); p.add_argument("--exclude")
    p.add_argument("--exclude-teams"); p.add_argument("--price", action="append", help='"Nom=9.5", répétable')
    a = ap.parse_args(argv)
    cmds = {"list": cmd_list, "init": cmd_init, "show": cmd_show, "lineup": cmd_lineup, "suggest": cmd_suggest,
            "add": cmd_add, "remove": cmd_remove, "swap": cmd_swap, "set": cmd_set}
    from src.data.injuries import InjuryReportError
    from src.data.schedule import ScheduleError
    from src.model.projections import LogsNotFoundError
    try:
        return cmds[a.cmd](a)
    except TeamError as e:
        print(f"Erreur : {e}", file=sys.stderr)
        return 1
    except InjuryReportError as e:
        print(f"Erreur : rapport des blessures introuvable : {e}\nRelance avec --injuries off.", file=sys.stderr)
        return 1
    except (ScheduleError, LogsNotFoundError) as e:
        print(f"Erreur : {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
