"""Demande précise : acheter quelques joueurs par rôle, poste et fourchette de prix (scipy MILP).

Exemple :
    python -m src.buy --buy "T:G@8-12,R:F@-5,6:*" --credits 25 --captain

Chaque slot s'écrit [rôle:]poste[@min-max] :
    rôle  : T (titulaire, défaut), R (remplaçant, banc à 50 %), 6 (sixième homme, 100 %)
    poste : G, F, C ou * (tous)
    prix  : "@8-12", "@-5" (max seul), "@8-" (min seul), "@9.5-9.5" ou "@9.5" (prix exact)
"""
import argparse
import sys
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_array

# Hypothèse NON vérifiée (reprise de src/optimizer.py) : les formations sont (G, F, C) parmi les titulaires.
import warnings

from src.data.injuries import InjuryReportError, get_report
from src.data.schedule import ScheduleError, fetch_schedule, first_game_dates, match_labels, parse_dates
from src.model.injury_probs import DEFAULT_UNKNOWN_P, is_uncertain
from src.model.projections import (LOGS_PATH, PROJ_SOURCES, LogsNotFoundError, attach_matches,
                                   build_projections, filter_available, short_reason, uncertain_mark)
from src.optimizer import FORMATIONS, MAX_PER_TEAM, load_snapshot

POSITIONS = ("G", "F", "C")
SQUAD_QUOTAS = {"G": 4, "F": 4, "C": 2}
SQUAD_SIZE = 10
N_STARTERS = 5
ROLES = ("T", "R", "6")
ROLE_WEIGHT = {"T": 1.0, "6": 1.0, "R": 0.5}   # le 6e homme compte à 100 % même sur le banc
ROLE_MAX = {"T": 5, "R": 4, "6": 1}
ROLE_LABEL = {"T": "titulaire", "R": "remplaçant", "6": "6e homme"}
CAPTAIN_BONUS = 0.5                             # capitaine : 1,5 x proj au lieu de 1,0 x proj
EPS = 1e-6


class BuyError(ValueError):
    """Demande invalide ou impossible à satisfaire, avec un message lisible."""


@dataclass(frozen=True)
class Slot:
    role: str
    pos: str
    lo: float = -np.inf
    hi: float = np.inf
    text: str = ""

    def accepts(self, pos, cr):
        return (self.pos == "*" or pos == self.pos) and self.lo - EPS <= cr <= self.hi + EPS


@dataclass
class Option:
    value: float          # points attendus pondérés (rôle et capitaine inclus)
    cost: float
    table: pd.DataFrame   # une ligne par slot, dans l'ordre de la demande


# ---------------------------------------------------------------- parsing

def _price(s, slot_text):
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        raise BuyError(f"Prix invalide « {s} » dans le slot « {slot_text} ».") from None


def parse_slot(text):
    raw = text.strip()
    t = raw.upper()
    head, _, rng = t.partition("@")
    role, _, pos = head.rpartition(":")
    role = role or "T"
    if role not in ROLES:
        raise BuyError(f"Rôle inconnu « {role} » dans « {raw} » (attendu : T, R ou 6).")
    if pos not in POSITIONS + ("*",):
        raise BuyError(f"Poste inconnu « {pos} » dans « {raw} » (attendu : G, F, C ou *).")
    lo, hi = -np.inf, np.inf
    if "@" in t:
        if "-" in rng:
            a, _, b = rng.partition("-")
            a, b = _price(a, raw), _price(b, raw)
            if a is None and b is None:
                raise BuyError(f"Fourchette vide dans « {raw} ».")
        else:
            a = b = _price(rng, raw)
            if a is None:
                raise BuyError(f"Fourchette vide dans « {raw} ».")
        lo = -np.inf if a is None else a
        hi = np.inf if b is None else b
        if lo > hi:
            raise BuyError(f"Fourchette inversée dans « {raw} » ({lo} > {hi}).")
    return Slot(role, pos, lo, hi, raw)


def parse_buy(spec):
    slots = [parse_slot(s) for s in spec.split(",") if s.strip()]
    if not slots:
        raise BuyError("Demande vide : précise au moins un slot, par exemple --buy \"T:G@8-12\".")
    return slots


def norm(name):
    """Nom en minuscules, sans accents ni espaces superflus (Jokić == jokic)."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(s.lower().split())


def split_names(s):
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def parse_owned_roles(s):
    """ "Nom A:T,Nom B:R" -> {"nom a": "T", "nom b": "R"} """
    out = {}
    for item in split_names(s):
        name, sep, role = item.rpartition(":")
        role = role.strip().upper()
        if not sep or not name.strip() or role not in ROLES:
            raise BuyError(f"--owned-roles : « {item} » doit être de la forme Nom:T, Nom:R ou Nom:6.")
        out[norm(name)] = role
    return out


def find_players(df, names):
    """Lignes de df dont le nom correspond exactement (accents et casse ignorés), et noms introuvables."""
    keys = df.name.map(norm)
    wanted = [norm(n) for n in names]
    missing = [n for n, w in zip(names, wanted) if not (keys == w).any()]
    return df[keys.isin(wanted)], missing


# ---------------------------------------------------------------- contrôles

def _owned_frame(owned):
    if owned is None or len(owned) == 0:
        return pd.DataFrame({"name": [], "team_code": [], "position": [], "role": []})
    owned = owned.copy()
    if "role" not in owned:
        owned["role"] = None
    return owned


def _fmt(counts):
    return f"{counts['G']}-{counts['F']}-{counts['C']}"


def check_request(slots, owned=None):
    """Lève BuyError si la demande (joueurs possédés inclus) viole une règle Dunkest."""
    owned = _owned_frame(owned)
    roles = owned.role.fillna("")

    n_total = len(slots) + len(owned)
    if n_total > SQUAD_SIZE:
        raise BuyError(f"{len(slots)} slots demandés + {len(owned)} joueurs possédés = {n_total} "
                       f"> {SQUAD_SIZE} joueurs dans l'effectif.")

    for r in ROLES:
        n = sum(s.role == r for s in slots) + int((roles == r).sum())
        if n > ROLE_MAX[r]:
            raise BuyError(f"Trop de {ROLE_LABEL[r]}s : {n} demandés (possédés inclus), "
                           f"maximum {ROLE_MAX[r]}.")

    # Formation contrôlée avant les quotas d'effectif : pour des titulaires, c'est le message le plus précis.
    need ={p: sum(s.role == "T" and s.pos == p for s in slots)
            + int(((roles == "T") & (owned.position == p)).sum()) for p in POSITIONS}
    n_any = sum(s.role == "T" and s.pos == "*" for s in slots)
    if not formations_for(need, n_any):
        avail = ", ".join(FORMATIONS)
        joker = f" + {n_any} titulaire(s) tous postes" if n_any else ""
        too_many = [f"{need[p]} {p} (maximum {max(f[i] for f in FORMATIONS.values())})"
                    for i, p in enumerate(POSITIONS) if need[p] > max(f[i] for f in FORMATIONS.values())]
        detail = (f" Trop de titulaires à un poste : {', '.join(too_many)}." if too_many else "")
        raise BuyError(
            f"Aucune formation ne convient aux titulaires demandés (possédés inclus) : "
            f"G-F-C = {_fmt(need)}{joker}, soit {sum(need.values()) + n_any} titulaires.{detail} "
            f"Il faudrait une formation au moins égale à {_fmt(need)} avec 5 titulaires au total ; "
            f"formations disponibles : {avail}.")

    for p, quota in SQUAD_QUOTAS.items():
        n = sum(s.pos == p for s in slots) + int((owned.position == p).sum())
        if n > quota:
            raise BuyError(f"Trop de {p} : {n} dans l'effectif (slots + possédés), maximum {quota}.")

    for team, n in owned.team_code.value_counts().items():
        if n > MAX_PER_TEAM:
            raise BuyError(f"Déjà {n} joueurs possédés chez {team} (maximum {MAX_PER_TEAM}).")


def formations_for(need, n_any=0):
    """Formations compatibles avec des titulaires déjà fixés par poste (+ n_any jokers)."""
    total = sum(need.values()) + n_any
    return [fm for fm, gfc in FORMATIONS.items()
            if total <= N_STARTERS and all(need[p] <= gfc[i] for i, p in enumerate(POSITIONS))]


# ---------------------------------------------------------------- MILP

def _diagnose(slots, elig, cr, credits):
    empty = [s.text for j, s in enumerate(slots) if not elig[:, j].any()]
    if empty:
        return "Aucun candidat pour : " + ", ".join(f"« {t} »" for t in empty) + \
               " (poste, fourchette de prix, --min-gp ou joueurs exclus/possédés)."
    floor = sum(cr[elig[:, j]].min() for j in range(len(slots)))
    if floor > credits + EPS:
        return (f"Budget insuffisant : les slots coûtent au moins {floor:.1f} crédits "
                f"(prix minimum de chaque fourchette) pour {credits:.1f} disponibles.")
    return ("Aucune combinaison ne respecte à la fois le budget, la limite de "
            f"{MAX_PER_TEAM} joueurs par équipe NBA, les quotas de postes et la formation des titulaires.")


def solve_buy(pool, slots, credits, owned=None, captain=False, alts=3):
    """Meilleures combinaisons pour remplir les slots.

    pool    : candidats (colonnes name, team_code, position, cr, proj ; gp facultatif), déjà
              privés des joueurs possédés ou exclus.
    owned   : joueurs possédés (name, team_code, position, role facultatif parmi T/R/6/None) ;
              ils comptent pour la limite par équipe, les quotas de postes et de rôles.
    captain : autorise un capitaine parmi les slots T (+0,5 x proj).
    Retourne une liste d'Option (meilleure d'abord) ; lève BuyError si rien n'est possible.
    """
    owned = _owned_frame(owned)
    check_request(slots, owned)
    pool = pool.reset_index(drop=True)
    cr_all = pool.cr.to_numpy(float)
    elig_all = np.array([[s.accepts(p, c) for s in slots]
                         for p, c in zip(pool.position, cr_all)], dtype=bool).reshape(len(pool), len(slots))
    keep = elig_all.any(axis=1)
    if not keep.any():
        raise BuyError(_diagnose(slots, elig_all, cr_all, credits))
    pool, elig = pool[keep].reset_index(drop=True), elig_all[keep]

    n, m = elig.shape
    pos = pool.position.to_numpy()
    teams = pool.team_code.to_numpy()
    cr = pool.cr.to_numpy(float)
    pr = pool.proj.to_numpy(float)
    w = np.array([ROLE_WEIGHT[s.role] for s in slots])
    t_slots = [j for j, s in enumerate(slots) if s.role == "T"]
    use_cap = captain and bool(t_slots)

    idx = np.arange(n * m).reshape(n, m)                  # x[i, j] : joueur i dans le slot j
    K = n * m                                             # k[i] : capitaine
    Y = K + (n if use_cap else 0)                         # y[f] : formation retenue
    N = Y + len(FORMATIONS)

    c = np.zeros(N)
    c[:K] = -(pr[:, None] * w[None, :]).ravel()           # on minimise l'opposé
    ub = np.zeros(N)
    ub[:K] = elig.ravel()                                 # éligibilité via les bornes
    if use_cap:
        c[K:Y] = -CAPTAIN_BONUS * pr
        ub[K:Y] = elig[:, t_slots].any(axis=1)
    ub[Y:] = 1

    rows, cols, vals, lo, hi = [], [], [], [], []

    def add(cc, vv, l, u):
        cc = np.asarray(cc).ravel()
        rows.extend([len(lo)] * len(cc))
        cols.extend(cc)
        vals.extend(np.broadcast_to(vv, cc.shape))
        lo.append(l)
        hi.append(u)

    for j in range(m):
        add(idx[:, j], 1.0, 1, 1)                         # chaque slot rempli une fois
    if m > 1:
        for i in range(n):
            add(idx[i], 1.0, -np.inf, 1)                  # un joueur, un slot
    add(idx, np.repeat(cr, m), -np.inf, credits + EPS)   # budget de la demande
    own_team = owned.team_code.value_counts()
    for t in np.unique(teams):
        add(idx[teams == t], 1.0, -np.inf, MAX_PER_TEAM - int(own_team.get(t, 0)))
    roles = owned.role.fillna("")
    fm_list = list(FORMATIONS.values())
    add(np.arange(Y, N), 1.0, 1, 1)                      # une formation
    for k, p in enumerate(POSITIONS):
        add(idx[pos == p], 1.0, -np.inf, SQUAD_QUOTAS[p] - int((owned.position == p).sum()))
        # titulaires du poste p (demandés + possédés T) <= quota de la formation choisie
        own_t = int(((roles == "T") & (owned.position == p)).sum())
        start_cols = idx[pos == p][:, t_slots].ravel()
        add(np.concatenate([start_cols, np.arange(Y, N)]),
            np.concatenate([np.ones(len(start_cols)), [-f[k] for f in fm_list]]), -np.inf, -own_t)
    if use_cap:
        add(np.arange(K, Y), 1.0, -np.inf, 1)             # au plus un capitaine
        for i in np.flatnonzero(ub[K:Y]):
            add(np.concatenate([[K + i], idx[i, t_slots]]),
                np.concatenate([[1.0], -np.ones(len(t_slots))]), -np.inf, 0)  # capitaine => slot T

    options = []
    while len(options) < max(alts, 1):
        A = csr_array((vals, (rows, cols)), shape=(len(lo), N))
        res = milp(c=c, constraints=LinearConstraint(A, lo, hi),
                   integrality=np.ones(N), bounds=Bounds(0, ub))
        if not res.success:
            if not options:
                raise BuyError(_diagnose(slots, elig, cr, credits))
            break
        z = np.round(res.x).astype(int)
        chosen = [int(np.argmax(z[idx[:, j]])) for j in range(m)]
        cap = int(np.argmax(z[K:Y])) if use_cap and z[K:Y].sum() else None
        table = pool.iloc[chosen].copy()
        table.insert(0, "slot", [s.text for s in slots])
        table.insert(1, "role", [s.role for s in slots])
        table["captain"] = [i == cap and s.role == "T" for i, s in zip(chosen, slots)]
        table["pts"] = table.proj.to_numpy() * (w + CAPTAIN_BONUS * table.captain.to_numpy())
        options.append(Option(float(table.pts.sum()), float(table.cr.sum()),
                              table.reset_index(drop=True)))
        add(idx[chosen], 1.0, -np.inf, len(chosen) - 1)   # coupe no-good sur l'ensemble des joueurs
    return options


# ---------------------------------------------------------------- CLI

def format_option(k, opt, credits, gp_label="mj"):
    t = opt.table
    out = pd.DataFrame({
        "slot": t.slot,
        "rôle": [ROLE_LABEL[r] + (" (CAP)" if cap else "") for r, cap in zip(t.role, t.captain)],
        "poste": t.position,
        "nom": [f"{n} {uncertain_mark(s)}".strip() for n, s in zip(t.name, t.status)] if "status" in t
        else t.name,
        **({"statut": t.status, "p_play": t.p_play.map(lambda v: f"{v:.2f}"),
            "motif": t.reason.map(short_reason)} if "status" in t else {}),
        "équipe": t.team_code,
        **({"match": t.match} if "match" in t else {}),
        "prix": t.cr.map(lambda v: f"{v:.1f}"),
        gp_label: t.gp.astype(int) if "gp" in t else "",
        "proj": t.proj.map(lambda v: f"{v:.1f}"),
        "pts": t.pts.map(lambda v: f"{v:.1f}"),
    })
    head = f"Option {k} : {opt.value:.1f} pts attendus pondérés | coût {opt.cost:.1f} / {credits:.1f}"
    body = "\n".join("  " + line for line in out.to_string(index=False).splitlines())
    return f"{head}\n{body}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Demande précise : quels joueurs acheter pour des slots donnés.")
    ap.add_argument("--buy", help='slots "[rôle:]poste[@min-max],..." ex. "T:G@8-12,R:F@-5,6:*"')
    ap.add_argument("--credits", type=float, help="budget total pour l'ensemble des slots demandés")
    ap.add_argument("--owned", default="", help='joueurs déjà dans ton équipe : "Nom A,Nom B"')
    ap.add_argument("--owned-roles", default="",
                    help='rôle des joueurs possédés : "Nom A:T,Nom B:R" (un nom cité ici est considéré possédé)')
    ap.add_argument("--exclude", default="", help='joueurs pris par des amis, jamais proposés : "Nom C"')
    ap.add_argument("--captain", action="store_true", help="autorise un capitaine (x1,5) parmi les slots T")
    ap.add_argument("--min-gp", type=int, default=1,
                    help="matchs joués minimum, comptés selon --proj-source (défaut 1)")
    ap.add_argument("--alts", type=int, default=3, help="nombre de combinaisons proposées (défaut 3)")
    ap.add_argument("--snapshot", default=None, help="CSV de prix (défaut : le plus récent)")
    ap.add_argument("--find", default=None, help="cherche un joueur par nom (sous-chaîne)")
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
    a = ap.parse_args(argv)

    df, path = load_snapshot(a.snapshot)
    print(f"Snapshot : {path} | projections : source {a.proj_source} | blessures : {a.injuries}")
    warnings.formatwarning = lambda msg, *args, **kw: f"Attention : {msg}\n"
    report = (lambda: get_report()[1]) if a.injuries == "on" else None
    try:
        dates = parse_dates(a.dates) if a.dates else None
        schedule = fetch_schedule() if dates else None
        # Règle Dunkest : seul le premier match de chaque équipe dans la fenêtre compte.
        fg = first_game_dates(schedule, dates) if dates else None
        d, gp_label = build_projections(df, a.proj_source, a.logs, injuries=report, dates=dates,
                                        first_games=fg, unknown_p=a.unknown_p)
        labels = match_labels(schedule, dates) if dates else None
    except LogsNotFoundError as e:
        print(f"Erreur : {e}", file=sys.stderr)
        return 1
    except InjuryReportError as e:
        print(f"Erreur : rapport des blessures introuvable : {e}\n"
              f"Relance avec --injuries off pour ignorer les blessures.", file=sys.stderr)
        return 1
    except ScheduleError as e:
        print(f"Erreur : {e}\nRelance sans --dates pour ignorer le calendrier.", file=sys.stderr)
        return 1
    if dates:
        print(f"Dates : {', '.join(map(str, dates))} | {len(labels)} équipes jouent")

    if a.find:
        hit = df[df.name.map(norm).str.contains(norm(a.find), regex=False)]
        hit = hit.rename(columns={"gp": "mj Dunkest"})
        cols = ["id", "proj"] if a.proj_source == "dunkest" else ["id", "gp", "proj"]
        hit = hit.merge(d[cols].rename(columns={"gp": gp_label}), on="id", how="left")
        if hit.empty:
            print(f"Aucun joueur ne correspond à « {a.find} ».")
        else:
            show = ["name", "team_code", "position", "cr", "mj Dunkest", "min"]
            show += ([gp_label] if gp_label != "mj Dunkest" else [])
            if "status" in d:
                hit = hit.merge(d[["id", "status", "p_play", "reason"]], on="id", how="left")
                hit["motif"] = hit.reason.map(short_reason, na_action="ignore")
                show += ["status", "p_play", "motif"]
            show += ["proj"]
            print(hit[show].round(2).to_string(index=False))
        if not a.buy:
            return 0

    if not a.buy:
        ap.error("--buy (ou --find) est requis")
    if a.credits is None:
        ap.error("--credits est requis avec --buy")

    try:
        slots = parse_buy(a.buy)
        owned_roles = parse_owned_roles(a.owned_roles)
        owned_names = split_names(a.owned)
        owned_names += [n for n in owned_roles if n not in {norm(x) for x in owned_names}]
        owned, missing = find_players(df, owned_names)
        if missing:
            raise BuyError(f"Joueur(s) possédé(s) introuvable(s) dans le snapshot : {', '.join(missing)}. "
                           f"Vérifie l'orthographe avec --find.")
        owned = owned.assign(role=owned.name.map(norm).map(owned_roles))
        excl, missing_ex = find_players(df, split_names(a.exclude))
        for nm in missing_ex:
            print(f"Attention : joueur exclu introuvable, ignoré : {nm}", file=sys.stderr)

        avail = filter_available(d, a.min_p_play)          # Out toujours exclus si --injuries on
        if dates:
            avail = attach_matches(avail, labels)          # équipes sans match aux dates exclues
        pool = avail[(avail.gp >= a.min_gp) & ~avail.id.isin(owned.id) & ~avail.id.isin(excl.id)]
        print(f"Demande : {', '.join(s.text for s in slots)} | budget {a.credits:.1f} | "
              f"capitaine : {'oui' if a.captain else 'non'} | {len(pool)} candidats ({gp_label} >= {a.min_gp})")
        if len(owned):
            print("Possédés : " + ", ".join(
                f"{r['name']} ({r.team_code}, {r.position}"
                + (f", {ROLE_LABEL[r.role]}" if isinstance(r.role, str) else "") + ")"
                for _, r in owned.iterrows()))
        options = solve_buy(pool, slots, a.credits, owned, captain=a.captain, alts=a.alts)
    except BuyError as e:
        print(f"Demande impossible : {e}", file=sys.stderr)
        return 1

    for k, opt in enumerate(options, 1):
        print()
        print(format_option(k, opt, a.credits, gp_label))
    if len(options) < a.alts:
        print(f"\nSeulement {len(options)} combinaison(s) possible(s).")
    if "status" in d:
        counts = [f"option {k} : {int(o.table.status.map(is_uncertain).sum())} / {len(o.table)}"
                  for k, o in enumerate(options, 1)]
        print(f"\nJoueurs proposés au statut inconnu (?) : {' | '.join(counts)}"
              f" (p_play = {a.unknown_p} pour ces joueurs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
