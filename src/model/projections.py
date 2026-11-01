"""Projections par match à partir de TOUS les matchs NBA (logs nba_api), et non du snapshot Dunkest.

Dunkest ne compte qu'un match par joueur et par journée de jeu (une journée peut durer jusqu'à 3 nuits) :
gp, pdk et les moyennes du snapshot ne couvrent donc qu'une partie des matchs. Ici on ignore cette
particularité et on projette la moyenne de score Dunkest sur l'ensemble des matchs NBA joués.
"""
import os
import re
import unicodedata

import numpy as np
import pandas as pd

from src.data.injuries import players_rows, restrict_dates, to_dunkest_name
from src.model.injury_probs import (DEFAULT_UNKNOWN_P, NOT_LISTED, NOT_SUBMITTED, OUT, OUT_OF_REPORT,
                                    is_uncertain, p_play_for)
from src.model.name_aliases import NAME_ALIASES

LOGS_PATH = "data/player_logs.csv"
SHRINK_K = 4   # calibré sur early_season : coefficient du score 0,51 sur 4 matchs, soit k voisin de 4
MIN_FIT_GAMES = 3
MIN_FIT_PLAYERS = 20
PROJ_SOURCES = ("nba", "dunkest")


class LogsNotFoundError(FileNotFoundError):
    """Fichier de logs absent : message lisible, à afficher sans traceback."""


def normalize_name(s):
    """Minuscules, sans accents ni ponctuation, espaces normalisés.

    Les suffixes (jr, sr, ii, iii, iv) sont conservés : "Jaren Jackson Jr." == "Jaren Jackson Jr".
    Les tirets deviennent des espaces, les autres signes (. ' etc.) sont supprimés.
    """
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"[-_/]", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def load_logs(path=LOGS_PATH):
    """Logs joueurs générés par src/data/box_scores.py, dédoublonnés par (PLAYER_ID, GAME_ID)."""
    if not os.path.exists(path):
        raise LogsNotFoundError(f"Logs NBA introuvables ({path}). Lance d'abord "
                                f"`python -m src.data.box_scores`, ou utilise --proj-source dunkest.")
    df = pd.read_csv(path, dtype={"GAME_ID": str}, parse_dates=["GAME_DATE"])
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")
    df["DUNKEST"] = pd.to_numeric(df["DUNKEST"], errors="coerce")
    return df.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"]).reset_index(drop=True)


def fit_prior(avg, cr, n):
    """Régression linéaire avg ~ cr sur les joueurs à n >= 3 (tous ceux qui ont joué s'ils sont < 20)."""
    played = n > 0
    base = played & (n >= MIN_FIT_GAMES)
    if base.sum() < MIN_FIT_PLAYERS:
        base = played
    if base.sum() < 2:
        raise ValueError("Pas assez de joueurs avec des matchs pour ajuster le prior prix -> score.")
    slope, inter = np.polyfit(cr[base], avg[base], 1)
    return inter + slope * cr


def dunkest_keys(names, aliases=NAME_ALIASES):
    """Clés de rapprochement des noms Dunkest : alias (nom Dunkest -> nom NBA) puis normalize_name."""
    alias_by_key = {normalize_name(a): b for a, b in aliases.items()}
    return names.map(lambda s: normalize_name(alias_by_key.get(normalize_name(s), s)))


def project_all_games(snapshot_df, logs_df, k=SHRINK_K, aliases=NAME_ALIASES):
    """Ajoute n, avg, mpg, prior, proj, matched au snapshot (une ligne par joueur du snapshot).

    n = matchs NBA joués (MIN > 0, toutes journées confondues), avg = moyenne DUNKEST sur ces matchs,
    mpg = minutes moyennes sur ces matchs (information seulement, non utilisée dans proj),
    proj = (n*avg + k*prior) / (n + k). Sans correspondance de nom : n = 0, proj = prior, matched = False.
    aliases : {nom Dunkest: nom NBA}, appliqué avant normalize_name (clés comparées sans accents ni casse).
    """
    played = logs_df[logs_df.MIN.fillna(0) > 0]
    agg = (played.assign(key=played.PLAYER_NAME.map(normalize_name))
           .groupby("key").agg(n=("DUNKEST", "size"), avg=("DUNKEST", "mean"), mpg=("MIN", "mean")))
    all_keys = set(logs_df.PLAYER_NAME.map(normalize_name))

    d = snapshot_df.copy()
    key = dunkest_keys(d.name, aliases)
    d["matched"] = key.isin(all_keys)
    d["n"] = key.map(agg.n).fillna(0).astype(int)
    d["avg"] = key.map(agg.avg)
    d["mpg"] = key.map(agg.mpg)
    d["prior"] = fit_prior(d.avg.fillna(0).to_numpy(), d.cr.to_numpy(float), d.n.to_numpy())
    d["proj"] = np.where(d.n > 0, (d.n * d.avg.fillna(0) + k * d.prior) / (d.n + k), d.prior)
    return d


def project_dunkest(df, k=SHRINK_K):
    """Ancienne logique (pour comparaison) : moyenne pdk/gp du snapshot, lissée vers un prior prix."""
    d = df[df.gp > 0].copy()
    d["avg"] = d.pdk / d.gp          # hypothèse : pdk est un cumul de saison (à vérifier)
    base = d[d.gp >= 3] if (d.gp >= 3).sum() >= 20 else d
    slope, inter = np.polyfit(base.cr, base.pdk / base.gp, 1)
    d["prior"] = inter + slope * d.cr
    d["proj"] = (d.gp * d.avg + k * d.prior) / (d.gp + k)
    return d


def match_report(d, top=15):
    """Récapitulatif du rapprochement snapshot / logs, avec les non rapprochés à corriger."""
    miss = d[~d.matched]
    lines = [f"Rapprochement snapshot / logs NBA : {int(d.matched.sum())} joueurs rapprochés, "
             f"{len(miss)} non rapprochés."]
    played = miss[miss.gp > 0].sort_values("cr", ascending=False)
    if len(played):
        lines.append(f"Non rapprochés ayant joué côté Dunkest ({len(played)}, {min(top, len(played))} "
                     f"premiers par prix) :")
        lines += [f"  {r['name']} ({r.team_code}, {r.cr:.1f} cr, gp Dunkest {int(r.gp)})"
                  for _, r in played.head(top).iterrows()]
    return "\n".join(lines)


def _team_target_dates(report, dates, first_games):
    """Date du match qui compte pour chaque équipe (règle Dunkest : premier match de la fenêtre).

    first_games (calendrier) fait foi s'il est fourni ; sinon premier match de l'équipe dans le rapport
    restreint aux dates (ou dans tout le rapport sans dates).
    """
    if first_games is not None:
        return dict(first_games)
    rep = restrict_dates(report, dates).dropna(subset=["team_code"])
    return rep.groupby("team_code").game_date.min().to_dict()


def match_injuries(d, report, aliases=NAME_ALIASES, dates=None, first_games=None):
    """Statut et motif du rapport pour chaque joueur de d, rapportés au match qui compte.

    Pour chaque équipe, le match qui compte est son premier match de la fenêtre (first_games, issu
    du calendrier) ; les lignes d'un match suivant dans la fenêtre sont ignorées. Pour un joueur :
    - ligne à la date de ce match : son statut ;
    - sinon ligne à une date antérieure : dernier statut connu (hypothèse : un Out reste Out), avec
      son âge en jours (status_age_days) et le motif préfixé « [statut du 01/11, 3 j avant] » ;
    - sinon selon son équipe à cette date : « rapport non soumis » si « NOT YET SUBMITTED »,
      « non listé » si l'équipe a des lignes joueurs, « hors rapport » sinon.
    Retourne (info, unmatched) : info a les colonnes status, reason, status_age_days alignées sur
    d.index ; unmatched = lignes joueurs du rapport (dans la fenêtre) sans joueur Dunkest.
    """
    rows = players_rows(report)
    rows = rows.assign(key=rows["Player Name"].map(to_dunkest_name).map(normalize_name))
    by_key = {k: g.sort_values("game_date", kind="stable") for k, g in rows.groupby("key")}
    ns = set(map(tuple, report.loc[report.not_submitted, ["team_code", "game_date"]].values))
    sub = set(map(tuple, rows[["team_code", "game_date"]].values))
    target = _team_target_dates(report, dates, first_games)

    keys = dunkest_keys(d.name, aliases)
    teams = d.team_code if "team_code" in d else pd.Series(None, index=d.index)
    out = []
    for k, team in zip(keys, teams):
        t = target.get(team)
        g = by_key.get(k)
        if t is None:
            out.append((OUT_OF_REPORT, "", np.nan))
            continue
        if g is not None:
            at = g[g.game_date == t]
            if len(at):
                out.append((at["Current Status"].iloc[0], at.Reason.iloc[0], np.nan))
                continue
            before = g[g.game_date < t]
            if len(before):
                r = before.iloc[-1]
                age = (t - r.game_date).days
                out.append((r["Current Status"], f"[statut du {r.game_date:%d/%m}, {age} j avant] {r.Reason}", age))
                continue
        if (team, t) in ns:
            out.append((NOT_SUBMITTED, "", np.nan))
        else:
            out.append((NOT_LISTED if (team, t) in sub else OUT_OF_REPORT, "", np.nan))
    info = pd.DataFrame(out, index=d.index, columns=["status", "reason", "status_age_days"])
    window = players_rows(restrict_dates(report, dates))
    unmatched = window[~window["Player Name"].map(to_dunkest_name).map(normalize_name).isin(set(keys))]
    return info, unmatched


def apply_injuries(d, report, aliases=NAME_ALIASES, dates=None, unknown_p=DEFAULT_UNKNOWN_P, first_games=None):
    """Ajoute status, reason, status_age_days, p_play, proj_if_play (ancienne proj) et
    proj = p_play * proj_if_play.

    Retourne (d, lignes du rapport non rapprochées). Un statut écrit mais inconnu reçoit 0,5
    (avertissement) ; « rapport non soumis » et « hors rapport » reçoivent unknown_p.
    first_games : {équipe: date du premier match de la fenêtre} (calendrier), voir match_injuries.
    """
    info, unmatched = match_injuries(d, report, aliases, dates, first_games)
    d = d.assign(status=info.status, reason=info.reason, status_age_days=info.status_age_days,
                 p_play=p_play_for(info.status, unknown_p), proj_if_play=d.proj)
    d["proj"] = d.p_play * d.proj_if_play
    return d, unmatched


def short_reason(reason, width=38):
    """Motif abrégé pour l'affichage (l'âge « [statut du 01/11, 3 j avant] » reste en tête)."""
    r = str(reason or "").replace("Injury/Illness - ", "")
    return r if len(r) <= width else r[:width - 1] + "…"


def attach_matches(d, labels):
    """Garde les joueurs dont l'équipe joue aux dates demandées et ajoute la colonne match.

    labels : {code équipe: "02/11 @HOU 8:30 pm ET"} (src.data.schedule.match_labels).
    """
    d = d[d.team_code.isin(set(labels))].copy()
    d["match"] = d.team_code.map(labels)
    return d


def uncertain_mark(status):
    """ "?" pour un statut inconnu (rapport non soumis ou hors rapport), "" sinon."""
    return "?" if is_uncertain(status) else ""


def filter_available(d, min_p_play=0.0):
    """Candidats : sans les joueurs Out (toujours) ni ceux dont p_play < min_p_play."""
    if "status" not in d:
        return d
    return d[(d.status != OUT) & (d.p_play >= min_p_play)].copy()


def injury_match_report(unmatched, n_report, top=15):
    """Joueurs du rapport sans correspondance Dunkest (équipe sans joueur Dunkest ou nom différent)."""
    lines = [f"Rapprochement rapport blessures / Dunkest : {n_report - len(unmatched)} joueurs rapprochés, "
             f"{len(unmatched)} non rapprochés (absents de Dunkest ou nom différent)."]
    shown = unmatched if top is None else unmatched.head(top)
    lines += [f"  {r['Player Name']} ({r.Team}, {r['Current Status']})" for _, r in shown.iterrows()]
    if len(shown) < len(unmatched):
        lines.append(f"  ... et {len(unmatched) - len(shown)} autres (python -m src.data.injuries pour tout voir)")
    return "\n".join(lines)


def build_projections(df, source="nba", logs_path=LOGS_PATH, verbose=True, injuries=None, dates=None, first_games=None,
                      unknown_p=DEFAULT_UNKNOWN_P):
    """Projections pour les CLI. Retourne (d, gp_label).

    source "nba" : la colonne gp vaut n (matchs NBA), l'ancienne valeur passe dans gp_dunkest.
    Seuls les joueurs ayant au moins un match sont conservés, comme dans l'ancienne project().
    injuries : DataFrame du rapport (src.data.injuries), fonction sans argument qui le renvoie
    (appelée après le chargement des logs, donc pas de réseau si les logs manquent), ou None.
    Les candidats se filtrent ensuite avec filter_available. dates restreint le rapport aux matchs
    de ces dates ; first_games ({équipe: date}, calendrier) fixe le match qui compte pour chaque
    équipe ; unknown_p est la probabilité des statuts inconnus (non soumis, hors rapport).
    """
    if source == "dunkest":
        d, gp_label = project_dunkest(df), "mj Dunkest"
    elif source == "nba":
        d = project_all_games(df, load_logs(logs_path))
        if verbose:
            print(match_report(d))
        d = d.assign(gp_dunkest=d.gp, gp=d.n)
        d, gp_label = d[d.gp > 0].copy(), "matchs NBA"
    else:
        raise ValueError(f"Source de projection inconnue : {source} (attendu : {', '.join(PROJ_SOURCES)}).")
    if callable(injuries):
        injuries = injuries()
    if injuries is not None:
        d, _ = apply_injuries(d, injuries, dates=dates, unknown_p=unknown_p, first_games=first_games)
        # Non rapprochés calculés sur tout le snapshot : un blessé sans match NBA reste « rapproché ».
        _, unmatched = match_injuries(df, injuries, dates=dates, first_games=first_games)
        if verbose:
            print(injury_match_report(unmatched, len(players_rows(restrict_dates(injuries, dates)))))
    return d, gp_label
