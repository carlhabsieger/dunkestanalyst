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

from src.model.name_aliases import NAME_ALIASES

LOGS_PATH = "data/player_logs.csv"
SHRINK_K = 3   # valeur provisoire, à remplacer par les coefficients du test early_season
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


def project_all_games(snapshot_df, logs_df, k=SHRINK_K, aliases=NAME_ALIASES):
    """Ajoute n, avg, prior, proj, matched au snapshot (une ligne par joueur du snapshot).

    n = matchs NBA joués (MIN > 0, toutes journées confondues), avg = moyenne DUNKEST sur ces matchs,
    proj = (n*avg + k*prior) / (n + k). Sans correspondance de nom : n = 0, proj = prior, matched = False.
    aliases : {nom Dunkest: nom NBA}, appliqué avant normalize_name (clés comparées sans accents ni casse).
    """
    alias_by_key = {normalize_name(a): b for a, b in aliases.items()}
    played = logs_df[logs_df.MIN.fillna(0) > 0]
    agg = (played.assign(key=played.PLAYER_NAME.map(normalize_name))
           .groupby("key").agg(n=("DUNKEST", "size"), avg=("DUNKEST", "mean")))
    all_keys = set(logs_df.PLAYER_NAME.map(normalize_name))

    d = snapshot_df.copy()
    key = d.name.map(lambda s: normalize_name(alias_by_key.get(normalize_name(s), s)))
    d["matched"] = key.isin(all_keys)
    d["n"] = key.map(agg.n).fillna(0).astype(int)
    d["avg"] = key.map(agg.avg)
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


def build_projections(df, source="nba", logs_path=LOGS_PATH, verbose=True):
    """Projections pour les CLI. Retourne (d, gp_label).

    source "nba" : la colonne gp vaut n (matchs NBA), l'ancienne valeur passe dans gp_dunkest.
    Seuls les joueurs ayant au moins un match sont conservés, comme dans l'ancienne project().
    """
    if source == "dunkest":
        return project_dunkest(df), "mj Dunkest"
    if source != "nba":
        raise ValueError(f"Source de projection inconnue : {source} (attendu : {', '.join(PROJ_SOURCES)}).")
    d = project_all_games(df, load_logs(logs_path))
    if verbose:
        print(match_report(d))
    d = d.assign(gp_dunkest=d.gp, gp=d.n)
    return d[d.gp > 0].copy(), "matchs NBA"
