"""Dernier rapport officiel des blessures NBA (package nbainjuries, PDF lu par tabula : Java requis).

    python -m src.data.injuries      # récupère, sauvegarde data/injuries_latest.csv, affiche les statuts

Les timestamps du package sont naïfs et en heure de l'Est (America/New_York).
"""
import os
import shutil
import sys
import time
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")
PARIS = ZoneInfo("Europe/Paris")
REPORT_PATH = "data/injuries_latest.csv"
MAX_TRIES = 36
PAUSE_S = 0.5
JAVA_HINT = "Java est requis pour lire le PDF : installe-le avec `sudo apt install default-jre`."
COLS = ["Game Date", "Game Time", "Matchup", "Team", "Player Name", "Current Status", "Reason"]


class InjuryReportError(RuntimeError):
    """Rapport indisponible (réseau, Java, aucun rapport) : message lisible, sans traceback."""


def to_dunkest_name(s):
    """ "Jackson Jr., Jaren" -> "Jaren Jackson Jr." : on coupe à la première virgule (nom, prénom)."""
    s = " ".join(str(s).split())
    last, sep, first = s.partition(",")
    return f"{first.strip()} {last.strip()}".strip() if sep else s


def _candidate_times(now_et, tries):
    """Heures xx:30 (publication habituelle) en remontant depuis now_et, naïves en ET."""
    t = now_et.replace(minute=30, second=0, microsecond=0)
    if t > now_et:
        t -= timedelta(hours=1)
    return [t - timedelta(hours=h) for h in range(tries)]


def _network_ok(url):
    import requests
    try:
        requests.head(url, timeout=10)
        return True
    except requests.exceptions.RequestException:
        return False


def fetch_latest_report(now=None, tries=MAX_TRIES, pause=PAUSE_S, check=None, get=None, sleep=time.sleep,
                        network_ok=None):
    """Remonte d'heure en heure (ET) jusqu'à un rapport valide. Retourne (heure ET naïve, DataFrame).

    check / get / sleep / network_ok sont injectables pour les tests (aucun réseau). Lève InjuryReportError.
    """
    if check is None or get is None:
        if shutil.which("java") is None:
            raise InjuryReportError("Java introuvable. " + JAVA_HINT)
        try:
            from nbainjuries import injury
        except ImportError as e:
            raise InjuryReportError(f"Package nbainjuries absent ({e}) : `pip install -r requirements.txt`.") from None
        check = check or injury.check_reportvalid
        get = get or (lambda t: injury.get_reportdata(t, return_df=True))
    now_et = (now or datetime.now(ET)).astimezone(ET).replace(tzinfo=None)
    times = _candidate_times(now_et, tries)
    for i, t in enumerate(times):
        if i:
            sleep(pause)
        if check(t):
            try:
                df = get(t)
            except Exception as e:     # tabula (Java), réseau, PDF illisible : tout devient un message
                hint = JAVA_HINT if "java" in f"{type(e).__name__} {e}".lower() else ""
                raise InjuryReportError(f"Rapport du {t:%Y-%m-%d %H:%M} ET trouvé mais illisible : "
                                        f"{type(e).__name__}: {e}. {hint}".strip()) from None
            return t, clean_report(df)
    if network_ok is None:
        from nbainjuries.injury import gen_url
        network_ok = lambda: _network_ok(gen_url(times[0]))
    if not network_ok():
        raise InjuryReportError("Réseau indisponible : impossible de joindre le site de la NBA.")
    raise InjuryReportError(f"Aucun rapport valide sur les {tries} dernières heures "
                            f"(depuis {times[-1]:%Y-%m-%d %H:%M} ET).")


def team_code(full_name):
    """ "Boston Celtics" -> "BOS" (table statique nba_api, sans réseau) ; None si inconnu."""
    global _TEAM_CODES
    if _TEAM_CODES is None:
        from nba_api.stats.static import teams
        _TEAM_CODES = {t["full_name"]: t["abbreviation"] for t in teams.get_teams()}
        _TEAM_CODES.update(TEAM_NAME_EXTRA)
    return _TEAM_CODES.get(" ".join(str(full_name).split()))


_TEAM_CODES = None
TEAM_NAME_EXTRA = {"LA Clippers": "LAC"}   # le rapport écrit « LA Clippers », nba_api « Los Angeles Clippers »


def clean_report(df):
    """Colonnes attendues, nom Dunkest, code équipe et date de match ajoutés.

    Les lignes « NOT YET SUBMITTED » (équipe sans joueur) sont gardées avec not_submitted = True.
    Remarque : la colonne Game Time de la source n'a pas d'AM/PM (« 07:30 (ET) » pour 19 h 30) ;
    elle n'est pas utilisée, l'heure vient du calendrier (src/data/schedule.py).
    """
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise InjuryReportError(f"Format de rapport inattendu, colonnes absentes : {missing}.")
    out = df[COLS].copy()
    out["not_submitted"] = out.Reason.astype(str).str.upper().str.contains("NOT YET SUBMITTED")
    out = out[out["Player Name"].notna() | out.not_submitted].copy()
    out["Current Status"] = out["Current Status"].where(out["Current Status"].isna(),
                                                        out["Current Status"].astype(str).str.strip())
    out["dunkest_name"] = out["Player Name"].map(to_dunkest_name, na_action="ignore")
    out["team_code"] = out.Team.map(team_code)
    out["game_date"] = pd.to_datetime(out["Game Date"], format="%m/%d/%Y", errors="coerce").dt.date
    unknown = sorted(out.loc[out.team_code.isna(), "Team"].astype(str).unique())
    if unknown:
        warnings.warn(f"Équipe(s) du rapport sans code connu : {', '.join(unknown)}", stacklevel=2)
    return out.reset_index(drop=True)


def players_rows(report):
    """Lignes joueurs seulement (sans les « NOT YET SUBMITTED »)."""
    return report[~report.not_submitted]


def restrict_dates(report, dates):
    """Lignes du rapport dont le match tombe à l'une des dates (toutes si dates est vide)."""
    return report if not dates else report[report.game_date.isin(set(dates))]


def report_teams(report):
    """(teams_in_report, teams_not_submitted) en codes équipe.

    Hypothèse : une équipe peut avoir soumis pour un match et pas pour le suivant ; on juge sur
    son premier match présent dans le rapport (déjà restreint aux dates demandées le cas échéant).
    """
    r = report.dropna(subset=["team_code"])
    teams_in = set(r.team_code)
    first = r.groupby("team_code").game_date.transform("min")
    first_game = r[r.game_date == first]
    teams_ns = set(first_game.loc[first_game.not_submitted, "team_code"])
    return teams_in, teams_ns


def save_report(df, t_et, path=REPORT_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.assign(report_time_et=t_et.strftime("%Y-%m-%d %H:%M")).to_csv(path, index=False)


def report_age_line(t_et, now=None):
    """ "2026-11-01 13:30 ET (19:30 Paris), il y a 2 h 05 min" """
    aware = t_et.replace(tzinfo=ET)
    age = (now or datetime.now(ET)) - aware
    mins = max(int(age.total_seconds() // 60), 0)
    return (f"{aware:%Y-%m-%d %H:%M} ET ({aware.astimezone(PARIS):%Y-%m-%d %H:%M} Paris), "
            f"il y a {mins // 60} h {mins % 60:02d} min")


def get_report(path=REPORT_PATH, verbose=True):
    """Récupère le dernier rapport en ligne et le sauvegarde. Pour les CLI."""
    t, df = fetch_latest_report()
    save_report(df, t, path)
    if verbose:
        _, ns = report_teams(df)
        print(f"Rapport blessures : {report_age_line(t)} | {len(players_rows(df))} joueurs listés, "
              f"{len(ns)} équipe(s) au premier match sans rapport soumis")
    return t, df


def main():
    try:
        t, df = get_report()
    except InjuryReportError as e:
        print(f"Erreur : {e}", file=sys.stderr)
        return 1
    print(f"Sauvegardé dans {REPORT_PATH}")
    print("\nStatuts rencontrés :")
    print(players_rows(df)["Current Status"].value_counts().to_string())
    teams_in, teams_ns = report_teams(df)
    print(f"\nÉquipes dans le rapport : {len(teams_in)} | premier match non soumis : "
          f"{', '.join(sorted(teams_ns)) or 'aucune'}")
    nys = df[df.not_submitted]
    if len(nys):
        print("Lignes NOT YET SUBMITTED par date : "
              + ", ".join(f"{d} : {n}" for d, n in nys.groupby("game_date").size().items()))
    try:                                     # rapprochement avec le dernier snapshot, s'il existe
        from src.model.projections import injury_match_report, match_injuries
        from src.optimizer import load_snapshot
        snap, path = load_snapshot()
    except (IndexError, FileNotFoundError):
        return 0
    _, unmatched = match_injuries(snap, df)
    print(f"\nRapprochement avec {path} :")
    print(injury_match_report(unmatched, len(players_rows(df)), top=None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
