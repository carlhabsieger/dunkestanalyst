"""Calendrier NBA (nba_api ScheduleLeagueV2) : quelles équipes jouent à une date donnée.

    python -m src.data.schedule 2026-11-01,2026-11-02

Source testée le 01/11/2026 : ScheduleLeagueV2(season="2026-27") renvoie la saison entière
(1274 matchs) en un appel, avec gameDate (date US), les codes équipes et gameStatusText
(« 7:30 pm ET »). ScoreboardV3 fonctionne aussi, mais date par date ; ScoreboardV2 échoue (réponse
vide, endpoint signalé comme défectueux par nba_api). Attention : gameDateTimeEst porte un suffixe
« Z » alors que l'heure est en ET ; on ne s'en sert pas.
"""
import os
import sys
from datetime import date, datetime

import pandas as pd

SEASON = "2026-27"
SCHEDULE_PATH = f"data/schedule_{SEASON}.csv"


class ScheduleError(RuntimeError):
    """Calendrier indisponible ou dates invalides : message lisible, sans traceback."""


def parse_dates(s):
    """ "2026-11-01,2026-11-02" -> [date(2026, 11, 1), date(2026, 11, 2)] """
    out = []
    for item in (s or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(datetime.strptime(item, "%Y-%m-%d").date())
        except ValueError:
            raise ScheduleError(f"Date invalide « {item} » (format attendu AAAA-MM-JJ).") from None
    if not out:
        raise ScheduleError("--dates est vide (format attendu : 2026-11-01,2026-11-02).")
    return sorted(set(out))


def _fetch_raw(season):
    from nba_api.stats.endpoints import scheduleleaguev2
    return scheduleleaguev2.ScheduleLeagueV2(league_id="00", season=season, timeout=60).get_data_frames()[0]


def clean_schedule(raw):
    """Une ligne par match : game_date (date US), away, home, time_et."""
    need = ["gameDate", "awayTeam_teamTricode", "homeTeam_teamTricode", "gameStatusText"]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise ScheduleError(f"Format de calendrier inattendu, colonnes absentes : {missing}.")
    s = pd.DataFrame({
        "game_date": pd.to_datetime(raw.gameDate, format="%m/%d/%Y %H:%M:%S", errors="coerce").dt.date,
        "away": raw.awayTeam_teamTricode, "home": raw.homeTeam_teamTricode,
        "time_et": raw.gameStatusText.astype(str).str.strip(),
    })
    return s.dropna(subset=["game_date", "away", "home"]).query("away != '' and home != ''").reset_index(drop=True)


def fetch_schedule(season=SEASON, path=SCHEDULE_PATH, fetch=None):
    """Calendrier de la saison, sauvegardé dans data/. fetch est injectable pour les tests."""
    try:
        raw = (fetch or _fetch_raw)(season)
    except ScheduleError:
        raise
    except Exception as e:
        raise ScheduleError(f"Calendrier NBA indisponible ({type(e).__name__}: {e}).") from None
    s = clean_schedule(raw)
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        s.to_csv(path, index=False)
    return s


def games_on(schedule, dates):
    """Une ligne par (équipe, match) aux dates demandées : team_code, game_date, opponent, label."""
    g = schedule[schedule.game_date.isin(set(dates))]
    home = pd.DataFrame({"team_code": g.home, "game_date": g.game_date, "opponent": g.away,
                         "label": [f"{d:%d/%m} vs {o} {t}" for d, o, t in zip(g.game_date, g.away, g.time_et)]})
    away = pd.DataFrame({"team_code": g.away, "game_date": g.game_date, "opponent": g.home,
                         "label": [f"{d:%d/%m} @{o} {t}" for d, o, t in zip(g.game_date, g.home, g.time_et)]})
    return pd.concat([home, away]).sort_values(["team_code", "game_date"]).reset_index(drop=True)


def first_games(schedule, dates):
    """Premier match de chaque équipe dans la fenêtre (règle Dunkest : seul celui-ci compte).

    Une ligne par équipe : team_code, game_date, opponent, label, n_games (matchs dans la fenêtre).
    """
    g = games_on(schedule, dates)
    n = g.groupby("team_code").size().rename("n_games")
    return g.drop_duplicates("team_code", keep="first").join(n, on="team_code").reset_index(drop=True)


def first_game_dates(schedule, dates):
    """{code équipe: date de son premier match dans la fenêtre}"""
    return dict(zip(*first_games(schedule, dates)[["team_code", "game_date"]].T.values))


def match_labels(schedule, dates):
    """{code équipe: "02/11 @HOU 8:30 pm ET"} : premier match de la fenêtre seulement.

    Si l'équipe joue encore ensuite dans la fenêtre : « (+1 match non compté) ».
    """
    f = first_games(schedule, dates)
    extra = [f" (+{k - 1} match{'s' if k > 2 else ''} non compté{'s' if k > 2 else ''})" if k > 1 else ""
             for k in f.n_games]
    return dict(zip(f.team_code, f.label + extra))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        dates = parse_dates(argv[0] if argv else date.today().isoformat())
        s = fetch_schedule()
    except ScheduleError as e:
        print(f"Erreur : {e}", file=sys.stderr)
        return 1
    print(f"Calendrier {SEASON} : {len(s)} matchs (sauvegardé dans {SCHEDULE_PATH})")
    for d in dates:
        day = s[s.game_date == d]
        print(f"\n{d} : {len(day)} match(s)")
        for _, r in day.iterrows():
            print(f"  {r.away}@{r.home}  {r.time_et}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
