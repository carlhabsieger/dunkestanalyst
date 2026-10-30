"""Un snapshot par jour de la liste des joueurs Dunkest (prix, score, stats)."""
import os
import sys
import datetime as dt
from zoneinfo import ZoneInfo
import pandas as pd
import requests

URL = "https://www.dunkest.com/api/stats/table"
CONFIG = "https://fantaking-api.dunkest.com/api/v1/leagues/1/config"
SEASON = os.getenv("DUNKEST_SEASON", "29")
KEEP = ["id", "first_name", "last_name", "team_code", "position",
        "cr", "gp", "pdk", "min", "starter"]
H = {"User-Agent": "Mozilla/5.0 (usage personnel, 1 requête/jour)",
     "X-Requested-With": "XMLHttpRequest",
     "Referer": "https://fantasy.dunkest.com/",
     "Accept": "application/json"}


def get_state():
    r = requests.get(CONFIG, headers=H, timeout=30)
    r.raise_for_status()
    d = r.json()["data"]
    return (d["current_matchday"]["number"], d["current_round"]["number"],
            d["status_id"])


def main():
    matchday, rnd, status = get_state()
    params = [("season_id", SEASON), ("mode", "dunkest"), ("stats_type", "tot"),
              ("min_cr", "4"), ("max_cr", "35"), ("sort_by", "pdk"),
              ("sort_order", "desc"), ("iframe", "yes")]
    params += [("teams[]", t) for t in range(1, 31)]
    params += [("positions[]", x) for x in (1, 2, 3)]
    params += [("weeks[]", w) for w in range(1, matchday + 1)]
    r = requests.get(URL, params=params, headers=H, timeout=30)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        raise SystemExit(f"Réponse vide (journée {matchday}) : "
                         "l'API a peut-être changé.")
    df = pd.DataFrame(rows)[KEEP]
    now = dt.datetime.now(ZoneInfo("Europe/Paris"))
    df.insert(0, "snapshot_ts", now.isoformat(timespec="minutes"))
    df.insert(1, "matchday", matchday)
    df.insert(2, "round", rnd)
    df.insert(3, "status_id", status)
    os.makedirs("snapshots", exist_ok=True)
    path = f"snapshots/prices_{now:%Y-%m-%d}.csv"
    df.to_csv(path, index=False)
    print(f"{len(df)} joueurs enregistrés dans {path} "
          f"(journée {matchday}, tour {rnd}, statut {status})")


if __name__ == "__main__":
    sys.exit(main())