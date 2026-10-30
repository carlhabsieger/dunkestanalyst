"""Récupère les logs joueurs (nba_api) et calcule le score Dunkest."""
import sys
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from src.scoring import BoxScore, dunkest_score

OUT = "data/player_logs.csv"


def fetch_player_logs(season="2026-27", season_type="Regular Season"):
    res = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="P",
        timeout=60,
    )
    return res.get_data_frames()[0]


def add_dunkest(df: pd.DataFrame) -> pd.DataFrame:
    def score(r):
        return dunkest_score(BoxScore(
            pts=int(r.PTS), fgm=int(r.FGM), fga=int(r.FGA),
            ftm=int(r.FTM), fta=int(r.FTA), fg3m=int(r.FG3M),
            oreb=int(r.OREB), dreb=int(r.DREB), ast=int(r.AST),
            stl=int(r.STL), blk=int(r.BLK), tov=int(r.TOV),
            team_won=(r.WL == "W"),
        ))
    df = df.copy()
    df["DUNKEST"] = df.apply(score, axis=1)
    return df


if __name__ == "__main__":
    df = add_dunkest(fetch_player_logs())
    df.to_csv(OUT, index=False)
    print(f"{len(df)} lignes enregistrées dans {OUT}")
    if len(sys.argv) > 1:
        name = sys.argv[1].lower()
        sub = df[df.PLAYER_NAME.str.lower().str.contains(name)]
        cols = ["GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "FGA", "FGM",
                "FTA", "FTM", "FG3M", "OREB", "DREB", "AST", "STL",
                "BLK", "TOV", "DUNKEST"]
        print(sub.sort_values("GAME_DATE").tail(5)[cols].to_string(index=False))
