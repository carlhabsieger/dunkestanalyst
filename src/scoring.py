"""Score Dunkest d'un joueur pour un match.
Règles : https://docs.dunkest.com/fantasy-rules/player-scoring
"""
from dataclasses import dataclass

# Hypothèse (non précisée dans la doc) : un triple-double remplace le
# double-double (+10 au lieu de +5+10). Passe à True pour cumuler.
CUMULATIVE_DOUBLES = False


@dataclass
class BoxScore:
    pts: int
    fgm: int
    fga: int
    ftm: int
    fta: int
    fg3m: int
    oreb: int
    dreb: int
    ast: int
    stl: int
    blk: int
    tov: int
    blka: int = 0          # tirs contrés subis (absent du game log de base)
    started: bool = False  # dans le cinq majeur
    team_won: bool = False


def _three_point_bonus(fg3m: int) -> float:
    if fg3m >= 5:
        return 5.0
    return {3: 3.0, 4: 4.0}.get(fg3m, 0.0)


def _doubles_bonus(b: BoxScore) -> float:
    cats = [b.pts, b.oreb + b.dreb, b.ast, b.stl, b.blk]
    n = sum(1 for c in cats if c >= 10)
    tiers = [(2, 5.0), (3, 10.0), (4, 50.0)]
    if CUMULATIVE_DOUBLES:
        return sum(v for k, v in tiers if n >= k)
    return max((v for k, v in tiers if n >= k), default=0.0)


def dunkest_score(b: BoxScore) -> float:
    score = (
        b.pts
        + b.dreb * 1.0
        + b.oreb * 1.25
        + b.ast * 1.5
        + b.stl * 1.5
        - b.tov * 1.5
        + b.blk * 1.5
        - b.blka * 0.5
        - (b.fga - b.fgm)   # tirs ratés (hypothèse : 2 pts et 3 pts)
        - (b.fta - b.ftm)   # lancers francs ratés
        + _three_point_bonus(b.fg3m)
        + _doubles_bonus(b)
        + (1.0 if b.started else 0.0)
    )
    if b.team_won:
        score *= 1.05       # hypothèse : 5 % du score total
    return score


def captain_score(score: float) -> float:
    return score * 1.5
    