from src.scoring import BoxScore, dunkest_score, captain_score


def make(**kw):
    base = dict(pts=25, fgm=9, fga=20, ftm=4, fta=5, fg3m=3, oreb=2,
                dreb=6, ast=5, stl=1, blk=0, tov=3, started=True)
    base.update(kw)
    return BoxScore(**base)


def test_basic():
    assert dunkest_score(make()) == 30.0


def test_win_and_captain():
    s = dunkest_score(make(team_won=True))
    assert round(s, 2) == 31.5
    assert round(captain_score(dunkest_score(make())), 2) == 45.0