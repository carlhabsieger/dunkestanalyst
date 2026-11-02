"""Historique des rapports de blessures : archivage, statut hérité, annulation, --stale-days (sans réseau)."""
from datetime import date, datetime

import pandas as pd
import pytest

from src import team as T
from src.data import injuries as inj
from src.data.injuries import archive_report, clean_report, load_reports, read_report_csv
from src.model.injury_probs import NOT_LISTED, STALE
from src.model.projections import apply_injuries

T1 = datetime(2026, 11, 1, 4, 30)          # rapport du 01/11 : Banchero Out (match ORL du 01/11)
T2 = datetime(2026, 11, 2, 4, 30)          # rapport du 02/11
WINDOW = [date(2026, 11, 4), date(2026, 11, 5)]
FIRST = {"ORL": date(2026, 11, 5), "BOS": date(2026, 11, 4)}


def rep(rows, t, not_submitted=()):
    """Rapport nettoyé daté t ; rows : (nom rapport, statut, équipe, date de match)."""
    lines = [{"Game Date": gd, "Game Time": "07:30 (ET)", "Matchup": "BOS@ORL", "Team": tm,
              "Player Name": n, "Current Status": s, "Reason": "Injury/Illness - Ankle"} for n, s, tm, gd in rows]
    lines += [{"Game Date": gd, "Game Time": "07:30 (ET)", "Matchup": "WAS@ORL", "Team": tm, "Player Name": None,
               "Current Status": None, "Reason": "NOT YET SUBMITTED"} for tm, gd in not_submitted]
    return clean_report(pd.DataFrame(lines)).assign(report_time=pd.Timestamp(t))


def r1():
    return rep([("Banchero, Paolo", "Out", "Orlando Magic", "11/01/2026"),
                ("Brown, Jaylen", "Questionable", "Boston Celtics", "11/01/2026")], T1)


def players():
    return pd.DataFrame({"id": [1, 2, 3], "name": ["Paolo Banchero", "Jaylen Brown", "Autre Orl"],
                         "team_code": ["ORL", "BOS", "ORL"], "proj": [40.0, 30.0, 10.0]})


def status_of(reports, now, stale_days=7, unknown_p=1.0):
    d, _ = apply_injuries(players(), reports, dates=WINDOW, first_games=FIRST, now=now, stale_days=stale_days,
                          unknown_p=unknown_p)
    return d.set_index("name")


# ------------------------------------------------ archivage

def test_archivage_un_fichier_par_rapport_sans_ecrasement(tmp_path):
    root = str(tmp_path / "injuries")
    df = r1().drop(columns="report_time")
    path, created = archive_report(df, T1, root)
    assert created and path.endswith("report_20261101_0430_ET.csv")
    before = open(path, encoding="utf-8").read()
    path2, created2 = archive_report(df.assign(Reason="modifié"), T1, root)
    assert path2 == path and not created2 and open(path, encoding="utf-8").read() == before
    back = read_report_csv(path)
    assert back.game_date.iloc[0] == date(2026, 11, 1) and back.report_time.iloc[0] == pd.Timestamp(T1)
    assert back.not_submitted.dtype == bool


def test_load_reports_archives_plus_dernier(tmp_path):
    root = str(tmp_path / "injuries")
    archive_report(r1().drop(columns="report_time"), T1, root)
    latest = rep([], T2, not_submitted=[("Orlando Magic", "11/02/2026")]).drop(columns="report_time")
    archive_report(latest, T2, root)                              # le dernier est aussi archivé
    allr = load_reports(latest, T2, root)
    assert sorted(allr.report_time.unique()) == [pd.Timestamp(T1), pd.Timestamp(T2)]   # sans doublon
    assert len(allr[allr.report_time == pd.Timestamp(T2)]) == len(latest)


def test_get_report_archive_et_renvoie_tout(tmp_path, monkeypatch):
    root = str(tmp_path / "injuries")
    archive_report(r1().drop(columns="report_time"), T1, root)
    latest = rep([], T2, not_submitted=[("Orlando Magic", "11/02/2026")]).drop(columns="report_time")
    monkeypatch.setattr(inj, "fetch_latest_report", lambda: (T2, latest))
    t, allr = inj.get_report(path=str(tmp_path / "latest.csv"), verbose=False, root=root)
    assert t == T2 and allr.report_time.nunique() == 2
    assert (tmp_path / "injuries" / "report_20261102_0430_ET.csv").exists()


# ------------------------------------------------ statut hérité, annulé, ancien

def test_statut_herite_avec_age():
    # 02/11 : ORL pas encore soumis, aucune ligne Banchero -> il reste Out (statut du 01/11).
    reports = pd.concat([r1(), rep([], T2, not_submitted=[("Orlando Magic", "11/02/2026")])])
    s = status_of(reports, now=datetime(2026, 11, 2, 9, 0))
    b = s.loc["Paolo Banchero"]
    assert b.status == "Out" and b.p_play == 0.0 and b.proj == 0.0 and b.status_age_days == 1
    assert b.reason.startswith("[dernier statut connu, il y a 1 j, rapport du 01/11 04:30 ET]")
    j = s.loc["Jaylen Brown"]                                     # statut non Out hérité : garde son p_play
    assert j.status == "Questionable" and j.p_play == 0.5 and j.status_age_days == 1


def test_statut_annule_par_un_rapport_plus_recent():
    # 02/11 : ORL a soumis (une autre ligne ORL) sans lister Banchero -> il redevient « non listé ».
    reports = pd.concat([r1(), rep([("Autre, Joueur", "Out", "Orlando Magic", "11/02/2026")], T2)])
    s = status_of(reports, now=datetime(2026, 11, 2, 9, 0))
    b = s.loc["Paolo Banchero"]
    assert b.status == NOT_LISTED and b.p_play == 1.0 and b.proj == 40.0 and "annulé" in b.reason
    assert s.loc["Jaylen Brown", "status"] == "Questionable"     # BOS n'a rien soumis depuis : inchangé


def test_annulation_seulement_par_un_rapport_de_son_equipe():
    reports = pd.concat([r1(), rep([("Tatum, Jayson", "Out", "Boston Celtics", "11/02/2026")], T2)])
    s = status_of(reports, now=datetime(2026, 11, 2, 9, 0))
    assert s.loc["Paolo Banchero", "status"] == "Out"           # rapport BOS : ne dit rien d'ORL
    assert s.loc["Jaylen Brown", "status"] == NOT_LISTED         # BOS a soumis sans Brown


@pytest.mark.parametrize("stale_days, expected", [(7, STALE), (10, "Out")])
def test_stale_days(stale_days, expected):
    reports = pd.concat([r1(), rep([], T2, not_submitted=[("Orlando Magic", "11/02/2026")])])
    s = status_of(reports, now=datetime(2026, 11, 10, 9, 0), stale_days=stale_days, unknown_p=0.7)
    b = s.loc["Paolo Banchero"]
    assert b.status == expected and b.status_age_days == 9
    assert b.p_play == (0.7 if expected == STALE else 0.0)


# ------------------------------------------------ cas Banchero dans team show (capitanat)

def demo_squad():
    spec = [("Paolo Banchero", "F", "ORL", 40), ("G1", "G", "BOS", 20), ("G2", "G", "BOS", 20),
            ("G3", "G", "MIA", 20), ("G4", "G", "MIA", 5), ("F2", "F", "NYK", 20), ("F3", "F", "NYK", 5),
            ("F4", "F", "LAL", 5), ("C1", "C", "LAL", 20), ("C2", "C", "DEN", 5)]
    return pd.DataFrame([{"id": i, "name": n, "position": p, "team_code": t, "cr": 8.0, "proj": float(v), "gp": 4}
                         for i, (n, p, t, v) in enumerate(spec)])


@pytest.mark.parametrize("orl_submitted, captain_is_banchero", [(False, False), (True, True)])
def test_banchero_capitaine_selon_rapport(orl_submitted, captain_is_banchero):
    later = (rep([("Autre, Joueur", "Out", "Orlando Magic", "11/02/2026")], T2) if orl_submitted
             else rep([], T2, not_submitted=[("Orlando Magic", "11/02/2026")]))
    reports = pd.concat([r1(), later])
    squad = demo_squad()
    d, _ = apply_injuries(squad, reports, dates=WINDOW, first_games={t: date(2026, 11, 5) for t in squad.team_code},
                          now=datetime(2026, 11, 2, 9, 0))
    team = {**T.EMPTY_TEAM, "players": list(squad.name)}
    owned = T.owned_frame(team, squad.drop(columns="proj"), d)
    value, fm, lu = T.lineup(owned)
    roles = lu.set_index("name").role
    if captain_is_banchero:
        assert roles["Paolo Banchero"] == "capitaine"
    else:
        assert roles["Paolo Banchero"] == "banc"                  # Out : proj 0, ni capitaine ni titulaire
        assert any("Paolo Banchero" in w and "Out" in w for w in T.team_warnings(owned))
