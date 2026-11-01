from datetime import date, datetime

import pandas as pd
import pytest

from src.data import injuries as inj
from src.data.injuries import (ET, InjuryReportError, clean_report, fetch_latest_report,
                               report_age_line, report_teams, team_code, to_dunkest_name)
from src.data.schedule import ScheduleError, fetch_schedule, match_labels, parse_dates
from src.model.injury_probs import NOT_LISTED, NOT_SUBMITTED, OUT_OF_REPORT, UnknownStatusWarning
from src.model.projections import apply_injuries, attach_matches, filter_available, uncertain_mark


def test_to_dunkest_name():
    assert to_dunkest_name("Antetokounmpo, Giannis") == "Giannis Antetokounmpo"
    assert to_dunkest_name("Jackson Jr., Jaren") == "Jaren Jackson Jr."
    assert to_dunkest_name("  Gilgeous-Alexander,  Shai ") == "Shai Gilgeous-Alexander"
    assert to_dunkest_name("Nikola Jokic") == "Nikola Jokic"                 # sans virgule : inchangé
    assert to_dunkest_name("Doe, John, Jr.") == "John, Jr. Doe"              # règle : 1re virgule seulement


def report(rows, team="Boston Celtics", date="11/01/2026", not_submitted=()):
    """Rapport synthétique au format nbainjuries ; not_submitted : [(équipe, date)] « NOT YET SUBMITTED »."""
    base = {"Game Date": date, "Game Time": "07:30 (ET)", "Matchup": "BOS@ORL", "Team": team,
            "Reason": "Injury/Illness - Test"}
    lines = [{**base, "Player Name": n, "Current Status": s} for n, s in rows]
    lines += [{**base, "Team": t, "Game Date": dt, "Player Name": None, "Current Status": None,
               "Reason": "NOT YET SUBMITTED"} for t, dt in not_submitted]
    return clean_report(pd.DataFrame(lines))


def players():
    names = ["Jaren Jackson Jr.", "Paolo Banchero", "Jalen Suggs", "Bobby Portis", "Joueur Sain", "Mystere Statut"]
    return pd.DataFrame({"id": range(len(names)), "name": names, "team_code": "BOS",
                         "proj": [30.0, 40.0, 20.0, 16.0, 25.0, 10.0]})


def test_projection_selon_statut():
    rep = report([("Jackson Jr., Jaren", "Questionable"), ("Banchero, Paolo", "Out"),
                  ("Suggs, Jalen", "Probable"), ("Portis Jr., Bobby", "Doubtful"),     # alias Dunkest
                  ("Inconnu, Absent", "Out")])
    d, unmatched = apply_injuries(players().drop(index=5), rep)
    d = d.set_index("name")
    assert d.loc["Jaren Jackson Jr.", "p_play"] == 0.5
    assert d.loc["Jaren Jackson Jr.", "proj"] == pytest.approx(0.5 * d.loc["Jaren Jackson Jr.", "proj_if_play"])
    assert d.loc["Jaren Jackson Jr.", "proj_if_play"] == 30.0
    assert d.loc["Paolo Banchero", "proj"] == 0.0
    assert d.loc["Bobby Portis", "status"] == "Doubtful" and d.loc["Bobby Portis", "p_play"] == 0.15
    assert d.loc["Joueur Sain", "status"] == NOT_LISTED and d.loc["Joueur Sain", "p_play"] == 1.0
    assert d.loc["Joueur Sain", "proj"] == 25.0
    assert list(unmatched["Player Name"]) == ["Inconnu, Absent"]


def test_out_exclu_et_seuil():
    rep = report([("Jackson Jr., Jaren", "Questionable"), ("Banchero, Paolo", "Out"), ("Suggs, Jalen", "Probable")])
    d, _ = apply_injuries(players().drop(index=5), rep)
    assert "Paolo Banchero" not in set(filter_available(d).name)
    assert len(filter_available(d)) == len(d) - 1
    kept = set(filter_available(d, min_p_play=0.6).name)
    assert "Jaren Jackson Jr." not in kept and "Jalen Suggs" in kept and "Joueur Sain" in kept


def test_statut_inconnu_avertissement():
    rep = report([("Statut, Mystere", "Game Time Decision")])
    with pytest.warns(UnknownStatusWarning, match="Game Time Decision"):
        d, _ = apply_injuries(players(), rep)
    j = d.set_index("name").loc["Mystere Statut"]
    assert j.p_play == 0.5 and j.proj == pytest.approx(5.0)


def test_fetch_remonte_heure_par_heure_sans_reseau():
    now = datetime(2026, 11, 1, 12, 10, tzinfo=ET)
    seen, pauses = [], []
    valid = datetime(2026, 11, 1, 9, 30)
    rep = report([("Suggs, Jalen", "Out")]).drop(columns="dunkest_name")

    def check(t):
        seen.append(t)
        return t == valid

    t, df = fetch_latest_report(now=now, check=check, get=lambda t: rep, sleep=pauses.append)
    assert t == valid and seen[0] == datetime(2026, 11, 1, 11, 30) and len(seen) == 3
    assert pauses == [0.5, 0.5] and df.dunkest_name.tolist() == ["Jalen Suggs"]


def test_fetch_echecs_messages_clairs(monkeypatch):
    now = datetime(2026, 11, 1, 12, 10, tzinfo=ET)
    nope = dict(now=now, check=lambda t: False, get=None, sleep=lambda s: None)
    with pytest.raises(InjuryReportError, match="36 dernières heures"):
        fetch_latest_report(**{**nope, "get": lambda t: None}, network_ok=lambda: True)
    with pytest.raises(InjuryReportError, match="Réseau indisponible"):
        fetch_latest_report(**{**nope, "get": lambda t: None}, network_ok=lambda: False)

    def java_ko(t):
        raise RuntimeError("`java` command is not found from this Python process.")
    with pytest.raises(InjuryReportError, match="sudo apt install default-jre"):
        fetch_latest_report(now=now, check=lambda t: True, get=java_ko, sleep=lambda s: None)

    monkeypatch.setattr(inj.shutil, "which", lambda name: None)   # Java absent : arrêt avant tout appel
    with pytest.raises(InjuryReportError, match="sudo apt install default-jre"):
        fetch_latest_report(now=now)


def test_heure_et_anciennete():
    line = report_age_line(datetime(2026, 11, 1, 4, 30), now=datetime(2026, 11, 1, 6, 35, tzinfo=ET))
    assert line == "2026-11-01 04:30 ET (2026-11-01 10:30 Paris), il y a 2 h 05 min"


def test_cli_rapport_introuvable_propose_injuries_off(tmp_path, monkeypatch, capsys):
    from src import buy
    snap = pd.DataFrame([{"id": i, "first_name": "Joueur", "last_name": f"J{i}", "team_code": "AAA",
                          "position": "G", "cr": 4.0 + i, "gp": 3, "pdk": 30.0, "min": 90}
                         for i in range(25)])
    logs = pd.DataFrame([{"PLAYER_ID": i, "PLAYER_NAME": f"Joueur J{i}", "GAME_ID": f"{i}-{g}",
                          "GAME_DATE": "2026-10-2" + str(g), "MIN": 30, "DUNKEST": 10.0 + i}
                         for i in range(25) for g in range(4)])
    snap.to_csv(tmp_path / "prices.csv", index=False)
    logs.to_csv(tmp_path / "logs.csv", index=False)

    def boom(*a, **k):
        raise InjuryReportError("Réseau indisponible : impossible de joindre le site de la NBA.")
    monkeypatch.setattr(buy, "get_report", boom)
    args = ["--find", "joueur", "--snapshot", str(tmp_path / "prices.csv"), "--logs", str(tmp_path / "logs.csv")]
    assert buy.main(args) == 1
    err = capsys.readouterr().err
    assert "--injuries off" in err and "Traceback" not in err
    assert buy.main(args + ["--injuries", "off"]) == 0                # sans blessures : pas de réseau


# ------------------------------------------------ statuts inconnus (équipe non soumise / hors rapport)

def trois_equipes():
    """BOS a soumis (une ligne joueur), ORL « NOT YET SUBMITTED », LAL absent du rapport."""
    rep = report([("Suggs, Jalen", "Out")], not_submitted=[("Orlando Magic", "11/01/2026")])
    d = pd.DataFrame({"id": [1, 2, 3], "name": ["Joueur Bos", "Joueur Orl", "Joueur Lal"],
                      "team_code": ["BOS", "ORL", "LAL"], "proj": [20.0, 20.0, 20.0]})
    return rep, d


def test_report_teams():
    rep, _ = trois_equipes()
    teams_in, teams_ns = report_teams(rep)
    assert teams_in == {"BOS", "ORL"} and teams_ns == {"ORL"}
    assert team_code("LA Clippers") == "LAC" and team_code("Boston Celtics") == "BOS"


def test_trois_statuts_sans_ligne():
    rep, d = trois_equipes()
    out, _ = apply_injuries(d, rep, unknown_p=0.7)
    s = out.set_index("name")
    assert s.loc["Joueur Bos", "status"] == NOT_LISTED and s.loc["Joueur Bos", "p_play"] == 1.0
    assert s.loc["Joueur Orl", "status"] == NOT_SUBMITTED and s.loc["Joueur Orl", "p_play"] == 0.7
    assert s.loc["Joueur Lal", "status"] == OUT_OF_REPORT and s.loc["Joueur Lal", "p_play"] == 0.7
    assert [uncertain_mark(x) for x in s.status] == ["", "?", "?"]
    defaut, _ = apply_injuries(d, rep)                                  # --unknown-p par défaut : 1.0
    assert defaut.p_play.tolist() == [1.0, 1.0, 1.0] and defaut.proj.tolist() == [20.0] * 3


def test_soumission_jugee_par_match_et_dates():
    # BOS a soumis pour le 01/11 mais pas pour le 02/11.
    rep = report([("Suggs, Jalen", "Out")], not_submitted=[("Boston Celtics", "11/02/2026")])
    d = pd.DataFrame({"id": [1], "name": ["Joueur Bos"], "team_code": ["BOS"], "proj": [20.0]})
    assert apply_injuries(d, rep)[0].status.iloc[0] == NOT_LISTED                   # 1er match : soumis
    d2 = apply_injuries(d, rep, dates=[date(2026, 11, 2)])[0]
    assert d2.status.iloc[0] == NOT_SUBMITTED
    d3 = apply_injuries(d, rep, dates=[date(2026, 11, 5)])[0]                       # date sans rapport
    assert d3.status.iloc[0] == OUT_OF_REPORT


# ------------------------------------------------ calendrier et --dates

def raw_schedule():
    return pd.DataFrame({
        "gameDate": ["11/01/2026 00:00:00", "11/02/2026 00:00:00", "11/02/2026 00:00:00"],
        "awayTeam_teamTricode": ["BOS", "BOS", "MIA"], "homeTeam_teamTricode": ["ORL", "HOU", "LAL"],
        "gameStatusText": ["3:30 pm ET", "8:30 pm ET", "10:00 pm ET"]})


def test_parse_dates():
    assert parse_dates("2026-11-02, 2026-11-01,2026-11-02") == [date(2026, 11, 1), date(2026, 11, 2)]
    for bad in ("2026-13-01", "01/11/2026", ""):
        with pytest.raises(ScheduleError):
            parse_dates(bad)


def test_calendrier_equipes_et_adversaires():
    s = fetch_schedule(fetch=lambda season: raw_schedule(), path=None)        # aucun réseau
    labels = match_labels(s, [date(2026, 11, 2)])
    assert labels == {"BOS": "02/11 @HOU 8:30 pm ET", "HOU": "02/11 vs BOS 8:30 pm ET",
                      "MIA": "02/11 @LAL 10:00 pm ET", "LAL": "02/11 vs MIA 10:00 pm ET"}
    both = match_labels(s, [date(2026, 11, 1), date(2026, 11, 2)])
    assert both["BOS"] == "01/11 @ORL 3:30 pm ET ; 02/11 @HOU 8:30 pm ET"
    d = pd.DataFrame({"name": ["A", "B", "C"], "team_code": ["BOS", "ORL", "LAL"]})
    kept = attach_matches(d, labels)
    assert kept.name.tolist() == ["A", "C"] and kept.match.tolist()[0] == "02/11 @HOU 8:30 pm ET"


def test_calendrier_indisponible_message():
    def ko(season):
        raise ConnectionError("pas de réseau")
    with pytest.raises(ScheduleError, match="Calendrier NBA indisponible"):
        fetch_schedule(fetch=ko, path=None)


def test_cli_dates_exclut_les_equipes_sans_match(tmp_path, monkeypatch, capsys):
    from src import buy
    teams = ["BOS", "ORL", "HOU"]
    snap = pd.DataFrame([{"id": i, "first_name": "Joueur", "last_name": f"J{i}", "team_code": teams[i % 3],
                          "position": "G", "cr": 4.0 + i, "gp": 3, "pdk": 30.0, "min": 90} for i in range(24)])
    logs = pd.DataFrame([{"PLAYER_ID": i, "PLAYER_NAME": f"Joueur J{i}", "GAME_ID": f"{i}-{g}",
                          "GAME_DATE": "2026-10-2" + str(g), "MIN": 30, "DUNKEST": 10.0 + i}
                         for i in range(24) for g in range(4)])
    snap.to_csv(tmp_path / "prices.csv", index=False)
    logs.to_csv(tmp_path / "logs.csv", index=False)
    monkeypatch.setattr(buy, "fetch_schedule", lambda: fetch_schedule(fetch=lambda s: raw_schedule(), path=None))
    args = ["--buy", "T:*,T:*", "--credits", "100", "--alts", "1", "--injuries", "off", "--dates", "2026-11-02",
            "--snapshot", str(tmp_path / "prices.csv"), "--logs", str(tmp_path / "logs.csv")]
    assert buy.main(args) == 0
    out = capsys.readouterr().out
    assert "ORL" not in out.split("Option 1")[1]                 # ORL ne joue pas le 02/11
    assert "@HOU 8:30 pm ET" in out or "vs BOS 8:30 pm ET" in out

def test_statut_anterieur_conserve_avec_dates():
    # Banchero Out pour le match du 01/11 ; rapport ORL du 02/11 pas encore soumis.
    rep = report([("Banchero, Paolo", "Out")], team="Orlando Magic",
                 not_submitted=[("Orlando Magic", "11/02/2026")])
    d = pd.DataFrame({"id": [1, 2], "name": ["Paolo Banchero", "Autre Orl"], "team_code": ["ORL", "ORL"],
                      "proj": [30.0, 10.0]})
    out = apply_injuries(d, rep, dates=[date(2026, 11, 2)])[0].set_index("name")
    assert out.loc["Paolo Banchero", "status"] == "Out" and out.loc["Paolo Banchero", "proj"] == 0.0
    assert out.loc["Paolo Banchero", "reason"].startswith("[statut du 01/11]")
    assert out.loc["Autre Orl", "status"] == NOT_SUBMITTED