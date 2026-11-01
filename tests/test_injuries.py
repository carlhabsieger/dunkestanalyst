from datetime import datetime

import pandas as pd
import pytest

from src.data import injuries as inj
from src.data.injuries import (ET, InjuryReportError, clean_report, fetch_latest_report,
                               report_age_line, to_dunkest_name)
from src.model.injury_probs import NOT_LISTED, UnknownStatusWarning
from src.model.projections import apply_injuries, filter_available


def test_to_dunkest_name():
    assert to_dunkest_name("Antetokounmpo, Giannis") == "Giannis Antetokounmpo"
    assert to_dunkest_name("Jackson Jr., Jaren") == "Jaren Jackson Jr."
    assert to_dunkest_name("  Gilgeous-Alexander,  Shai ") == "Shai Gilgeous-Alexander"
    assert to_dunkest_name("Nikola Jokic") == "Nikola Jokic"                 # sans virgule : inchangé
    assert to_dunkest_name("Doe, John, Jr.") == "John, Jr. Doe"              # règle : 1re virgule seulement


def report(rows):
    """Rapport synthétique au format nbainjuries."""
    base = {"Game Date": "11/01/2026", "Game Time": "07:30 (ET)", "Matchup": "AAA@BBB", "Team": "Team A",
            "Reason": "Injury/Illness - Test"}
    return clean_report(pd.DataFrame([{**base, "Player Name": n, "Current Status": s} for n, s in rows]))


def players():
    names = ["Jaren Jackson Jr.", "Paolo Banchero", "Jalen Suggs", "Bobby Portis", "Joueur Sain", "Mystere Statut"]
    return pd.DataFrame({"id": range(len(names)), "name": names, "proj": [30.0, 40.0, 20.0, 16.0, 25.0, 10.0]})


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
