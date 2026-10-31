import pandas as pd
import pytest

from src.model.projections import (LogsNotFoundError, load_logs, normalize_name,
                                   project_all_games)


def test_normalize_name():
    assert normalize_name("Nikola Jokić") == normalize_name("Nikola Jokic")
    assert normalize_name("Jaren Jackson Jr.") == normalize_name("Jaren Jackson Jr")
    assert normalize_name("Jaren Jackson Jr.") != normalize_name("Jaren Jackson")   # suffixe conservé
    assert normalize_name("  Shai   GILGEOUS-Alexander ") == "shai gilgeous alexander"
    assert normalize_name("De'Aaron Fox") == normalize_name("DeAaron Fox")


def _logs_rows(pid, name, scores, start="2026-10-20", dates=None):
    dates = dates or [str(d.date()) for d in pd.date_range(start, periods=len(scores), freq="2D")]
    return [{"PLAYER_ID": pid, "PLAYER_NAME": name, "GAME_ID": f"{pid}-{i}", "GAME_DATE": dt,
             "MIN": 30, "DUNKEST": s} for i, (s, dt) in enumerate(zip(scores, dates))]


def make_data():
    """Snapshot synthétique (25 remplisseurs à avg = 2 x prix + 4 joueurs testés) et ses logs."""
    snap, logs = [], []
    for i in range(25):
        cr = 4.0 + 0.5 * i
        snap.append({"id": i, "name": f"Joueur {i}", "team_code": "AAA", "position": "G",
                     "cr": cr, "gp": 3, "pdk": 3 * 2 * cr})
        logs += _logs_rows(i, f"Joueur {i}", [2 * cr] * 4)
    # 6 matchs NBA dont 2 sur deux nuits consécutives (même journée Dunkest) : Dunkest n'en compte que 5.
    snap.append({"id": 100, "name": "Nikola Jokić", "team_code": "BBB", "position": "C",
                 "cr": 20.0, "gp": 5, "pdk": 5 * 50.0})
    logs += _logs_rows(100, "Nikola Jokic", [50, 52, 48, 60, 30, 46],
                       dates=["2026-10-20", "2026-10-22", "2026-10-24", "2026-10-25", "2026-10-27",
                              "2026-10-29"])
    snap.append({"id": 101, "name": "Inconnu Absent", "team_code": "CCC", "position": "F",
                 "cr": 10.0, "gp": 2, "pdk": 30.0})
    # Même prix, même moyenne (bien au-dessus du prior), mais 2 matchs contre 20.
    for pid, name, n in ((102, "Court Echantillon", 2), (103, "Long Echantillon", 20)):
        snap.append({"id": pid, "name": name, "team_code": "CCC", "position": "F",
                     "cr": 8.0, "gp": n, "pdk": n * 40.0})
        logs += _logs_rows(pid, name, [40.0] * n)
    return pd.DataFrame(snap), pd.DataFrame(logs)


def test_tous_les_matchs_comptes():
    snap, logs = make_data()
    d = project_all_games(snap, logs).set_index("id")
    j = d.loc[100]
    assert j.matched and j.n == 6 and j.gp == 5
    assert j.avg == pytest.approx(sum([50, 52, 48, 60, 30, 46]) / 6)
    assert j.proj == pytest.approx((6 * j.avg + 3 * j.prior) / (6 + 3))


def test_joueur_absent_recoit_le_prior():
    snap, logs = make_data()
    j = project_all_games(snap, logs).set_index("id").loc[101]
    assert not j.matched and j.n == 0
    assert j.proj == pytest.approx(j.prior)


def test_lissage_plus_fort_avec_peu_de_matchs():
    snap, logs = make_data()
    d = project_all_games(snap, logs).set_index("id")
    short, long_ = d.loc[102], d.loc[103]
    assert short.prior == pytest.approx(long_.prior) and short.avg == long_.avg == 40
    assert short.prior < short.proj < long_.proj < 40
    # Part de l'écart au prior conservée : n / (n + k)
    assert (short.proj - short.prior) / (40 - short.prior) == pytest.approx(2 / 5)
    assert (long_.proj - long_.prior) / (40 - long_.prior) == pytest.approx(20 / 23)


def test_load_logs_dedoublonne(tmp_path):
    _, logs = make_data()
    p = tmp_path / "logs.csv"
    pd.concat([logs, logs.head(3)]).to_csv(p, index=False)
    assert len(load_logs(p)) == len(logs)


def test_logs_absents_message_clair(tmp_path, capsys):
    with pytest.raises(LogsNotFoundError, match=r"python -m src\.data\.box_scores"):
        load_logs(tmp_path / "absent.csv")

    # Les CLI affichent le message et renvoient 1, sans traceback.
    from src import buy
    from src.model import compare_proj
    snap = make_data()[0]
    snap[["first_name", "last_name"]] = snap.name.str.split(" ", n=1, expand=True)
    snap["min"] = 100
    sp = tmp_path / "prices.csv"
    snap.drop(columns="name").to_csv(sp, index=False)
    missing = str(tmp_path / "absent.csv")
    assert buy.main(["--find", "jok", "--snapshot", str(sp), "--logs", missing]) == 1
    assert compare_proj.main(["--snapshot", str(sp), "--logs", missing]) == 1
    err = capsys.readouterr().err
    assert err.count("python -m src.data.box_scores") == 2 and "Traceback" not in err
