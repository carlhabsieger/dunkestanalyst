"""Format des snapshots : nouvelles colonnes num_rounds / round_started_at, anciens fichiers toujours lisibles.

Aucun réseau : requests.get est remplacé par de fausses réponses.
"""
import runpy
from pathlib import Path

import pandas as pd
import pytest

from src.data import snapshot_prices
from src.model.projections import build_projections
from src.optimizer import load_snapshot

OLD_COLS = ["snapshot_ts", "matchday", "round", "status_id", "id", "first_name", "last_name", "team_code",
            "position", "cr", "gp", "pdk", "min", "starter"]
CONFIG = {"data": {"status_id": 1, "current_matchday": {"id": 1862, "number": 5, "num_rounds": 3},
                   "current_round": {"id": 2902, "number": 2, "started_at": "2026-11-01T20:29:00.000000Z"}}}


class FakeResp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def table_rows(n=25):
    return [{"id": i, "first_name": "Joueur", "last_name": f"J{i}", "team_code": ["BOS", "ORL", "LAL"][i % 3],
             "position": "GFC"[i % 3], "cr": 4.0 + i, "gp": 3, "pdk": 30.0 + i, "min": 90, "starter": 0,
             "extra": "ignorée"} for i in range(n)]


def fake_get(url, **kw):
    return FakeResp(CONFIG if "config" in url else table_rows())


def write_old(path, cr_shift=0.0):
    df = pd.DataFrame(table_rows()).drop(columns="extra")
    df["cr"] += cr_shift
    for i, (c, v) in enumerate([("snapshot_ts", "2026-10-31T09:38+01:00"), ("matchday", 5), ("round", 1),
                                ("status_id", 1)]):
        df.insert(i, c, v)
    df[OLD_COLS].to_csv(path, index=False)


def test_snapshot_prices_ajoute_deux_colonnes_a_la_fin(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(snapshot_prices.requests, "get", fake_get)
    snapshot_prices.main()
    (f,) = list((tmp_path / "snapshots").glob("prices_*.csv"))
    df = pd.read_csv(f)
    assert list(df.columns) == OLD_COLS + ["num_rounds", "round_started_at"]   # existantes inchangées
    assert (df.num_rounds == 3).all() and (df.round_started_at == "2026-11-01T20:29:00.000000Z").all()
    assert (df.matchday == 5).all() and (df["round"] == 2).all()


def test_champs_absents_de_la_config(tmp_path, monkeypatch):
    cfg = {"data": {"status_id": 1, "current_matchday": {"number": 5}, "current_round": {"number": 2}}}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(snapshot_prices.requests, "get",
                        lambda url, **kw: FakeResp(cfg if "config" in url else table_rows()))
    snapshot_prices.main()
    df = pd.read_csv(next((tmp_path / "snapshots").glob("prices_*.csv")))
    assert df.num_rounds.isna().all() and df.round_started_at.isna().all()


@pytest.mark.parametrize("new_format", [False, True])
def test_load_snapshot_et_projection_ancien_et_nouveau(tmp_path, monkeypatch, new_format):
    p = tmp_path / "prices.csv"
    write_old(p)
    if new_format:
        df = pd.read_csv(p).assign(num_rounds=3, round_started_at="2026-11-01T20:29:00.000000Z")
        df.to_csv(p, index=False)
    snap, _ = load_snapshot(str(p))
    assert "name" in snap and len(snap) == 25
    d, label = build_projections(snap, source="dunkest", verbose=False)
    assert len(d) == 25 and d.proj.notna().all() and label == "mj Dunkest"


def test_buy_lit_les_deux_formats(tmp_path, capsys):
    from src import buy
    for name, new in (("old.csv", False), ("new.csv", True)):
        p = tmp_path / name
        write_old(p)
        if new:
            pd.read_csv(p).assign(num_rounds=3, round_started_at="x").to_csv(p, index=False)
        args = ["--buy", "T:*", "--credits", "50", "--alts", "1", "--injuries", "off", "--proj-source",
                "dunkest", "--snapshot", str(p)]
        assert buy.main(args) == 0
    assert capsys.readouterr().out.count("Option 1") == 2


def test_snapshot_check_compare_ancien_et_nouveau(tmp_path, monkeypatch, capsys):
    """snapshot_check.py (script) : ancien fichier comparé à un nouveau, sans réseau."""
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    write_old(snaps / "prices_2026-10-31.csv")
    write_old(snaps / "prices_2026-11-01.csv", cr_shift=0.3)
    new = pd.read_csv(snaps / "prices_2026-11-01.csv").assign(num_rounds=3, round_started_at="2026-11-01T20:29Z")
    new.to_csv(snaps / "prices_2026-11-01.csv", index=False)
    monkeypatch.chdir(tmp_path)
    import requests
    monkeypatch.setattr(requests, "get", fake_get)
    # snapshot_check.py s'exécute à l'import : on le lance comme script, sans le modifier.
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "src" / "data" / "snapshot_check.py"),
                   run_name="__main__")
    out = capsys.readouterr().out
    assert "prices_2026-11-01.csv" in out and "prix changés : 25" in out
