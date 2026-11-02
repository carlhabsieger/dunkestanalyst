"""Gestion des équipes (src/team.py) : sans réseau, données synthétiques."""
import json

import pandas as pd
import pytest

from src import team as T
from src.team import TeamError


# ------------------------------------------------ noms

NAMES = ["Nikola Jokić", "Jalen Williams", "Jaylin Williams", "Bobby Portis", "Shai Gilgeous-Alexander"]


def test_resolution_tolerante():
    assert T.resolve_name("nikola jokic", NAMES) == "Nikola Jokić"            # accents et casse
    assert T.resolve_name("NIKOLA JOKC", NAMES) == "Nikola Jokić"             # faute de frappe (difflib)
    assert T.resolve_name("Bobby Portis Jr.", NAMES) == "Bobby Portis"        # alias nom NBA -> Dunkest
    assert T.resolve_name("gilgeous", NAMES) == "Shai Gilgeous-Alexander"     # sous-chaîne unique


def test_resolution_ambigue_ou_introuvable():
    with pytest.raises(TeamError, match="ambigu.*Jalen Williams.*Jaylin Williams"):
        T.resolve_name("Williams", NAMES)
    with pytest.raises(TeamError, match="introuvable"):
        T.resolve_name("Zzz Inconnu", NAMES)


# ------------------------------------------------ effectif de référence

def squad_rows():
    """10 joueurs : six à 30 points (titulaires + 6e homme), quatre à 2 points (banc à 50 %)."""
    spec = [("G1", "G", "AAA", 30), ("G2", "G", "BBB", 30), ("G3", "G", "CCC", 30), ("G4", "G", "DDD", 2),
            ("F1", "F", "AAA", 30), ("F2", "F", "BBB", 30), ("F3", "F", "CCC", 2), ("F4", "F", "DDD", 2),
            ("C1", "C", "EEE", 30), ("C2", "C", "EEE", 2)]
    return [{"id": i, "name": n, "position": p, "team_code": t, "cr": 8.0, "proj": float(v), "gp": 4}
            for i, (n, p, t, v) in enumerate(spec)]


def owned():
    return pd.DataFrame(squad_rows())


def cand(name, pos, team, proj, cr=8.0, i=100):
    return pd.DataFrame([{"id": i, "name": name, "position": pos, "team_code": team, "cr": cr,
                          "proj": float(proj), "gp": 4}])


BASE = 30 * 5 + 30 + 15 + 0.5 * 2 * 4          # 5 titulaires + 6e homme + bonus capitaine + banc


def test_lineup_capitaine_parmi_les_titulaires():
    o = owned()
    o.loc[o.name == "C1", "proj"] = 40.0
    value, fm, lu = T.lineup(o)
    roles = lu.set_index("name").role
    assert roles["C1"] == "capitaine"                                         # le meilleur titulaire
    assert (lu.role == "capitaine").sum() == 1 and (lu.role == "titulaire").sum() == 4
    assert (lu.role == "6e homme").sum() == 1 and value == pytest.approx(BASE + 10 * 1.5)


def test_baseline_suggest_egale_lineup():
    base, opts, best = T.suggest(owned(), cand("X", "G", "FFF", 1), cash=0, max_transfers=1)
    assert base.value == pytest.approx(T.lineup(owned())[0]) == pytest.approx(BASE)


def test_a_gain_brut_4_net_negatif_rien_recommande():
    base, opts, best = T.suggest(owned(), cand("X", "G", "FFF", 10), cash=0, max_transfers=1)
    o1 = opts[1]
    assert o1.plan.n_transfers == 0 and best is None                          # au plus 1 : rien de rentable
    forced = o1.forced                                                        # exactement 1 transfert
    gross = forced.value - base.value
    assert forced.inn == ["X"] and forced.out == ["G4"]
    assert gross == pytest.approx(4.0) and gross - 5 == pytest.approx(-1.0)


def test_b_gain_brut_9_recommande():
    base, opts, best = T.suggest(owned(), cand("X", "G", "FFF", 20), cash=0, max_transfers=1)
    assert best is not None and best.plan.inn == ["X"] and best.plan.out == ["G4"]
    assert best.gross == pytest.approx(9.0) and best.cost == 5 and best.net == pytest.approx(4.0)


def test_paires_par_poste():
    c = pd.concat([cand("NewC", "C", "FFF", 25, i=101), cand("NewG", "G", "GGG", 25, i=102)])
    _, _, best = T.suggest(owned(), c, cash=0, max_transfers=2)
    assert sorted(best.plan.moves(owned())) == [("C2", "NewC"), ("G4", "NewG")]


def test_seuil_min_net_gain():
    _, _, best = T.suggest(owned(), cand("X", "G", "FFF", 20), cash=0, max_transfers=1, min_net_gain=5.0)
    assert best is None                                                       # net 4 < seuil 5


def test_c_budget_prix_possedes_plus_cash():
    star = cand("Star", "G", "FFF", 100, cr=9.0)
    _, _, best = T.suggest(owned(), star, cash=0.0, max_transfers=1)
    assert best is None                                                       # 80 + 0 < 81 nécessaires
    _, _, best = T.suggest(owned(), star, cash=1.0, max_transfers=1)
    assert best.plan.inn == ["Star"] and best.plan.squad.cr.sum() <= 81.0 + 1e-9
    assert best.plan.credits_left == pytest.approx(0.0)


def test_d_trois_par_equipe():
    o = owned()
    o.loc[o.name == "F3", "team_code"] = "AAA"                                # AAA : G1, F1, F3
    z = cand("Zed", "G", "AAA", 100)
    _, opts, best = T.suggest(o, z, cash=0, max_transfers=2)
    for opt in opts:
        assert opt.plan.squad.team_code.value_counts().max() <= 3
    assert "Zed" in best.plan.inn and best.plan.out[0] in {"G1"}              # seul un G de AAA peut sortir


def test_e_joueur_out_possede_signale_et_remplace():
    snap = owned().drop(columns=["proj"])
    d = owned().assign(status="non listé", p_play=1.0, reason="")
    d.loc[d.name == "G1", ["status", "p_play", "proj", "reason"]] = ["Out", 0.0, 0.0, "Injury"]
    team = {**T.EMPTY_TEAM, "players": list(snap.name)}
    o = T.owned_frame(team, snap, d)
    assert any("G1" in w and "Out" in w for w in T.team_warnings(o))
    _, _, best = T.suggest(o, cand("W", "G", "FFF", 25).assign(status="non listé", p_play=1.0, reason=""),
                           cash=0, max_transfers=1)
    assert best is not None and best.plan.out == ["G1"] and best.net > 0


def test_f_absent_du_snapshot_sans_prix():
    snap = owned().drop(columns=["proj"])
    team = {**T.EMPTY_TEAM, "players": list(snap.name[:9]) + ["Fantome Absent"]}
    with pytest.raises(TeamError, match="price_override"):
        T.player_info(team, snap)
    team["price_overrides"] = {"Fantome Absent": {"cr": 5.0, "position": "C", "team_code": "ZZZ"}}
    info = T.player_info(team, snap)
    assert info.set_index("name").loc["Fantome Absent", "cr"] == 5.0


def test_horizon_tout_l_effectif_compte_h_fois():
    # X (6) remplace G4 (2) sur le banc : +0,5 x (6 - 2) = +2 par journée.
    x = cand("X", "G", "FFF", 6)
    base3, opts3, best3 = T.suggest(owned(), x, cash=0, max_transfers=1, horizon=3)
    assert base3.value == pytest.approx(3 * BASE)                          # baseline comptée 3 fois
    assert best3 is not None and best3.plan.inn == ["X"] and best3.plan.out == ["G4"]
    assert best3.gross == pytest.approx(6.0) and best3.cost == 5 and best3.net == pytest.approx(1.0)
    _, opts1, best1 = T.suggest(owned(), x, cash=0, max_transfers=1)       # sans l'option
    assert best1 is None
    forced = opts1[1].forced
    assert forced.value - BASE == pytest.approx(2.0) and forced.value - BASE - 5 == pytest.approx(-3.0)


# ------------------------------------------------ add / remove / swap

def snapshot_pool():
    rows = squad_rows() + [
        {"id": 50, "name": "Extra Guard", "position": "G", "team_code": "FFF", "cr": 5.0, "gp": 4},
        {"id": 51, "name": "Extra Center", "position": "C", "team_code": "FFF", "cr": 5.0, "gp": 4},
        {"id": 52, "name": "Autre Aaa", "position": "F", "team_code": "AAA", "cr": 5.0, "gp": 4},
    ]
    return pd.DataFrame(rows).drop(columns=["proj"])


def test_add_refus_au_dela_de_10_et_quota():
    snap = snapshot_pool()
    full = {**T.EMPTY_TEAM, "players": list(snap.name[:10])}
    with pytest.raises(TeamError, match="limité à 10"):
        T.add_player(full, "extra guard", snap)
    four_g = {**T.EMPTY_TEAM, "players": ["G1", "G2", "G3", "G4"]}
    with pytest.raises(TeamError, match="Trop de G"):
        T.add_player(four_g, "Extra Guard", snap)
    t, nm = T.add_player({**T.EMPTY_TEAM, "players": ["G1"]}, "extra  GUARD", snap)
    assert nm == "Extra Guard" and t["players"] == ["G1", "Extra Guard"]


def test_add_refus_quatrieme_joueur_meme_equipe():
    snap = snapshot_pool()
    snap.loc[snap.name == "G2", "team_code"] = "AAA"                          # AAA : G1, F1, G2, Autre Aaa
    t, _ = T.add_player({**T.EMPTY_TEAM, "players": ["G1", "F1"]}, "Autre Aaa", snap)   # 3e AAA : accepté
    with pytest.raises(TeamError, match="AAA"):
        T.add_player(t, "G2", snap)                                           # 4e AAA : refusé
    coach = {**T.EMPTY_TEAM, "players": ["G1", "F1"], "coach": "Coach Test (AAA)"}
    with pytest.raises(TeamError, match="AAA"):
        T.add_player(coach, "Autre Aaa", snap)                                # coach noté (AAA) compte


def test_remove_et_swap():
    snap = snapshot_pool()
    t = {**T.EMPTY_TEAM, "players": list(snap.name[:10])}
    t2, nm = T.remove_player(t, "g4")
    assert nm == "G4" and len(t2["players"]) == 9
    t3, out, inn = T.swap_player(t, "G4", "Extra Guard", snap)
    assert (out, inn) == ("G4", "Extra Guard") and len(t3["players"]) == 10
    with pytest.raises(TeamError, match="Trop de C"):
        T.swap_player(t, "G4", "Extra Center", snap)                          # 3e pivot refusé


# ------------------------------------------------ stockage et CLI

def test_stockage_et_sauvegarde(tmp_path):
    team = {**T.EMPTY_TEAM, "players": ["G1"], "cash": 1.4}
    T.save_team("demo", team, root=str(tmp_path))
    T.save_team("demo", {**team, "cash": 2.0}, root=str(tmp_path))
    data = json.loads((tmp_path / "demo.json").read_text(encoding="utf-8"))
    assert list(data) == list(T.EMPTY_TEAM) and data["cash"] == 2.0
    backups = list((tmp_path / ".backup").glob("demo-*.json"))
    assert len(backups) == 1 and json.loads(backups[0].read_text())["cash"] == 1.4
    assert T.load_team("demo", root=str(tmp_path))["players"] == ["G1"]
    with pytest.raises(TeamError, match="introuvable"):
        T.load_team("absente", root=str(tmp_path))


def test_cli_init_add_remove(tmp_path, monkeypatch, capsys):
    snap = snapshot_pool().assign(first_name="", last_name="", pdk=10.0, min=90)
    snap[["first_name", "last_name"]] = snap.name.str.split(" ", n=1, expand=True).fillna("")
    snap.loc[snap.last_name == "", "last_name"] = snap.first_name
    snap.loc[snap.first_name == snap.last_name, "first_name"] = "X"
    p = tmp_path / "prices.csv"
    snap.drop(columns="name").to_csv(p, index=False)
    monkeypatch.chdir(tmp_path)
    assert T.main(["init", "essai", "--cash", "1.5", "--snapshot", str(p)]) == 0
    assert T.main(["add", "essai", "extra guard", "--snapshot", str(p)]) == 0
    assert T.main(["add", "essai", "Williams", "--snapshot", str(p)]) == 1    # introuvable : message, code 1
    assert T.main(["remove", "essai", "Extra Guard", "--snapshot", str(p)]) == 0
    assert json.loads((tmp_path / "teams" / "essai.json").read_text())["players"] == []
    assert "introuvable" in capsys.readouterr().err


def test_html_autonome():
    h = T.render_html("Équipe <x>", ["ligne"], pd.DataFrame({"joueur": ["A ?"], "statut": ["Out"]}), ["Out"])
    assert "<script" not in h and "http" not in h and "&lt;x&gt;" in h and "class='bad'" in h
