import numpy as np
import pandas as pd
import pytest

from src.buy import BuyError, check_request, parse_buy, parse_slot, solve_buy


def make_pool():
    """21 joueurs, 3 équipes (AAA, BBB, CCC) : 7 par équipe (3 G, 3 F, 1 C)."""
    rows = []
    for t, team in enumerate(["AAA", "BBB", "CCC"]):
        for k, pos in enumerate(["G", "G", "G", "F", "F", "F", "C"]):
            cr = 4.0 + 2.0 * k + t          # prix de 4 à 18
            rows.append({"name": f"{team} {pos}{k}", "team_code": team, "position": pos,
                         "cr": cr, "gp": 4, "proj": 2.0 * cr + (3 - t)})
    return pd.DataFrame(rows)


def names(opt):
    return list(opt.table.name)


def test_parse_slot_variants():
    s = parse_slot("T:G@8-12")
    assert (s.role, s.pos, s.lo, s.hi) == ("T", "G", 8.0, 12.0)
    s = parse_slot("R:F@-5")
    assert (s.role, s.pos, s.lo, s.hi) == ("R", "F", -np.inf, 5.0)
    s = parse_slot("c@8-")
    assert (s.role, s.pos, s.lo, s.hi) == ("T", "C", 8.0, np.inf)
    s = parse_slot("6:*@9.5-9.5")
    assert (s.role, s.pos, s.lo, s.hi) == ("6", "*", 9.5, 9.5)
    assert len(parse_buy("T:G@8-12,R:F@-5,6:*")) == 3
    for bad in ("X:G", "T:Z", "G@12-8", "G@", "G@a-3"):
        with pytest.raises(BuyError):
            parse_slot(bad)


def test_a_titulaire_prefere_au_remplacant_budget_serre():
    pool = make_pool()
    # Budget serré : un seul joueur cher possible ; il doit aller sur le slot T (poids 1,0), pas sur R (0,5).
    opt = solve_buy(pool, parse_buy("R:*,T:*"), credits=22, alts=1)[0]
    t = opt.table.set_index("role")
    assert t.loc["T", "proj"] > t.loc["R", "proj"]
    assert t.loc["T", "cr"] > t.loc["R", "cr"]
    assert opt.cost <= 22 + 1e-9
    assert opt.value == pytest.approx(t.loc["T", "proj"] + 0.5 * t.loc["R", "proj"])


def test_b_fourchette_de_prix_respectee():
    pool = make_pool()
    opts = solve_buy(pool, parse_buy("T:G@6-8,R:F@-11,6:*@15-15"), credits=100, alts=3)
    assert len(opts) == 3
    for opt in opts:
        t = opt.table.set_index("slot")
        assert 6 <= t.loc["T:G@6-8", "cr"] <= 8 and t.loc["T:G@6-8", "position"] == "G"
        assert t.loc["R:F@-11", "cr"] <= 11 and t.loc["R:F@-11", "position"] == "F"
        assert t.loc["6:*@15-15", "cr"] == 15
    assert len({frozenset(names(o)) for o in opts}) == 3      # coupes no-good : combinaisons distinctes


def test_c_limite_trois_par_equipe_avec_owned():
    pool = make_pool()
    pool.loc[pool.team_code == "AAA", "proj"] += 20      # AAA domine
    owned = pool[pool.name.isin(["AAA G0", "AAA F3"])]
    cands = pool[~pool.name.isin(owned.name)]
    # Sans possédés, les 2 meilleurs joueurs sont chez AAA ; avec 2 AAA possédés, un seul est permis.
    free = solve_buy(cands, parse_buy("T:*,T:*"), credits=100, alts=1)[0]
    assert (free.table.team_code == "AAA").sum() == 2
    for opt in solve_buy(cands, parse_buy("T:*,T:*"), credits=100, owned=owned, alts=3):
        assert (opt.table.team_code == "AAA").sum() + len(owned) <= 3


def test_d_trois_pivots_titulaires_impossible():
    pool = make_pool()
    with pytest.raises(BuyError, match="formation"):
        solve_buy(pool, parse_buy("T:C,T:C,T:C"), credits=100)
    # Même message en passant par un titulaire possédé décrit avec --owned-roles.
    owned = pool[pool.name == "AAA C6"].assign(role="T")
    with pytest.raises(BuyError, match="formation"):
        check_request(parse_buy("T:G,T:G,T:G,T:C"), owned)


def test_d_bis_joker_titulaire_respecte_la_formation():
    pool = make_pool()
    # 3 G titulaires fixés + 2 jokers : seule 3-1-1 convient, donc pas de 4e G titulaire.
    opt = solve_buy(pool, parse_buy("T:G,T:G,T:G,T:*,T:*"), credits=200, alts=1)[0]
    assert sorted(opt.table.position) == ["C", "F", "G", "G", "G"]


def test_e_capitaine_x1_5():
    pool = make_pool()
    spec = parse_buy("T:G@-8,T:F@-12,R:C@-17")
    base = solve_buy(pool, spec, credits=100, alts=1)[0]
    cap = solve_buy(pool, spec, credits=100, captain=True, alts=1)[0]
    t = cap.table
    assert t.captain.sum() == 1
    capt = t[t.captain].iloc[0]
    assert capt.role == "T"
    assert capt.pts == pytest.approx(1.5 * capt.proj)
    # Les fourchettes sont serrées : même effectif, le capitaine est le meilleur titulaire.
    assert set(names(cap)) == set(names(base))
    assert capt.proj == t[t.role == "T"].proj.max()
    assert cap.value == pytest.approx(base.value + 0.5 * capt.proj)


def test_budget_insuffisant_message():
    with pytest.raises(BuyError, match="Budget insuffisant"):
        solve_buy(make_pool(), parse_buy("T:G@10-,R:F@10-"), credits=15)


def test_regles_roles():
    with pytest.raises(BuyError, match="6e homme"):
        check_request(parse_buy("6:G,6:F"))
    with pytest.raises(BuyError, match="remplaçant"):
        check_request(parse_buy("R:G,R:G,R:F,R:F,R:C"))
    with pytest.raises(BuyError, match="Trop de C"):
        check_request(parse_buy("T:C,R:C,R:C"))
