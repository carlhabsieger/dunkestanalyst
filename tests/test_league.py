"""Ligue (Google Sheet en xlsx) : lecture, sync, exclusions, cache. Aucun réseau."""
import json
import os
import time

import openpyxl
import pandas as pd
import pytest

from src import league as L
from src import team as T
from src.league import LeagueError

POS = ["Guard"] * 4 + ["Forward"] * 4 + ["Centre"] * 2
TEAMS = {
    "Raph": (["Zach Lavine", None, "Raph G3", "Raph G4", "Raph F1", "Raph F2", "Raph F3", "Raph F4",
              "Raph C1", "Raph C2"], ("Steve Kerr", "Golden State Warriors")),
    "Tim": ([f"Tim {p}" for p in ["G1", "G2", "G3", "G4", "F1", "F2", "F3", "F4", "C1", "C2"]],
            ("Joe Mazzulla", "Boston Celtics")),
    "Carl": (["carl g1", "Carl G2", "Carl G3", "Carl G4", "Carl F1", "Carl F2", "Carl F3", "Carl F4",
              "nikola jokić", "Carl C2"], ("Joe Mazzulla", "Boston Celtics")),     # coach identique à Tim
}
NBA_TEAMS = ["Atlanta Hawks", "Boston Celtics", "Chicago Bulls", "Denver Nuggets", "Miami Heat"]


def make_xlsx(path, col0=1, row0=5):
    """Mise en page du vrai fichier ; col0 / row0 décalent tout (aucune position en dur dans le code)."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Équipes"
    ws.cell(1, 1, "Fantasy NBA 2026/27 - Suivi des équipes")
    for b, (who, (players, coach)) in enumerate(TEAMS.items()):
        c = col0 + 4 * b
        ws.cell(row0, c, "Équipe de :"); ws.cell(row0, c + 1, who)
        ws.cell(row0 + 2, c, "Poste"); ws.cell(row0 + 2, c + 1, "Joueur"); ws.cell(row0 + 2, c + 2, "Équipe NBA")
        for i, (pos, p) in enumerate(zip(POS, players)):
            ws.cell(row0 + 3 + i, c, pos)
            if p:
                ws.cell(row0 + 3 + i, c + 1, p); ws.cell(row0 + 3 + i, c + 2, NBA_TEAMS[i % 5])
        ws.cell(row0 + 13, c, "Coach"); ws.cell(row0 + 13, c + 1, coach[0]); ws.cell(row0 + 13, c + 2, coach[1])
        ws.cell(row0 + 15, c, "Postes à pourvoir"); ws.cell(row0 + 15, c + 2, sum(p is None for p in players))
        ws.cell(row0 + 16, c, "Statut"); ws.cell(row0 + 16, c + 2,
                                                  "EN COURS" if None in players else "ÉQUIPE COMPLÈTE")
    j = wb.create_sheet("Joueurs")
    j.append(["Joueur", "Poste", "Équipe NBA", "Crédits (départ)", "Pris par", "Disponible (liste)"])
    j.append(["Zach Lavine", "Guard", "Sacramento Kings", 9.0, "Raph", None])
    j.append(["Free Agent", "Guard", "Miami Heat", 6.0, None, "Free Agent"])
    t = wb.create_sheet("Transferts")
    t.append(["Fenêtre n°", "Participant", "Joueur sorti", "Joueur entré", "Contrôle", None, "Participants"])
    wb.save(path)
    return path


def snapshot_df():
    """Noms canoniques Dunkest (casse et accents différents du fichier)."""
    rows = []
    for who, (players, _) in TEAMS.items():
        for i, (pos, p) in enumerate(zip(POS, players)):
            if not p:
                continue
            name = {"Zach Lavine": "Zach LaVine", "nikola jokić": "Nikola Jokić", "carl g1": "Carl G1"}.get(p, p)
            rows.append({"name": name, "position": pos[0], "team_code": ["ATL", "CHI", "DEN", "MIA", "NYK"][i % 5]})
    rows.append({"name": "Free Agent", "position": "G", "team_code": "MIA"})
    df = pd.DataFrame(rows)
    df["first_name"] = df.name.str.split(" ", n=1).str[0]
    df["last_name"] = df.name.str.split(" ", n=1).str[1]
    return df.assign(id=range(len(df)), cr=8.0, gp=4, pdk=40.0, min=100)


@pytest.fixture(autouse=True)
def fresh_memo(monkeypatch):
    monkeypatch.setattr(L, "_MEMO", {})


# ------------------------------------------------ lecture du classeur

@pytest.mark.parametrize("col0, row0", [(1, 5), (2, 7)])
def test_parse_teams_sans_positions_en_dur(tmp_path, col0, row0):
    wb = L.open_workbook(make_xlsx(tmp_path / "l.xlsx", col0, row0))
    teams = L.parse_teams(wb)
    assert list(teams) == ["Raph", "Tim", "Carl"]
    raph = teams["Raph"]
    assert raph["to_fill"] == 1 and raph["status"] == "EN COURS"
    assert [p for _, p, _ in raph["players"]][:2] == ["Zach Lavine", None] and len(raph["players"]) == 10
    assert teams["Carl"]["coach"] == ("Joe Mazzulla", "Boston Celtics")
    assert L.transfer_header(wb) == ["Fenêtre n°", "Participant", "Joueur sorti", "Joueur entré", "Contrôle"]


def test_autres_participants_sans_coachs(tmp_path):
    teams = L.parse_teams(L.open_workbook(make_xlsx(tmp_path / "l.xlsx")))
    others = L.other_players(teams, "Carl")
    assert len(others) == 19 and "Zach Lavine" in others and "Tim G1" in others
    assert not {"Joe Mazzulla", "Steve Kerr"} & set(others) and "nikola jokić" not in others
    with pytest.raises(LeagueError, match="Participant"):
        L.other_players(teams, "Inconnu")


# ------------------------------------------------ exclusions : seulement pour l'équipe avec "league"

def test_exclusions_seulement_avec_league(tmp_path):
    wb = L.open_workbook(make_xlsx(tmp_path / "l.xlsx"))
    names = list(snapshot_df().name)
    with_league = {**T.EMPTY_TEAM, "league": {"participant": "Carl"}}
    excl = L.league_exclusions(with_league, names, wb=wb, verbose=False)
    assert "Zach LaVine" in excl and "Tim C2" in excl and len(excl) == 19       # noms Dunkest résolus
    assert L.league_exclusions({**T.EMPTY_TEAM}, names, wb=wb, verbose=False) == []


def test_league_of_sans_configuration_ne_touche_pas_le_fichier(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    os.makedirs("teams")
    content = json.dumps({**T.EMPTY_TEAM, "players": ["Tim G1"]}, indent=2)
    (tmp_path / "teams" / "claude.json").write_text(content, encoding="utf-8")
    assert L.league_exclusions_for("claude", list(snapshot_df().name)) == []
    assert (tmp_path / "teams" / "claude.json").read_text(encoding="utf-8") == content
    assert not (tmp_path / "teams" / ".backup").exists()
    assert "aucune exclusion" in capsys.readouterr().out


def test_fichier_equipe_vide_message_clair(tmp_path):
    (tmp_path / "vide.json").write_text("", encoding="utf-8")
    with pytest.raises(T.TeamError, match="vide"):
        T.load_team("vide", root=str(tmp_path))


# ------------------------------------------------ sync

def setup_sync(tmp_path, monkeypatch, players=None):
    monkeypatch.chdir(tmp_path)
    xlsx = make_xlsx(tmp_path / "l.xlsx")
    snap = snapshot_df()
    snap.drop(columns="name").to_csv(tmp_path / "prices.csv", index=False)
    os.makedirs("teams")
    team = {**T.EMPTY_TEAM, "players": players or ["Carl G1", "Carl G2", "Carl G3", "Old Guy", "Carl F1", "Carl F2",
                                                   "Carl F3", "Carl F4", "Nikola Jokic", "Carl C2"],
            "cash": 1.1, "coach": "Ancien Coach", "league": {"participant": "Carl"}}
    path = tmp_path / "teams" / "manugreg.json"
    path.write_text(json.dumps(team, ensure_ascii=False, indent=2), encoding="utf-8")
    return ["sync", "manugreg", "--file", str(xlsx), "--snapshot", str(tmp_path / "prices.csv")], path


def test_sync_diff_et_confirmation_refusee(tmp_path, monkeypatch, capsys):
    args, path = setup_sync(tmp_path, monkeypatch)
    before = path.read_text(encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda q: "n")
    assert T.main(args) == 1
    out = capsys.readouterr().out
    assert "- retiré  : Old Guy" in out and "+ ajouté  : Carl G4" in out
    assert "Nikola" not in out.split("Différences")[1].split("cash")[0]        # Jokic == Jokić : pas de diff
    assert "Joe Mazzulla (BOS)" in out and "inchangé" in out
    assert path.read_text(encoding="utf-8") == before                          # rien écrit


def test_sync_yes_ecrit_et_sauvegarde(tmp_path, monkeypatch, capsys):
    args, path = setup_sync(tmp_path, monkeypatch)
    assert T.main(args + ["--yes"]) == 0
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["cash"] == 1.1 and data["league"] == {"participant": "Carl"}
    assert data["coach"] == "Joe Mazzulla (BOS)"
    assert "Carl G4" in data["players"] and "Old Guy" not in data["players"] and "Nikola Jokić" in data["players"]
    assert len(list((tmp_path / "teams" / ".backup").glob("manugreg-*.json"))) == 1
    assert T.main(args) == 0 and "Aucune différence" in capsys.readouterr().out


def test_sync_nom_non_resolu_rien_ecrit(tmp_path, monkeypatch, capsys):
    args, path = setup_sync(tmp_path, monkeypatch)
    snap = snapshot_df()
    snap[snap.name != "Carl F4"].drop(columns="name").to_csv(tmp_path / "prices.csv", index=False)
    before = path.read_text(encoding="utf-8")
    assert T.main(args + ["--yes"]) == 1
    assert "NON RÉSOLU : Carl F4" in capsys.readouterr().out
    assert path.read_text(encoding="utf-8") == before


def test_sync_refuse_sans_league(tmp_path, monkeypatch, capsys):
    args, path = setup_sync(tmp_path, monkeypatch)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["league"]
    path.write_text(json.dumps(data), encoding="utf-8")
    assert T.main(args + ["--yes"]) == 1 and "pas de synchronisation" in capsys.readouterr().err


# ------------------------------------------------ téléchargement et cache

class FakeHttp:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, url):
        self.calls.append(url)
        return self.responses.pop(0)


def xlsx_bytes(tmp_path):
    return make_xlsx(tmp_path / "src.xlsx").read_bytes()


CFG = {"sheet_id": "ID", "xlsx_path": "absent.xlsx", "cache_hours": 1}


def test_cache_recent_aucune_requete(tmp_path):
    cache = tmp_path / "cache.xlsx"
    cache.write_bytes(xlsx_bytes(tmp_path))
    http = FakeHttp([])
    src = L.fetch_workbook_file(CFG, cache_path=str(cache), http_get=http)
    assert src.origin == "cache" and http.calls == []


def test_cache_perime_une_requete_puis_memo(tmp_path):
    cache = tmp_path / "cache.xlsx"
    cache.write_bytes(b"PK ancien")
    old = time.time() - 3 * 3600
    os.utime(cache, (old, old))
    http = FakeHttp([(200, "application/vnd.openxml", xlsx_bytes(tmp_path))])
    src = L.fetch_workbook_file(CFG, cache_path=str(cache), http_get=http)
    assert src.origin == "téléchargement" and http.calls == ["https://docs.google.com/spreadsheets/d/ID/export?format=xlsx"]
    assert L.fetch_workbook_file(CFG, cache_path=str(cache), http_get=http) is src and len(http.calls) == 1
    assert L.parse_teams(L.open_workbook(src.path))["Carl"]["coach"][0] == "Joe Mazzulla"


def test_reponse_html_de_connexion(tmp_path, monkeypatch):
    login = (200, "text/html; charset=utf-8", b"<!DOCTYPE html><html><body>Connexion Google</body></html>")
    http = FakeHttp([login, login])
    with pytest.raises(LeagueError, match=r"Fichier > Télécharger > \.xlsx"):
        L.fetch_workbook_file({**CFG, "xlsx_path": str(tmp_path / "league.xlsx")},
                              cache_path=str(tmp_path / "cache.xlsx"), http_get=http)
    assert len(http.calls) == 2 and "drive.google.com/uc?export=download&id=ID" in http.calls[1]


def test_html_puis_fichier_local_ou_cache_perime(tmp_path, monkeypatch):
    login = (200, "text/html", b"<html>login</html>")
    local = make_xlsx(tmp_path / "league.xlsx")
    src = L.fetch_workbook_file({**CFG, "xlsx_path": str(local)}, cache_path=str(tmp_path / "c.xlsx"),
                                http_get=FakeHttp([login, login]))
    assert src.origin == "local" and src.warnings
    monkeypatch.setattr(L, "_MEMO", {})
    cache = tmp_path / "old.xlsx"
    cache.write_bytes(xlsx_bytes(tmp_path))
    old = time.time() - 5 * 3600
    os.utime(cache, (old, old))
    src = L.fetch_workbook_file(CFG, cache_path=str(cache), http_get=FakeHttp([login, (500, "text/plain", b"")]))
    assert src.origin == "cache périmé" and "périmé" in src.warnings[0]


# ------------------------------------------------ onglet Transferts

def test_lignes_transferts_ordre_et_orthographe_du_fichier(tmp_path):
    wb = L.open_workbook(make_xlsx(tmp_path / "l.xlsx"))
    rows = L.transfer_rows([("Carl G1", "Zach LaVine")], "Carl", L.transfer_header(wb), L.sheet_player_names(wb))
    assert rows == [["<à compléter>", "Carl", "Carl G1", "Zach Lavine", "(calculé par le Sheet)"]]
