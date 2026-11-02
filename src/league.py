"""Ligue privée : lecture du Google Sheet (xlsx) en LECTURE SEULE.

Configuration non versionnée : config/league.json
    {"sheet_id": "...", "gid": 1748654859, "xlsx_path": "data/league.xlsx", "cache_hours": 1}

Chargement du classeur, une seule requête par exécution au plus :
1. --file chemin si donné ;
2. cache data/league_cache.xlsx s'il a moins de cache_hours heures ;
3. téléchargement : export xlsx de Google Sheets, puis lien de téléchargement Drive si le premier échoue
   (aucune autre URL) ;
4. si la réponse est une page HTML (connexion) ou une erreur : le fichier local xlsx_path s'il existe,
   sinon le cache périmé avec avertissement, sinon arrêt avec un message clair.
"""
import json
import os
import time
from dataclasses import dataclass, field

CONFIG_PATH = "config/league.json"
CACHE_PATH = "data/league_cache.xlsx"
DEFAULT_CACHE_HOURS = 1.0
EXPORT_URLS = ("https://docs.google.com/spreadsheets/d/{id}/export?format=xlsx",
               "https://drive.google.com/uc?export=download&id={id}")
DOWNLOAD_HINT = ("télécharge le fichier (Fichier > Télécharger > .xlsx) dans {path} : il sera utilisé à la "
                 "place du téléchargement.")
TRANSFER_SHEET = "Transferts"
TEAMS_SHEET = "Équipes"
PLAYERS_SHEET = "Joueurs"


class LeagueError(RuntimeError):
    """Configuration absente, fichier illisible, participant inconnu : message lisible."""


# ---------------------------------------------------------------- configuration et chargement

def load_config(path=CONFIG_PATH):
    if not os.path.exists(path):
        raise LeagueError(f"Configuration de ligue absente ({path}). Crée-la avec : "
                          '{"sheet_id": "...", "gid": 0, "xlsx_path": "data/league.xlsx"}')
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    if not cfg.get("sheet_id"):
        raise LeagueError(f"{path} : champ sheet_id manquant.")
    return cfg


def _http_get(url):
    import requests
    r = requests.get(url, timeout=30, allow_redirects=True,
                     headers={"User-Agent": "Mozilla/5.0 (usage personnel, lecture seule)"})
    return r.status_code, r.headers.get("content-type", ""), r.content


def is_xlsx(content):
    return content[:2] == b"PK"            # un xlsx est une archive zip ; une page de connexion est du HTML


@dataclass
class Source:
    path: str
    origin: str                     # "fichier", "cache", "téléchargement", "local", "cache périmé"
    age_s: float
    url: str = ""
    warnings: list = field(default_factory=list)

    def describe(self):
        h, m = divmod(int(self.age_s // 60), 60)
        extra = f" ({self.url})" if self.url else ""
        return f"Ligue : {self.path}, source {self.origin}{extra}, âge {h} h {m:02d} min"


_MEMO = {}


def fetch_workbook_file(cfg, file=None, cache_path=CACHE_PATH, http_get=None, now=None, refresh=False):
    """Chemin d'un xlsx à lire, selon l'ordre décrit en tête de module. Une requête par exécution au plus."""
    now = now or time.time()
    if file:
        if not os.path.exists(file):
            raise LeagueError(f"Fichier introuvable : {file}")
        return Source(file, "fichier", now - os.path.getmtime(file))
    key = (cfg["sheet_id"], cache_path)
    if key in _MEMO and not refresh:
        return _MEMO[key]
    max_age = float(cfg.get("cache_hours", DEFAULT_CACHE_HOURS)) * 3600
    cache_age = now - os.path.getmtime(cache_path) if os.path.exists(cache_path) else None
    if cache_age is not None and cache_age < max_age and not refresh:
        _MEMO[key] = Source(cache_path, "cache", cache_age)
        return _MEMO[key]
    http_get = http_get or _http_get
    errors = []
    for tpl in EXPORT_URLS:
        url = tpl.format(id=cfg["sheet_id"])
        try:
            status, ctype, content = http_get(url)
        except Exception as e:
            errors.append(f"{url} : {type(e).__name__}")
            continue
        if status == 200 and is_xlsx(content):
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(content)
            _MEMO[key] = Source(cache_path, "téléchargement", 0.0, url)
            return _MEMO[key]
        kind = "page HTML (connexion ?)" if b"<html" in content[:2000].lower() or "html" in ctype else f"HTTP {status}"
        errors.append(f"{url} : {kind}")
    local = cfg.get("xlsx_path", "data/league.xlsx")
    msg = "Téléchargement impossible (" + " ; ".join(errors) + "). "
    if os.path.exists(local):
        _MEMO[key] = Source(local, "local", now - os.path.getmtime(local), warnings=[msg + f"Fichier local utilisé : {local}"])
        return _MEMO[key]
    if cache_age is not None:
        _MEMO[key] = Source(cache_path, "cache périmé", cache_age,
                            warnings=[msg + "Cache périmé utilisé, à rafraîchir."])
        return _MEMO[key]
    raise LeagueError(msg + DOWNLOAD_HINT.format(path=local))


def open_workbook(path):
    import openpyxl
    try:
        return openpyxl.load_workbook(path, data_only=True, read_only=False)
    except Exception as e:
        raise LeagueError(f"Classeur illisible ({path}) : {type(e).__name__}: {e}") from None


# ---------------------------------------------------------------- lecture du classeur

def parse_teams(wb):
    """{participant: {"players": [(poste, joueur ou None, équipe)], "coach": (nom, équipe), "to_fill", "status"}}.

    Les blocs sont repérés par les cellules commençant par « Équipe de » ; le participant est la cellule
    suivante. Dans chaque bloc : en-tête « Poste », joueurs jusqu'à la ligne « Coach », puis les lignes
    « Postes à pourvoir » et « Statut » (valeur deux colonnes à droite du libellé).
    """
    if TEAMS_SHEET not in wb.sheetnames:
        raise LeagueError(f"Onglet « {TEAMS_SHEET} » absent (onglets : {', '.join(wb.sheetnames)}).")
    ws = wb[TEAMS_SHEET]
    out = {}
    for row in ws.iter_rows():
        for c in row:
            if not (isinstance(c.value, str) and c.value.strip().startswith("Équipe de")):
                continue
            col, r0 = c.column, c.row
            name = ws.cell(r0, col + 1).value
            if not name:
                rest = c.value.split(":", 1)[-1].strip()
                name = rest or None
            if not name:
                continue
            block = {"players": [], "coach": None, "to_fill": None, "status": None}
            r = r0 + 1
            while r <= ws.max_row and str(ws.cell(r, col).value or "").strip() != "Poste":
                r += 1
            r += 1
            while r <= ws.max_row:
                label = str(ws.cell(r, col).value or "").strip()
                player, team = ws.cell(r, col + 1).value, ws.cell(r, col + 2).value
                if label == "Coach":
                    block["coach"] = (str(player).strip() if player else None, team)
                    break
                if label:
                    block["players"].append((label, str(player).strip() if player else None, team))
                r += 1
            for rr in range(r + 1, min(ws.max_row, r + 6) + 1):
                label = str(ws.cell(rr, col).value or "").strip()
                if label.startswith("Postes à pourvoir"):
                    block["to_fill"] = ws.cell(rr, col + 2).value
                elif label.startswith("Statut"):
                    block["status"] = ws.cell(rr, col + 2).value
            if block["to_fill"] is None:
                block["to_fill"] = sum(p is None for _, p, _ in block["players"])
            out[str(name).strip()] = block
    if not out:
        raise LeagueError(f"Aucun bloc « Équipe de » trouvé dans l'onglet {TEAMS_SHEET}.")
    return out


def transfer_header(wb):
    """Colonnes de l'onglet Transferts, dans l'ordre, jusqu'à la première cellule vide de la ligne 1."""
    if TRANSFER_SHEET not in wb.sheetnames:
        raise LeagueError(f"Onglet « {TRANSFER_SHEET} » absent.")
    cols = []
    for c in next(wb[TRANSFER_SHEET].iter_rows(min_row=1, max_row=1)):
        if c.value in (None, ""):
            break
        cols.append(str(c.value).strip())
    return cols


def sheet_player_names(wb):
    """Noms tels qu'écrits dans l'onglet Joueurs (colonne A), pour les listes déroulantes."""
    if PLAYERS_SHEET not in wb.sheetnames:
        return []
    return [r[0] for r in wb[PLAYERS_SHEET].iter_rows(min_row=2, max_col=1, values_only=True) if r[0]]


def other_players(teams, participant):
    """Joueurs (noms du fichier) de tous les AUTRES participants ; les coachs ne sont jamais exclus."""
    if participant not in teams:
        raise LeagueError(f"Participant « {participant} » absent du fichier (participants : {', '.join(teams)}).")
    return [p for who, b in teams.items() if who != participant for _, p, _ in b["players"] if p]


# ---------------------------------------------------------------- liens avec les équipes

def team_league(team):
    """Configuration "league" d'une équipe, ou None (jamais de synchronisation ni d'exclusion)."""
    lg = team.get("league")
    return lg if isinstance(lg, dict) and lg.get("participant") else None


def league_context(file=None, cfg=None, verbose=True, **kw):
    """(source, classeur) : charge une seule fois par exécution."""
    cfg = cfg or ({} if file else load_config())      # --file : pas besoin de configuration
    src = fetch_workbook_file(cfg, file=file, **kw)
    if verbose:
        print(src.describe())
        for w in src.warnings:
            print(f"ATTENTION : {w}")
    return src, open_workbook(src.path)


def resolve_all(names, snapshot_names):
    """(noms Dunkest résolus, {nom du fichier: nom Dunkest}, [non résolus avec raison]) sans deviner."""
    from src.team import TeamError, resolve_name
    mapping, unresolved = {}, []
    for n in names:
        try:
            mapping[n] = resolve_name(n, snapshot_names)
        except TeamError as e:
            unresolved.append(f"{n} : {e}")
    return list(mapping.values()), mapping, unresolved


def league_exclusions(team, snapshot_names, file=None, verbose=True, wb=None):
    """Noms Dunkest des joueurs des autres participants pour une équipe configurée, sinon []."""
    lg = team_league(team)
    if lg is None:
        return []
    if wb is None:
        _, wb = league_context(file=file, verbose=verbose)
    names = other_players(parse_teams(wb), lg["participant"])
    resolved, _, unresolved = resolve_all(names, snapshot_names)
    if verbose:
        print(f"Ligue : {len(resolved)} joueurs des autres participants exclus des candidats.")
        for u in unresolved:
            print(f"ATTENTION : joueur de la ligue non résolu (non exclu) : {u}")
    return resolved


def league_exclusions_for(team_name, snapshot_names, file=None):
    """--league-of NOM (buy, optimizer) : exclusions de ligue de l'équipe NOM, [] si elle n'en a pas."""
    from src.team import load_team
    team = load_team(team_name)
    if team_league(team) is None:
        print(f"Info : l'équipe « {team_name} » n'a pas de configuration \"league\" : aucune exclusion de ligue.")
        return []
    return league_exclusions(team, snapshot_names, file=file)


def sheet_spelling(dunkest_name, sheet_names):
    """Nom tel qu'il apparaît dans l'onglet Joueurs (liste déroulante), sinon le nom Dunkest."""
    from src.model.projections import normalize_name
    key = normalize_name(dunkest_name)
    for n in sheet_names:
        if normalize_name(n) == key:
            return n
    return dunkest_name


def transfer_rows(moves, participant, header, sheet_names):
    """Lignes à recopier dans l'onglet Transferts, dans l'ordre des colonnes du fichier."""
    rows = []
    for out, inn in moves:
        values = {"Participant": participant, "Joueur sorti": sheet_spelling(out, sheet_names),
                  "Joueur entré": sheet_spelling(inn, sheet_names)}
        row = []
        for col in header:
            if col in values:
                row.append(values[col])
            elif col.lower().startswith("fenêtre"):
                row.append("<à compléter>")
            elif col.lower().startswith("contrôle"):
                row.append("(calculé par le Sheet)")
            else:
                row.append("")
        rows.append(row)
    return rows
