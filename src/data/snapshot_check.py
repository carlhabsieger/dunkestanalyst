"""Compare les deux derniers snapshots et l'état en direct de Dunkest (1 requête)."""
import glob
import pandas as pd
import requests

CONFIG = "https://fantaking-api.dunkest.com/api/v1/leagues/1/config"
STATUS = {1: "trades_open", 2: "matches_in_progress", 3: "substitutions_open",
          4: "calculation_in_progress", 5: "matchday_ended", 6: "inactive"}

files = sorted(glob.glob("snapshots/prices_*.csv"))
if not files:
    raise SystemExit("Aucun snapshot.")
new = pd.read_csv(files[-1])
sid = int(new.status_id.iloc[0])
print(f"Dernier snapshot : {files[-1]} | journée {new.matchday.iloc[0]}, "
      f"tour {new['round'].iloc[0]}, statut {sid} ({STATUS.get(sid)}) | "
      f"pris à {new.snapshot_ts.iloc[0]} | {len(new)} joueurs")

r = requests.get(CONFIG, timeout=30, headers={
    "User-Agent": "Mozilla/5.0 (usage personnel)", "Accept": "application/json"})
r.raise_for_status()
d = r.json()["data"]
start = pd.Timestamp(d["current_round"]["started_at"]).tz_convert("Europe/Paris")
live = d["status_id"]
print(f"En direct : journée {d['current_matchday']['number']}, "
      f"tour {d['current_round']['number']}, statut {live} ({STATUS.get(live)}), "
      f"tour daté du {start:%d/%m %H:%M} (Paris)")
if live != sid:
    print("=> Le statut a changé depuis le snapshot : relance snapshot_prices.")

if len(files) >= 2:
    old = pd.read_csv(files[-2])
    m = old.merge(new, on="id", suffixes=("_old", "_new"))
    m["d_cr"] = (m.cr_new - m.cr_old).round(2)
    m["d_gp"] = m.gp_new - m.gp_old
    m["d_pdk"] = (m.pdk_new - m.pdk_old).round(2)
    m["name"] = m.first_name_new + " " + m.last_name_new
    print(f"\nComparaison avec {files[-2]} ({len(m)} joueurs communs) :")
    print(f"  matchs joués en plus : {(m.d_gp != 0).sum()} joueurs | "
          f"prix changés : {(m.d_cr != 0).sum()} | scores changés : {(m.d_pdk != 0).sum()}")
    if (m.d_cr != 0).sum() == 0:
        print("  ATTENTION : aucun prix n'a bougé, snapshot probablement pris avant la mise à jour.")
    cols = ["name", "team_code_new", "cr_old", "cr_new", "d_cr", "d_gp", "d_pdk"]
    print("\nPlus fortes hausses de prix :")
    print(m.sort_values("d_cr", ascending=False).head(5)[cols].to_string(index=False))
    print("\nPlus fortes baisses :")
    print(m.sort_values("d_cr").head(5)[cols].to_string(index=False))
else:
    print("\nUn seul snapshot : la comparaison viendra avec le suivant.")
if sid != 1:
    print("\nNote : statut différent de 1 (trades_open). Mon hypothèse, non vérifiée : "
          "les prix ne sont définitifs qu'une fois le marché ouvert.")