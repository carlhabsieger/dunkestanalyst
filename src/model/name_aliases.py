"""Alias explicites nom Dunkest -> nom NBA (PLAYER_NAME des logs nba_api).

Uniquement les correspondances vérifiées dans data/player_logs.csv (même équipe des deux côtés).
Ajouter ici les non rapprochés signalés par le récapitulatif de project_all_games.
"""
NAME_ALIASES = {
    "Alexandre Sarr": "Alex Sarr",                 # WAS
    "Gregory Jackson": "GG Jackson",               # MEM
    "Bobby Portis": "Bobby Portis Jr.",            # MIA
    "Robert Williams": "Robert Williams III",      # POR
    "Morez Johnson": "Morez Johnson Jr.",          # DAL
}
