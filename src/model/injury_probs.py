"""Probabilité de jouer selon le statut du rapport officiel des blessures NBA."""
import warnings

# valeurs provisoires, à calibrer sur l'historique des rapports
STATUS_P_PLAY = {
    "Out": 0.0,
    "Doubtful": 0.15,
    "Questionable": 0.5,
    "Probable": 0.85,
    "Available": 1.0,
}
UNKNOWN_P_PLAY = 0.5
NOT_LISTED = "non listé"       # joueur absent du rapport : p = 1.0
OUT = "Out"


class UnknownStatusWarning(UserWarning):
    pass


def p_play_for(statuses):
    """Série de statuts -> série de probabilités ; statut inconnu : 0,5 avec avertissement."""
    known = set(STATUS_P_PLAY) | {NOT_LISTED}
    unknown = sorted(set(statuses.dropna()) - known)
    if unknown:
        warnings.warn(f"Statut(s) de blessure inconnu(s), probabilité {UNKNOWN_P_PLAY} appliquée : "
                      f"{', '.join(unknown)}", UnknownStatusWarning, stacklevel=2)
    return statuses.map(lambda s: 1.0 if s == NOT_LISTED else STATUS_P_PLAY.get(s, UNKNOWN_P_PLAY))
