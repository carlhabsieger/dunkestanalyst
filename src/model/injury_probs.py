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
UNKNOWN_P_PLAY = 0.5           # statut écrit dans le rapport mais absent de STATUS_P_PLAY
OUT = "Out"

# Joueurs sans ligne dans le rapport
NOT_LISTED = "non listé"              # équipe présente et rapport soumis : p = 1
NOT_SUBMITTED = "rapport non soumis"  # équipe présente, rapport pas encore soumis : p = unknown_p
OUT_OF_REPORT = "hors rapport"        # équipe absente du rapport (pas de match couvert) : p = unknown_p
UNCERTAIN = (NOT_SUBMITTED, OUT_OF_REPORT)
DEFAULT_UNKNOWN_P = 1.0


class UnknownStatusWarning(UserWarning):
    pass


def p_play_for(statuses, unknown_p=DEFAULT_UNKNOWN_P):
    """Série de statuts -> série de probabilités.

    Statut inconnu (écrit dans le rapport) : 0,5 avec avertissement ; « non listé » : 1 ;
    « rapport non soumis » et « hors rapport » : unknown_p (statut réel inconnu).
    """
    known = set(STATUS_P_PLAY) | {NOT_LISTED} | set(UNCERTAIN)
    unknown = sorted(set(statuses.dropna()) - known)
    if unknown:
        warnings.warn(f"Statut(s) de blessure inconnu(s), probabilité {UNKNOWN_P_PLAY} appliquée : "
                      f"{', '.join(unknown)}", UnknownStatusWarning, stacklevel=2)

    def p(s):
        if s == NOT_LISTED:
            return 1.0
        if s in UNCERTAIN:
            return unknown_p
        return STATUS_P_PLAY.get(s, UNKNOWN_P_PLAY)
    return statuses.map(p)


def is_uncertain(status):
    return status in UNCERTAIN
