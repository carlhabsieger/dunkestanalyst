# dunkestanalyst

## Projections (`src/model/projections.py`)

Dunkest ne compte qu'un match par joueur et par journée de jeu (une journée peut durer jusqu'à
3 nuits) : `gp`, `pdk` et les moyennes du snapshot ne couvrent donc qu'une partie des matchs.
Par défaut, les projections utilisent TOUS les matchs NBA joués (logs nba_api, score Dunkest
recalculé par `src/scoring.py`) :

- n = matchs NBA joués (minutes > 0), avg = moyenne du score Dunkest sur ces matchs ;
- prior = régression linéaire avg ~ prix sur les joueurs à n >= 3 ;
- proj = (n x avg + k x prior) / (n + k), avec k = `SHRINK_K` = 4 (calibré sur `early_season` : coefficient
  0,51 sur 4 matchs, soit 4 / (4 + k) = 0,51).

Les noms du snapshot sont rapprochés de ceux des logs par `normalize_name` (casse, accents et
ponctuation ignorés, suffixes Jr/II/III conservés). Les joueurs non rapprochés sont listés à chaque
lancement ; ils reçoivent proj = prior et n = 0, et ne sont donc pas proposés.

```
python -m src.data.box_scores          # met à jour data/player_logs.csv (requis pour la source nba)
python -m src.model.compare_proj       # plus grands écarts entre les deux sources, écart moyen par poste
python -m src.model.compare_proj --prior price_min   # expérience : a priori prix + minutes/match
```

`--prior price_min` n'est branché nulle part ailleurs : il est ajusté sur les mêmes joueurs que ceux
projetés et ne prouve pas qu'il prédit mieux que l'a priori prix seul.

`src/optimizer.py` et `src/buy.py` acceptent `--proj-source nba` (défaut) ou `dunkest` (ancienne
logique pdk/gp, pour comparaison), et `--logs` pour un autre fichier de logs. Avec la source nba,
la colonne affichée « matchs NBA » (et le filtre `--min-gp`) porte sur les matchs NBA.

### Limites des projections

- Le score calculé depuis nba_api n'inclut pas le bonus de cinq majeur (+1) ni les contres subis
  (-0,5) : léger biais pour les titulaires.
- Le premier match compté par Dunkest peut différer de la moyenne tous matchs (fatigue sur matchs
  consécutifs) : non mesuré.
- Les projections ignorent toujours le calendrier et l'injury report ; le coach n'est pas modélisé.

## Blessures (`src/data/injuries.py`, `src/model/injury_probs.py`)

Le rapport officiel des blessures NBA (package `nbainjuries`) est lu à chaque lancement de
l'optimiseur et de `buy.py` : on remonte d'heure en heure (heure de l'Est, au plus 36 essais)
jusqu'au dernier rapport publié, sauvegardé dans `data/injuries_latest.csv`. L'heure du rapport
(ET et Paris) et son ancienneté sont affichées.

Chaque statut donne une probabilité de jouer (`STATUS_P_PLAY`) : Out 0, Doubtful 0,15,
Questionable 0,5, Probable 0,85, Available 1. Un statut écrit mais inconnu reçoit 0,5 avec un
avertissement. Un joueur sans ligne dans le rapport prend un statut selon son équipe :

| Statut | Cas | p_play |
|---|---|---|
| non listé | l'équipe a soumis son rapport pour son premier match du rapport | 1 |
| rapport non soumis ? | ce premier match est « NOT YET SUBMITTED » | `--unknown-p` (défaut 1) |
| hors rapport ? | l'équipe n'a aucun match couvert par le rapport | `--unknown-p` (défaut 1) |

Le « ? » est affiché à côté du nom, et chaque sortie se termine par le nombre de joueurs proposés
au statut inconnu. La projection devient proj = p_play x proj_if_play.

### Historique des rapports

Chaque rapport récupéré est archivé dans `data/injuries/report_AAAAMMJJ_HHMM_ET.csv` (jamais
réécrit) ; les outils lisent tous les rapports archivés plus le dernier. Pour un joueur et le premier
match de son équipe dans la fenêtre :

- on prend son statut dans le rapport le plus récent où il figure, pour un match antérieur ou égal ;
- ce statut est annulé (« non listé ») si un rapport plus récent, soumis par son équipe (lignes de
  l'équipe, pas « NOT YET SUBMITTED »), ne le mentionne plus ;
- sinon il est conservé avec son âge : motif « [dernier statut connu, il y a N j, rapport du 01/11
  04:30 ET] » ; un Out garde p = 0, un autre statut garde son p_play ;
- au-delà de `--stale-days` jours (défaut 7, compté depuis le rapport), il devient « ancien ? »
  avec p = `--unknown-p`.

```
python -m src.data.injuries --at "2026-11-01 04:30"   # archive un rapport passé (heure ET)
```

Limite : une équipe qui soumet son rapport sans aucun blessé n'apparaît pas forcément dans le
rapport ; dans ce cas, l'ancien statut n'est pas annulé.

### Calendrier (`src/data/schedule.py`)

`--dates 2026-11-01,2026-11-02` (optimiseur et `buy.py`) exclut les équipes sans match à ces dates
et affiche le match de chaque joueur proposé (« 02/11 @HOU 8:30 pm ET »). Source : nba_api
`ScheduleLeagueV2` (saison entière en un appel, sauvegardée dans `data/schedule_2026-27.csv`).
Règle Dunkest : dans une fenêtre de plusieurs dates, seul le **premier match** de chaque équipe
compte. La colonne match et le statut blessure se rapportent à ce match (« (+1 match non compté) »
signale un second match dans la fenêtre). Un joueur listé seulement pour un match antérieur garde
son dernier statut, avec son âge dans le motif (« [statut du 01/11, 3 j avant] »).

`--window-from-dunkest` n'existe pas : l'API de config Dunkest (`leagues/1/config`) ne donne ni les
dates des journées ni le nombre de tours de la prochaine journée (seulement `num_rounds` et
`started_at` du tour en cours), donc la fenêtre ne peut pas être calculée sans hypothèse non vérifiée.

```
python -m src.data.schedule 2026-11-02     # matchs du jour et heures
```

Note : la colonne « Game Time » du rapport officiel n'a pas d'AM/PM (« 07:30 (ET) » pour
19 h 30, vérifié dans le texte du PDF) ; elle n'est pas utilisée, l'heure vient du calendrier.

```
python -m src.data.injuries                         # dernier rapport, statuts, non rapprochés
python -m src.optimizer --coach-cost 9              # blessures prises en compte (défaut --injuries on)
python -m src.buy --buy "T:G@8-12" --credits 12 --min-p-play 0.6
```

- `--injuries on|off` (défaut on) : avec on, les joueurs Out ne sont jamais proposés. Si le rapport
  est introuvable, l'outil s'arrête et propose `--injuries off` (pas de bascule silencieuse).
- `--min-p-play` (défaut 0) : exclut les candidats dont la probabilité de jouer est sous le seuil.

### Limites des blessures

- Les probabilités par statut sont provisoires (non calibrées sur l'historique des rapports).
- L'espérance p x proj ignore la possibilité de remplacer un titulaire absent entre deux tours.
- Le rapport est mis à jour tard dans la journée, et le marché Dunkest peut fermer avant les statuts
  finaux.
- Le rapport ne couvre que les matchs à venir proches : « rapport non soumis » et « hors rapport »
  gardent p = 1 par défaut (statut réel inconnu, signalé par « ? »).
- La soumission est jugée sur le premier match de l'équipe dans le rapport (une équipe peut avoir
  soumis pour un match et pas pour le suivant).
- La répartition des minutes des coéquipiers d'un absent n'est pas encore modélisée.
- Java est requis (lecture du PDF par tabula) : `sudo apt install default-jre`.

## Gestion des équipes (`src/team.py`)

Un fichier par équipe, `teams/<nom>.json`, versionné et modifiable à la main :

```json
{
  "players": ["Paolo Banchero", "Ty Jerome", "Jalen Smith", "Cedric Coward", "Zach LaVine",
              "Duncan Robinson", "Paul Reed", "Jonathan Kuminga", "Ja'Kobe Walter", "Jaxson Hayes"],
  "cash": 3.8,
  "coach": "Joe Mazzulla (BOS)",
  "excluded": ["Nikola Jokić"],
  "excluded_from_teams": ["equipe2"],
  "price_overrides": {"Joueur Absent": {"cr": 9.5, "position": "F", "team_code": "BOS"}}
}
```

- `excluded` : joueurs pris par d'autres ; `excluded_from_teams` : autres équipes de l'utilisateur dont
  les joueurs ne sont pas proposés à l'achat.
- `price_overrides` : prix d'un joueur absent du snapshot (un nombre suffit s'il est dans le snapshot ;
  sinon un objet avec `position` et `team_code`, nécessaires aux quotas).
- `coach` : texte libre, non modélisé ; s'il se termine par un code équipe entre parenthèses
  (« Nom (BOS) »), il compte dans la limite de 3 par équipe NBA.

```
python -m src.team list
python -m src.team init NOM [--players "A,B,..."] [--cash 1.4]
python -m src.team show NOM [--dates a,b,c] [--html rapport.html]
python -m src.team add NOM "Joueur" | remove NOM "Joueur" | swap NOM "Ancien" "Nouveau"
python -m src.team set NOM --cash 1.4 --coach "..." --exclude "A,B" --exclude-teams "autre" --price "Nom=9.5"
python -m src.team lineup NOM --dates a,b,c        # meilleure composition sans transfert (gratuit)
python -m src.team suggest NOM --dates a,b,c [--max-transfers 3] [--horizon 1] [--min-net-gain 1.0] [--allow-out]
```

Les noms se résolvent sans accents ni casse, avec la table d'alias et une correspondance approchée ;
un nom ambigu affiche les candidats sans rien choisir. `add`, `swap` et `init` refusent plus de
10 joueurs, plus de 4 G / 4 F / 2 C et plus de 3 joueurs d'une équipe NBA. Chaque écriture copie
l'ancienne version dans `teams/.backup/` (à ignorer dans git : non fait, à décider).

**Règle des 5 points.** Changer de formation, de cinq majeur, de 6e homme ou de capitaine est gratuit.
Chaque joueur transféré coûte 5 points, retranchés du score de la journée suivante. `suggest` compare,
pour k = 0 à `--max-transfers`, le meilleur effectif avec au plus k joueurs nouveaux (objectif : points
attendus moins 5 par transfert) et ne recommande un transfert que si le gain net atteint
`--min-net-gain` (défaut 1). Quand le meilleur plan utilise moins de k transferts, la meilleure option à
exactement k transferts est aussi affichée (gain net négatif). Si le marché n'est pas ouvert (statut
Dunkest différent de `trades_open`), un avertissement s'affiche sans bloquer.

### Limites de la gestion des équipes

- Projections bruitées (peu de matchs) : un écart de moins de 2 points entre deux options est dans le bruit.
- Prix de vente supposé égal au prix actuel du snapshot (NON vérifié) : budget = prix actuels + cash.
- `--horizon H` est optimiste : les achats comptent H fois, le coût de 5 points une seule fois, et les
  joueurs gardés ne sont pas recomptés.
- Coach non modélisé ; redistribution des minutes des absents absente (phase 3).
- Un joueur possédé Out, sans match dans la fenêtre ou sans match NBA compte pour 0.
- Le statut de blessure suit l'historique des rapports archivés (voir « Historique des rapports ») :
  un Out reste Out tant qu'un rapport plus récent de son équipe ne l'a pas retiré.

## Demande précise (`src/buy.py`)

Trouve les meilleurs joueurs à acheter pour quelques places précises de l'équipe, en tenant compte
du rôle de chaque place : un titulaire ou le sixième homme compte 1,0 x proj, un remplaçant
0,5 x proj, le capitaine (option `--captain`) 1,5 x proj.

```
python -m src.buy --buy "SLOT,SLOT,..." --credits BUDGET [options]
```

Chaque slot s'écrit `[rôle:]poste[@min-max]` :

| Élément | Valeurs |
|---|---|
| rôle | `T` titulaire (défaut), `R` remplaçant (banc, 50 %), `6` sixième homme (100 %) |
| poste | `G`, `F`, `C` ou `*` (tous postes) |
| prix | `@8-12`, `@-5` (max seul), `@8-` (min seul), `@9.5-9.5` ou `@9.5` (prix exact) |

Options :

- `--credits` : budget total pour l'ensemble des slots demandés (obligatoire).
- `--owned "Nom A,Nom B"` : joueurs déjà dans l'équipe. Ils ne sont pas proposés et comptent dans
  la limite de 3 par équipe NBA et dans les quotas de postes (4 G, 4 F, 2 C).
- `--owned-roles "Nom A:T,Nom B:R"` : rôle des joueurs possédés, pour qu'ils comptent aussi dans
  les quotas de rôles (5 T, 4 R, un 6e homme) et la formation. Un nom cité ici est considéré possédé.
- `--exclude "Nom C"` : joueurs pris par des amis, jamais proposés.
- `--captain` : autorise un capitaine parmi les slots `T`.
- `--min-gp` (défaut 1), `--alts` (nombre de combinaisons, défaut 3), `--snapshot` (CSV, défaut le
  plus récent), `--find "nom"` (recherche un joueur).

Les noms se comparent sans tenir compte des accents ni de la casse. Les titulaires demandés (et les
possédés marqués `T`) doivent entrer dans au moins une formation (2-2-1, 1-2-2, 2-1-2, 1-3-1,
3-1-1) ; sinon l'outil s'arrête en indiquant la formation qu'il faudrait.

Exemples :

```
# Un meneur titulaire entre 8 et 12 crédits et un ailier remplaçant à 5 max, capitaine possible
python -m src.buy --buy "T:G@8-12,R:F@-5" --credits 17 --captain

# Un sixième homme de n'importe quel poste à exactement 9,5 crédits
python -m src.buy --buy "6:*@9.5-9.5" --credits 9.5

# Deux titulaires et un 6e homme, en tenant compte de l'équipe actuelle et des joueurs des amis
python -m src.buy --buy "T:*,T:C,6:*" --credits 40 --owned "Domantas Sabonis" \
    --owned-roles "Keyonte George:T" --exclude "Yaxel Lendeborg" --captain
```

### Limites connues

- `proj` est une moyenne de saison lissée sur 3 à 4 matchs ; pas de calendrier ni d'injury report
  (un joueur dont l'équipe ne joue pas peut être proposé).
- Le coach n'est pas modélisé.
- La correspondance formation = (G, F, C) parmi les titulaires n'est pas vérifiée.
- Le capitaine n'est choisi que parmi les slots `T` demandés, pas parmi les titulaires possédés.
