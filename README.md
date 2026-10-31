# dunkestanalyst

## Projections (`src/model/projections.py`)

Dunkest ne compte qu'un match par joueur et par journée de jeu (une journée peut durer jusqu'à
3 nuits) : `gp`, `pdk` et les moyennes du snapshot ne couvrent donc qu'une partie des matchs.
Par défaut, les projections utilisent TOUS les matchs NBA joués (logs nba_api, score Dunkest
recalculé par `src/scoring.py`) :

- n = matchs NBA joués (minutes > 0), avg = moyenne du score Dunkest sur ces matchs ;
- prior = régression linéaire avg ~ prix sur les joueurs à n >= 3 ;
- proj = (n x avg + k x prior) / (n + k), avec k = `SHRINK_K` = 3 (valeur provisoire).

Les noms du snapshot sont rapprochés de ceux des logs par `normalize_name` (casse, accents et
ponctuation ignorés, suffixes Jr/II/III conservés). Les joueurs non rapprochés sont listés à chaque
lancement ; ils reçoivent proj = prior et n = 0, et ne sont donc pas proposés.

```
python -m src.data.box_scores          # met à jour data/player_logs.csv (requis pour la source nba)
python -m src.model.compare_proj       # plus grands écarts entre les deux sources, écart moyen par poste
```

`src/optimizer.py` et `src/buy.py` acceptent `--proj-source nba` (défaut) ou `dunkest` (ancienne
logique pdk/gp, pour comparaison), et `--logs` pour un autre fichier de logs. Avec la source nba,
la colonne affichée « matchs NBA » (et le filtre `--min-gp`) porte sur les matchs NBA.

### Limites des projections

- Le score calculé depuis nba_api n'inclut pas le bonus de cinq majeur (+1) ni les contres subis
  (-0,5) : léger biais pour les titulaires.
- Le premier match compté par Dunkest peut différer de la moyenne tous matchs (fatigue sur matchs
  consécutifs) : non mesuré.
- Les projections ignorent toujours le calendrier et l'injury report ; le coach n'est pas modélisé.

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
