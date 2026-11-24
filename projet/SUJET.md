# Projet Usine 4.0

Application au deep learning, ING3 option IA, CY Tech. Enseignant : Franklin Brasa.

## Le besoin

Une aciérie contrôle aujourd'hui ses tôles à l'œil : des opérateurs regardent défiler les tôles et signalent
les défauts de surface. Le contrôle est fatigant, irrégulier d'une équipe à l'autre, et certains défauts
passent. La direction se demande si une caméra et un modèle de vision pourraient faire ce travail, ou au moins
aider les opérateurs.

Elle dispose de photos de défauts déjà annotées (le jeu NEU-DET : six types de défauts, avec leur position sur
l'image), mais certains défauts sont rares sur la ligne : pour ceux-là, on n'aura que quelques images.

La même usine voudrait aussi anticiper les pannes de ses pompes à partir de leurs capteurs, pour intervenir
avant l'arrêt plutôt qu'après.

Elle vous confie l'étude des deux besoins. À la fin, la direction veut savoir s'il faut lancer chaque projet,
et à quelles conditions.

## Votre mission

Concevoir, évaluer et défendre un système d'inspection des tôles et un système de surveillance des pompes,
puis faire une recommandation à l'usine. Les trois parties sont obligatoires.

1. Inspection des tôles : reconnaître le type d'un défaut et le localiser sur la tôle (NEU-DET).
2. Défauts rares : comment traiter un type de défaut dont on n'a qu'une quinzaine d'images ? Vous choisissez
   le type de défaut et le réduisez vous-mêmes à 15 images dans l'entraînement, puis vous mesurez ce que le
   système perd sur ce défaut et ce que vos solutions lui font regagner.
3. Maintenance des pompes (séries temporelles) : détecter les anomalies de fonctionnement d'une pompe à partir
   de ses capteurs (jeu SKAB, anomalies réelles étiquetées), assez tôt pour intervenir. Cette partie ne
   concerne pas les tôles : c'est un second besoin de la même usine.

Le reste est à vous : les modèles, la façon de découper le problème (un classifieur puis un détecteur, un
détecteur seul, autre chose), les métriques, les seuils, les données supplémentaires si vous en trouvez de
publiques. Les dernières étapes des TD 1 à 4 donnent des points de départ ; vous pouvez vous en écarter, à
condition de le justifier. Si vous avez
le temps, testez aussi la robustesse du système d'inspection à des conditions dégradées (flou, éclairage,
bruit) : ce n'est pas exigé, mais cela renforce une recommandation.

## Ce que vous devez montrer

Quelle que soit votre approche, votre étude doit permettre à l'usine de vous faire confiance. Pour chacune
des trois parties :

1. Une solution simple sert de référence : une méthode sans deep learning, réglée sérieusement. Le deep
   learning doit montrer ce qu'il apporte par rapport à elle.
2. Les mesures sont honnêtes : tous les choix se font sur une validation, et le test ne sert qu'une fois, à la
   fin. Les méthodes comparées le sont sur les mêmes données, avec la même métrique.
3. Les choix sont chiffrés. Vous fixez vous-mêmes des hypothèses sur ce que coûte un défaut ou une panne non
   vus et une fausse alerte, vous les écrivez, et vos seuils en découlent.
4. Les limites sont dites : dans quels cas le système se trompe, et ce que cela coûterait.
5. Le travail se relance : un autre ingénieur doit pouvoir reproduire vos résultats à partir de votre dépôt.

## Organisation

Groupes de 3 ou 4. Chaque groupe envoie à l'enseignant sa composition et l'adresse de son dépôt.

Un membre du groupe crée le dépôt du groupe comme il a créé son dépôt personnel (README du dépôt du cours,
Installation, étape 3) : un dépôt GitHub privé et vide, relié au dépôt du cours. Il y ajoute en
collaborateurs les autres membres et l'enseignant (`FranklinB13`). Les autres membres le clonent, puis lancent
`uv sync`.

Les TD restent individuels : chacun peut continuer à les faire dans son dépôt personnel. Ce qui est corrigé,
c'est la copie que chacun en dépose dans son dossier du dépôt du groupe (voir ci-dessous).

## Architecture du dépôt du groupe

```
README.md          le README du groupe
projet/            le projet : code du pipeline et rapport.pdf
prenom_nom/        les TD d'un membre (minuscules, sans accent)
    tp1_vision.py
    tp2_detection.py
    ...
    __marimo__/
        tp1_vision.html
        tp2_detection.html
        ...
prenom_nom/        le membre suivant, de la même façon
```

Le reste (`outils/`, `pyproject.toml`, `uv.lock`, etc.) vient du dépôt du cours : n'y touchez pas.

Dans son dossier, chaque membre dépose tous ses TD : le fichier `.py` complété et le fichier `.html` que marimo
a enregistré dans `__marimo__/` pendant qu'il travaillait. Le `.html` est indispensable : c'est lui qui montre
les résultats obtenus. Pour qu'un TD copié se relance depuis ce dossier, changez une ligne dans sa cellule
d'import :

```
# avant
sys.path.insert(0, str(Path(mo.notebook_dir()) / "outils"))
# après
sys.path.insert(0, str(Path(mo.notebook_dir()).parent / "outils"))
```

Le `README.md` à la racine remplace celui du cours. Il doit permettre à un ingénieur qui découvre le dépôt de
tout relancer sans vous poser de question : les membres du groupe, l'installation, la commande qui lance le
pipeline et sa durée mesurée, où se trouvent les résultats, et une ligne sur l'usage des outils d'IA.

## Rendu

À rendre deux semaines après le Jour 6, à 23 h 59, sur le dépôt du groupe. C'est le dernier commit poussé
avant cette heure qui est corrigé.

- Le code du projet, dans `projet/` : un pipeline Python, pas de notebook. Une commande, donnée dans le
  README, relance tout, des données jusqu'aux résultats sur le test.
- Les sauvegardes nécessaires (poids des modèles retenus, résultats des calculs longs), dans `projet/` :
  `modeles/`, `cache/` et `sorties/` ne sont pas versionnés. À partir de ces sauvegardes, le pipeline doit se
  relancer en moins de 15 minutes sur une machine de l'école. Les entraînements eux-mêmes restent dans le
  code, pour qu'on puisse les refaire.
- Un rapport de 10 pages au maximum, `projet/rapport.pdf`, écrit pour la direction de l'usine : les deux
  problèmes, vos solutions, les résultats sur le test, les limites, et votre recommandation pour chacun
  (déployer, déployer sous conditions, ou ne pas déployer), argumentée. Il se termine par une ligne qui indique
  si vous avez utilisé des outils d'IA, et pour quoi faire.
- Le `README.md` du groupe, à la racine.
- Les TD de chaque membre, dans son dossier `prenom_nom/`.
- Un historique Git régulier, où l'on voit le travail de chaque membre.

## Usage des outils d'IA

Utilisez le moins possible les outils d'IA générative (assistants de code, générateurs de texte), aussi bien
pour le code que pour le rapport. Le projet sert à vérifier que vous savez mener cette étude vous-mêmes.

Le code doit être commenté : un lecteur doit comprendre ce que fait chaque partie et pourquoi. Le rapport doit
être bien rédigé, avec vos propres mots. Une mise en page simple écrite par vous est préférable à un document
soigné produit par une IA.

## Notation

Le projet donne une note de groupe, sur 20. Les TD donnent à chacun un bonus individuel, jusqu'à 2 points,
ajouté à la note du projet. La note finale ne dépasse pas 20.

| Critère | Points |
|---|---|
| Rigueur de la démarche (tôles et défauts rares) | 6 |
| Pertinence des choix (tôles et défauts rares) | 5 |
| Maintenance des pompes (séries temporelles) | 3 |
| Analyse critique et recommandation | 4 |
| Reproductibilité et qualité du code | 2 |
| Bonus individuel : les TD | + 2 |

Pour chaque critère, voici ce qui distingue une note faible, moyenne et haute.

Rigueur de la démarche, tôles et défauts rares (6 points) :

- Faible (0 à 2) : pas de référence sans deep learning, ou des réglages faits sur le test, ou des méthodes comparées sur des données différentes ;
- Moyenne (3 à 4) : référence et protocole présents, mais référence peu réglée ou comparaisons incomplètes ;
- Haute (5 à 6) : référence réglée avec soin, validation et test séparés pour l'inspection et pour les défauts rares, test mesuré une seule fois, comparaisons équitables.

Pertinence des choix, tôles et défauts rares (5 points) :

- Faible (0 à 1) : choix non expliqués, métrique par défaut sans réflexion ;
- Moyenne (2 à 3) : choix expliqués, mais seuils ou hypothèses de coût peu reliés au besoin ;
- Haute (4 à 5) : découpage du problème, modèles et métriques justifiés par le besoin de l'usine, hypothèses de coût écrites et seuils qui en découlent, défauts rares traités avec la même exigence.

Maintenance des pompes, séries temporelles (3 points) :

- Faible (0 à 1) : pas de référence sans deep learning, ou normalisation et seuils calculés avec les données de test, ou résultats mesurés sur l'entraînement ;
- Moyenne (2) : référence et modèle comparés sur le même découpage, statistiques calculées sur l'entraînement seul, mais seuil peu relié aux coûts ou résultats peu analysés ;
- Haute (3) : découpage justifié (ce qui sert de « normal », sur quoi on teste), référence réglée, seuil issu des hypothèses de coût, résultats analysés expérience par expérience (anomalies manquées, fausses alertes, délai avant la première alerte).

Analyse critique et recommandation (4 points) :

- Faible (0 à 1) : résultats présentés sans analyse, recommandation absente ou non argumentée ;
- Moyenne (2 à 3) : erreurs commentées, recommandation présente mais peu reliée aux mesures ;
- Haute (4) : cas d'échec analysés et chiffrés, limites assumées, recommandation qui découle des mesures pour chacun des deux besoins, rapport clair et bien rédigé.

Reproductibilité et qualité du code (2 points) :

- Faible (0) : le dépôt ne se relance pas ;
- Moyenne (1) : il se relance, mais avec des corrections, ou le code est peu commenté, ou le README ne suffit pas ;
- Haute (2) : il se relance tel quel en moins de 15 minutes en suivant le README, le code est commenté, l'architecture du dépôt est respectée, l'historique montre un travail régulier.

Bonus individuel, les TD (jusqu'à 2 points) :

- 0 : TD absents du dossier, ou rendus sans leur `.html` (les résultats ne sont pas visibles) ;
- 1 : la plupart des TD rendus avec leurs sorties, mais des étapes non faites ou des questions sans réponse ;
- 2 : tous les TD rendus avec leurs sorties, étapes faites, réponses aux questions écrites avec vos mots.

Sont pénalisés dans tous les cas : un réglage fait sur le test, des chiffres qui ne viennent d'aucune
exécution, un dépôt qui ne se relance pas. Un résultat modeste bien mesuré vaut mieux qu'un bon score mal
mesuré.

## Données et outils

NEU-DET est à télécharger sur Kaggle et à placer dans `donnees/NEU/` (README du dépôt du cours, section
Données). SKAB se télécharge avec `tp.charger_skab()`. Les fonctions de `outils/tp_utils.py` restent
utilisables dans `projet/` ; les fonctions écrites pendant les TD ne s'importent pas, il faut les recopier.

Pour NEU-DET, utilisez un seul découpage pour tout le projet : `tp.decoupage_neu()` répartit les images en
entraînement, validation et test, et `tp.charger_neu` (classification) comme `tp.voc_vers_yolo(...,
parties=tp.decoupage_neu())` (détection) le respectent, à paramètres égaux. Les étapes 11 des TD 1 et 2 ne
découpent pas les images de la même façon : ne réutilisez pas tel quel le jeu YOLO créé au TD 2.

## Aide et indications

Quelques pièges fréquents, à vérifier avant de rendre :

- Une image ne doit jamais se trouver à la fois dans l'entraînement d'un modèle et dans le test d'un autre,
  si les deux modèles forment un même système.
- Réduire un type de défaut à 15 images ne concerne que l'entraînement : la validation et le test gardent
  toutes leurs images, sinon on ne peut plus mesurer ce que le système vaut sur ce défaut.
- Quand les types de défauts n'ont pas le même nombre d'images, une accuracy globale peut cacher un type
  jamais reconnu : regardez les résultats type par type.
- Sur CPU, un entraînement complet peut durer longtemps : mettez au point vos réglages sur un sous-ensemble
  ou à résolution réduite, lancez les entraînements longs tôt (le soir, pendant une pause), et sauvegardez
  les poids pour ne pas avoir à les refaire.
- Pour SKAB, tout ce qui est calculé à partir des données (moyennes, écarts-types, seuils) se calcule sur la
  partie d'entraînement, puis s'applique tel quel au reste.
- Pour SKAB encore, découpez dans le temps et non au hasard : deux fenêtres voisines se recouvrent presque
  entièrement, et en placer une en entraînement et l'autre en test revient à tester sur l'entraînement.
- Un seuil qui maximise le F1 n'est pas forcément le bon : demandez-vous ce que coûte chaque erreur.
- Ouvrez chaque `.html` de votre dossier dans un navigateur : s'il ne montre pas vos résultats, relancez le TD
  dans marimo pour qu'il soit enregistré à nouveau.
- Avant de rendre, clonez le dépôt du groupe dans un dossier vide et relancez tout en suivant le README :
  c'est ce que fera le correcteur.
