import marimo

__generated_with = "0.25.0"
app = marimo.App(
    width="medium",
    app_title="TD 5b - Apprentissage par renforcement",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _():
    import copy
    import sys
    import time
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn

    sys.path.insert(0, str(Path(mo.notebook_dir()) / "outils"))
    import tests_tp5 as tests
    import tp_utils as tp

    plt.rcParams["figure.dpi"] = 90
    # Budgets d'interaction (réduits quand on vérifie seulement que le notebook s'exécute)
    N_PAS_TABLE = 6_000 if tp.MODE_TEST else 60_000
    N_PAS = 5_000 if tp.MODE_TEST else 30_000
    return N_PAS, N_PAS_TABLE, copy, mo, nn, np, plt, tests, time, torch, tp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TD 5b — Apprentissage par renforcement

    **Module Application au deep learning · ING3 IA · durée : 2h15 (après-midi)**

    Dans tous les TD précédents, on disposait d'exemples avec la bonne réponse. Cet après-midi, personne ne donne
    la bonne réponse : un **agent** agit, observe ce qui se passe, reçoit une **récompense**, et doit trouver seul
    une stratégie. Vous allez faire tenir un mât en équilibre sur un chariot de quatre façons : avec une règle
    écrite à la main, avec une table (Q-learning), avec un réseau (DQN), puis avec une bibliothèque (PPO).
    Et vous verrez sur un second problème pourquoi cette famille de méthodes est la plus délicate du module.

    **Rappel du fonctionnement du notebook**

    - Exécuter une cellule : bouton de lecture à sa droite, ou Ctrl + Entrée (Cmd + Entrée sur Mac).
    - Cellules **À compléter** : remplacez le `raise NotImplementedError` par votre code, puis lisez la cellule
      de tests juste en dessous. Une étape se débloque quand les tests de l'étape précédente sont réussis.
    - Les calculs longs ne partent qu'au clic sur leur bouton.
    - Le code de certaines cellules fournies est replié : vous pouvez l'afficher depuis le menu de la cellule.
    - Cellules **Votre réponse** : double-cliquez sur le texte pour l'éditer.
    - Pour tester du code librement, ajoutez une cellule avec le bouton + qui apparaît entre deux cellules.
    - **Particularité de marimo** : une variable ne peut être définie que dans **une seule cellule**. Dans vos
      cellules de test, donnez à vos variables un nom qui commence par un tiret bas (`_x`, `_modele`) : elles
      restent alors locales à la cellule. Sinon, marimo affiche une erreur du type
      *« This cell redefines variables from other cells »* : il suffit de renommer la variable.

    | Étape | Contenu | Durée indicative |
    |---|---|---|
    | 1 | Le cours en bref et premier contact avec l'environnement (vu en cours : à relire si besoin) | — |
    | 2 | Deux politiques sans apprentissage : le hasard et une règle écrite à la main | 15 min |
    | 3 | Baseline classique : le Q-learning dans une table | 35 min |
    | 4 | DQN : remplacer la table par un réseau | 35 min |
    | 5 | PPO avec une bibliothèque, et comparaison | 20 min |
    | 6 | Une limite à voir de ses yeux : la récompense rare | 15 min |
    | 7 | Fiche de décision et bilan de la journée | 15 min |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 1 — Le cours en bref

    ### 1.1 Rappel : ce qui change par rapport à l'apprentissage supervisé

    | | Apprentissage supervisé (TD 1 à 4) | Apprentissage par renforcement |
    |---|---|---|
    | Ce qu'on reçoit | des exemples avec la bonne réponse | une **récompense** (un nombre) après chaque action |
    | D'où viennent les données | d'un jeu de données fixé à l'avance | de l'agent lui-même : ce qu'il fait décide de ce qu'il verra |
    | Ce qu'on optimise | l'erreur sur chaque exemple | la somme des récompenses **futures** |

    Un réseau de neurones reste ce qu'il a toujours été : une fonction réglable par descente de gradient.
    Ce qui change, c'est la cible qu'on lui donne.

    ### 1.2 Vocabulaire

    - **État** $s$ : ce que l'agent observe. **Action** $a$ : ce qu'il décide. **Récompense** $r$ : le signal reçu.
    - **Politique** $\pi$ : la règle qui choisit l'action à partir de l'état. C'est elle qu'on cherche.
    - **Épisode** : une partie, du début jusqu'à l'échec ou la fin du temps imparti.
    - **Retour** : la somme des récompenses à venir, $r_0 + \gamma r_1 + \gamma^2 r_2 + \dots$ Le facteur
      $\gamma$ (entre 0 et 1, souvent 0,99) donne moins de poids aux récompenses lointaines.
    - **Valeur** $Q(s, a)$ : le retour qu'on peut espérer si on fait l'action $a$ dans l'état $s$, puis qu'on agit
      au mieux. Si on connaît $Q$, la politique est immédiate : choisir l'action de plus grande valeur.
    - **Exploration** : essayer des actions qu'on croit moins bonnes, pour découvrir si on se trompe.

    ### 1.3 Trois familles d'algorithmes

    | Famille | Idée | Exemples |
    |---|---|---|
    | Fondée sur la valeur | apprendre $Q$, puis choisir l'action de plus grande valeur | Q-learning, **DQN** |
    | Fondée sur la politique | régler directement la politique : rendre plus probables les actions qui ont rapporté | REINFORCE |
    | Acteur-critique | les deux à la fois : une politique (l'acteur) et une valeur qui la juge (le critique) | **PPO**, **SAC** (actions continues, robotique) |

    **Le lien avec le reste du module.** DQN a appris à jouer aux jeux Atari à partir des pixels de l'écran :
    c'est un CNN (TD 1) qui lit l'image et produit une valeur $Q$ par touche de la manette. Le réseau ne change
    pas de nature ; c'est la façon de l'entraîner qui est nouvelle.

    **Où on le rencontre en entreprise.** Robotique, pilotage de procédés et d'installations énergétiques,
    planification, jeux, recommandation séquentielle. Presque toujours avec un **simulateur** : apprendre par
    essais et erreurs sur une vraie machine coûte cher et peut être dangereux.

    ### 1.4 L'environnement du TD : CartPole

    Un chariot roule sur un rail et porte un mât articulé. À chaque pas, l'agent pousse le chariot à gauche
    (action 0) ou à droite (action 1). Il reçoit **+1 par pas** tant que le mât tient. L'épisode s'arrête si le
    mât penche de plus de 12 degrés, si le chariot sort du rail, ou au bout de 500 pas. Le score maximal est
    donc **500**. L'état est fait de 4 nombres : position et vitesse du chariot, angle et vitesse angulaire du mât.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, tp):
    _rng = np.random.default_rng(0)
    _total, _images = tp.jouer_episode(lambda obs: _rng.integers(2), graine=0)
    mo.vstack([
        mo.Html(tp.balise_gif(_images, duree_ms=120)),
        mo.md(f"Un épisode avec des actions tirées au hasard (ralenti) : le mât tombe après **{int(_total)}** pas."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Tout environnement de la bibliothèque **Gymnasium** se pilote avec deux fonctions : `reset()` démarre un
    épisode et renvoie le premier état ; `step(action)` joue un pas et renvoie l'état suivant, la récompense,
    et deux indicateurs de fin (`termine` : échec ; `tronque` : temps écoulé). Exécutez la cellule, puis changez
    l'action jouée pour voir l'effet sur l'état.
    """)
    return


@app.cell
def _(mo, tp):
    _env = tp.creer_cartpole()
    _obs, _ = _env.reset(seed=0)
    _lignes = [{"pas": 0, "action": "", "position": f"{_obs[0]:+.3f}", "vitesse": f"{_obs[1]:+.3f}",
                "angle (rad)": f"{_obs[2]:+.3f}", "vitesse angulaire": f"{_obs[3]:+.3f}", "récompense": "", "terminé": ""}]
    for _pas in range(1, 9):
        _action = 1                                   # 1 = pousser à droite, 0 = pousser à gauche
        _obs, _recompense, _termine, _tronque, _ = _env.step(_action)
        _lignes.append({"pas": _pas, "action": _action, "position": f"{_obs[0]:+.3f}", "vitesse": f"{_obs[1]:+.3f}",
                        "angle (rad)": f"{_obs[2]:+.3f}", "vitesse angulaire": f"{_obs[3]:+.3f}",
                        "récompense": _recompense, "terminé": "oui" if _termine else "non"})
        if _termine:
            break
    tp.tableau(_lignes, mo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 2 — Deux politiques sans apprentissage

    Avant d'entraîner quoi que ce soit, deux repères : le **hasard**, et une **règle écrite à la main**, comme
    le ferait un automaticien. Une politique est simplement une fonction qui reçoit l'état et renvoie une action.
    """)
    return


@app.cell
def _(mo, np, tp):
    _rng = np.random.default_rng(0)

    def politique_hasard(obs):
        return int(_rng.integers(2))

    def politique_regle(obs):
        position, vitesse, angle, vitesse_angulaire = obs
        return int(angle > 0)                         # pousser du côté où le mât penche

    score_hasard = tp.evaluer_politique(politique_hasard, 20)
    score_regle_angle = tp.evaluer_politique(politique_regle, 20)
    mo.md(f"Score moyen sur 20 épisodes : hasard **{score_hasard:.0f}**, "
          f"règle « pousser du côté où le mât penche » **{score_regle_angle:.0f}** (maximum possible : 500).")
    return score_hasard, score_regle_angle


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La règle réagit trop tard : quand l'angle change de signe, le mât a déjà pris de la vitesse. Un automaticien
    ajouterait un terme d'**anticipation** : pousser à droite si `angle + k × vitesse_angulaire > 0`.
    Faites varier `k`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    curseur_k = mo.ui.slider(0, 1, step=0.05, value=0, label="Poids k de la vitesse angulaire", show_value=True)
    curseur_k
    return (curseur_k,)


@app.cell(hide_code=True)
def _(curseur_k, mo, tp):
    _k = curseur_k.value
    _score = tp.evaluer_politique(lambda obs: int(obs[2] + _k * obs[3] > 0), 20)
    _total, _images = tp.jouer_episode(lambda obs: int(obs[2] + _k * obs[3] > 0), graine=0, max_images=150)
    mo.vstack([
        mo.md(f"Avec k = {_k:.2f} : score moyen **{_score:.0f}** sur 20 épisodes. Les 150 premiers pas d'un épisode :"),
        mo.Html(tp.balise_gif(_images, duree_ms=40)),
    ])
    return


@app.cell
def _(mo, tp):
    # La règle retenue comme baseline pour la suite du TD
    def politique_regle_anticipee(obs):
        return int(obs[2] + 0.25 * obs[3] > 0)

    score_regle_anticipee = tp.evaluer_politique(politique_regle_anticipee, 20)
    mo.md(f"Règle retenue pour la suite, avec k = 0,25 : score moyen **{score_regle_anticipee:.0f}** sur 20 épisodes.")
    return (score_regle_anticipee,)


@app.cell
def _(mo):
    mo.md(r"""
    **Votre réponse :** à partir de quelle valeur de `k` la règle atteint-elle le score maximal ? Qu'a-t-il fallu
    **savoir** sur le problème pour écrire cette règle ? Gardez votre réponse en tête : elle servira à l'étape 7.

    ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 3 — Baseline classique : le Q-learning dans une table

    Supposons maintenant qu'on ne sache **rien** de la physique du problème. Le Q-learning (1989, aucun réseau de
    neurones) apprend la valeur $Q(s, a)$ de chaque couple état-action dans un simple **tableau**, à partir des
    seules récompenses.

    Un tableau exige un nombre fini d'états. On découpe donc chacun des 4 nombres de l'état en intervalles :
    6 pour la position, 6 pour la vitesse, 8 pour l'angle, 8 pour la vitesse angulaire. La fonction
    `tp.discretiser` fait ce découpage : elle transforme les 4 nombres en 4 indices de case.
    """)
    return


@app.cell
def _(np, tp):
    _obs = np.array([0.3, -0.5, 0.02, 1.2])           # position, vitesse, angle, vitesse angulaire
    _etat = tp.discretiser(_obs)
    _Q = np.zeros(tp.FORME_TABLE_Q)
    print("état observé     :", _obs)
    print("case de la table :", _etat)
    print("forme de la table Q :", _Q.shape, "soit", _Q.size, "valeurs")
    print("valeurs Q de cet état :", _Q[_etat], "(une par action, toutes nulles au départ)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### À compléter — TODO 1 : explorer ou exploiter

    Si l'agent choisit toujours l'action qu'il croit la meilleure, il ne découvrira jamais qu'une autre était
    préférable. La politique **epsilon-gloutonne** règle ce dilemme :

    - avec la probabilité `epsilon`, tirer une action **au hasard** (exploration) ;
    - sinon, choisir l'action de plus grande valeur (exploitation).

    Entrées : `q_valeurs`, un tableau NumPy contenant une valeur par action ; `epsilon` ; `rng`, le générateur
    aléatoire à utiliser (`rng.random()` renvoie un nombre entre 0 et 1, `rng.integers(n)` un entier entre 0 et n − 1).
    Sortie : le numéro de l'action, converti en `int`.
    """)
    return


@app.cell
def _(np):
    def choisir_action(q_valeurs, epsilon, rng):
        """Politique epsilon-gloutonne. Renvoie le numéro (int) de l'action choisie."""
        # TODO 1
        raise NotImplementedError("À compléter")
    return (choisir_action,)


@app.cell(hide_code=True)
def _(choisir_action, mo, tests, tp):
    ok_todo1, _affichage = tp.verifier(tests.tests_epsilon(choisir_action), mo)
    _affichage
    return (ok_todo1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### À compléter — TODO 2 : la mise à jour du Q-learning

    L'agent était dans l'état $s$, a joué $a$, a reçu $r$ et se retrouve dans l'état $s'$. Ce qu'il vient de vivre
    lui donne une nouvelle estimation de $Q(s, a)$, la **cible** :

    $$\text{cible} = r + \gamma \, \max_{a'} Q(s', a')$$

    c'est-à-dire la récompense reçue, plus ce qu'on espère obtenir ensuite en agissant au mieux. Si l'épisode est
    terminé (`termine` vaut `True`), il n'y a pas de suite : la cible est simplement $r$.
    On déplace alors la valeur de la table d'une fraction $\alpha$ vers cette cible :

    $$Q(s, a) \leftarrow Q(s, a) + \alpha \, \big(\text{cible} - Q(s, a)\big)$$

    La fonction modifie `Q` sur place et ne renvoie rien. L'état est un tuple d'indices : on lit les valeurs de
    l'état suivant avec `Q[etat_suivant]` et on accède à la case à modifier avec `Q[etat][action]`.
    """)
    return


@app.cell
def _(np):
    def mise_a_jour_q(Q, etat, action, recompense, etat_suivant, termine, alpha, gamma):
        """Applique une étape de Q-learning à la table Q (modifiée sur place)."""
        # TODO 2
        raise NotImplementedError("À compléter")
    return (mise_a_jour_q,)


@app.cell(hide_code=True)
def _(mise_a_jour_q, mo, tests, tp):
    ok_todo2, _affichage = tp.verifier(tests.tests_q_learning(mise_a_jour_q), mo)
    _affichage
    return (ok_todo2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### La boucle d'apprentissage (fournie, à lire)

    Elle utilise vos deux fonctions. Repérez les trois moments : agir, observer, mettre à jour.
    L'exploration `epsilon` part de 1 (tout au hasard) et descend jusqu'à 0,05 à la moitié de l'entraînement.
    """)
    return


@app.cell
def _(choisir_action, mise_a_jour_q, np, tp):
    def entrainer_tabulaire(n_pas=60_000, graine=0, alpha=0.1, gamma=0.99):
        """Q-learning tabulaire sur CartPole. Renvoie (table Q, historique [(pas de fin d'épisode, score)])."""
        rng = np.random.default_rng(graine)
        env = tp.creer_cartpole()
        Q = np.zeros(tp.FORME_TABLE_Q)                    # 6 x 6 x 8 x 8 états, 2 actions
        historique, pas = [], 0
        while pas < n_pas:
            obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
            etat, total, fini = tp.discretiser(obs), 0.0, False
            while not fini:
                epsilon = max(0.05, 1 - pas / (0.5 * n_pas))
                action = choisir_action(Q[etat], epsilon, rng)                               # agir
                obs, recompense, termine, tronque, _ = env.step(action)                      # observer
                suivant = tp.discretiser(obs)
                mise_a_jour_q(Q, etat, action, recompense, suivant, termine, alpha, gamma)   # mettre à jour
                etat, total, pas, fini = suivant, total + recompense, pas + 1, termine or tronque
            historique.append((pas, total))
        return Q, historique

    def politique_table(Q):
        """La politique qui suit la table, sans exploration."""
        return lambda obs: int(np.argmax(Q[tp.discretiser(obs)]))

    return entrainer_tabulaire, politique_table


@app.cell(hide_code=True)
def _(mo):
    bouton_tabulaire = mo.ui.run_button(label="Entraîner le Q-learning tabulaire (3 essais de 60 000 pas)")
    bouton_tabulaire
    return (bouton_tabulaire,)


@app.cell
def _(
    N_PAS_TABLE,
    bouton_tabulaire,
    entrainer_tabulaire,
    mo,
    np,
    ok_todo1,
    ok_todo2,
    plt,
    politique_table,
    time,
    tp,
):
    tp.bloquer_si(not (ok_todo1 and ok_todo2), mo, "Terminez les TODO 1 et 2 pour débloquer l'entraînement.")
    _cache = tp.charger_cache("tp5_tabulaire")
    mo.stop(not (tp.lance(bouton_tabulaire) or _cache is not None), mo.md("Cliquez pour lancer (une vingtaine de secondes)."))
    if tp.lance(bouton_tabulaire):
        _courbes, _scores, _visites, _debut = [], [], [], time.perf_counter()
        for _graine in mo.status.progress_bar(range(3), title="Q-learning tabulaire"):
            _Q, _historique = entrainer_tabulaire(N_PAS_TABLE, _graine)
            _courbes.append(tp.lisser_par_pas(_historique, N_PAS_TABLE))
            _scores.append(tp.evaluer_politique(politique_table(_Q)))
            _visites.append(float((_Q != 0).any(axis=-1).mean()))
        tp.sauver_cache("tp5_tabulaire", courbes=np.array(_courbes), scores=np.array(_scores), visites=np.array(_visites),
                        duree=np.array((time.perf_counter() - _debut) / 3), table=_Q)
        _cache = tp.charger_cache("tp5_tabulaire")
    resultats_table = _cache
    _x = np.arange(1, _cache["courbes"].shape[1] + 1) * 500
    _fig, _ax = plt.subplots(figsize=(6, 2.8))
    for _i, _c in enumerate(_cache["courbes"]):
        _ax.plot(_x, _c, label=f"essai {_i + 1}")
    _ax.set_xlabel("pas d'interaction avec l'environnement")
    _ax.set_ylabel("score (10 derniers épisodes)")
    _ax.set_ylim(0, 510)
    _ax.legend(fontsize=7)
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(f"Score final sans exploration (10 épisodes) : **{', '.join(f'{s:.0f}' for s in _cache['scores'])}** selon l'essai. "
              f"Durée d'un essai : {float(_cache['duree']):.1f} s. "
              f"Part des {int(np.prod(tp.FORME_TABLE_Q[:-1]))} états de la table visités au moins une fois : "
              f"{_cache['visites'].mean():.0%}."),
    ])
    return (resultats_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expérience : le rôle de $\gamma$ et de $\alpha$

    Relancez un essai en changeant le facteur $\gamma$ (le poids donné au futur) ou le pas $\alpha$.
    Essayez en particulier $\gamma = 0{,}5$ : l'agent ne regarde alors que quelques pas devant lui.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    reglages_table = mo.ui.dictionary({
        "gamma": mo.ui.slider(0.5, 0.99, step=0.01, value=0.99, label="gamma", show_value=True),
        "alpha": mo.ui.slider(0.01, 1.0, step=0.01, value=0.1, label="alpha", show_value=True),
    }).form(submit_button_label="Lancer un essai avec ces réglages")
    reglages_table
    return (reglages_table,)


@app.cell(hide_code=True)
def _(
    N_PAS_TABLE,
    entrainer_tabulaire,
    mo,
    ok_todo1,
    ok_todo2,
    politique_table,
    reglages_table,
    tp,
):
    mo.stop(reglages_table.value is None or not (ok_todo1 and ok_todo2), mo.md("Réglez les curseurs puis validez."))
    _Q, _ = entrainer_tabulaire(N_PAS_TABLE, graine=0, alpha=reglages_table.value["alpha"],
                                gamma=reglages_table.value["gamma"])
    mo.md(f"gamma = {reglages_table.value['gamma']:.2f}, alpha = {reglages_table.value['alpha']:.2f} : "
          f"score final **{tp.evaluer_politique(politique_table(_Q)):.0f}**.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Votre réponse :**

    1. Sans rien connaître de la physique, la table fait-elle mieux que le hasard ? Mieux que la règle écrite à la main ?
    2. Les trois essais ne diffèrent que par leurs tirages aléatoires. Que pensez-vous de l'écart entre leurs scores ?
    3. Avec ce découpage, la table a 2 304 états. Combien en aurait-elle si l'état comptait 20 nombres découpés
       chacun en 8 intervalles ? Et si l'état était une image ? Concluez sur la limite de l'approche.

    ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 4 — DQN : remplacer la table par un réseau

    Une table ne **généralise** pas : ce qu'elle apprend pour une case ne dit rien de la case voisine, et elle
    devient impossible à remplir dès que l'état est grand. DQN (2015) remplace la table par un réseau qui reçoit
    l'état (les 4 nombres, sans découpage) et renvoie une valeur $Q$ par action.

    La cible reste celle du Q-learning, et le réseau est entraîné par régression vers cette cible. Mais entraîner
    un réseau sur des données qu'il produit lui-même est instable. DQN ajoute deux ingrédients :

    - une **mémoire de rejeu** : on stocke les transitions vécues, et on apprend sur des lots tirés au hasard dans
      cette mémoire. Les exemples d'un lot ne sont plus des pas consécutifs très ressemblants, et chaque
      transition sert plusieurs fois ;
    - un **réseau cible** : une copie du réseau, qui évolue lentement, sert à calculer les cibles. Sans elle, la
      cible bouge à chaque mise à jour du réseau qu'on entraîne, comme si on visait une cible montée sur son
      propre fusil.
    """)
    return


@app.cell
def _(nn, torch):
    class ReseauQ(nn.Module):
        """Reçoit un état (4 nombres), renvoie une valeur Q par action (2 nombres)."""

        def __init__(self, cache=64):
            super().__init__()
            self.couches = nn.Sequential(nn.Linear(4, cache), nn.ReLU(), nn.Linear(cache, cache), nn.ReLU(), nn.Linear(cache, 2))

        def forward(self, etats):
            return self.couches(etats)

    def politique_reseau(reseau):
        """La politique qui suit le réseau, sans exploration."""
        def politique(obs):
            with torch.no_grad():
                return int(reseau(torch.as_tensor(obs, dtype=torch.float32)[None]).argmax())
        return politique

    return ReseauQ, politique_reseau


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### À compléter — TODO 3 : les cibles de DQN

    On tire un lot de $B$ transitions dans la mémoire. Entrées de la fonction :

    - `recompenses` : tenseur `(B,)` ;
    - `q_suivants` : tenseur `(B, 2)`, les valeurs données par le **réseau cible** pour les états suivants ;
    - `termines` : tenseur `(B,)` qui vaut 1 si la transition a terminé l'épisode, 0 sinon ;
    - `gamma`.

    Renvoyez le tenseur `(B,)` des cibles : $r + \gamma \, (1 - \text{terminé}) \, \max_{a'} Q_{\text{cible}}(s', a')$.

    La cible est une **constante** pour l'apprentissage : aucun gradient ne doit la traverser. Terminez par `.detach()`.
    Aide : `q_suivants.max(dim=1).values` donne le maximum de chaque ligne.
    """)
    return


@app.function
def cibles_dqn(recompenses, q_suivants, termines, gamma):
    """Cibles de régression de DQN, une par transition, sans gradient."""
    # TODO 3
    raise NotImplementedError("À compléter")


@app.cell(hide_code=True)
def _(mo, tests, tp):
    ok_todo3, _affichage = tp.verifier(tests.tests_cibles(cibles_dqn), mo)
    _affichage
    return (ok_todo3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### La boucle d'apprentissage de DQN (fournie, à lire)

    Comparez-la à celle du Q-learning : la structure est la même. La table est devenue un réseau, la mise à jour
    d'une case est devenue une étape de descente de gradient sur un lot tiré de la mémoire.
    Tous les 2 500 pas, le réseau est évalué sans exploration, et on **conserve le meilleur** : c'est l'usage
    en pratique, vous allez voir pourquoi.
    """)
    return


@app.cell
def _(ReseauQ, choisir_action, copy, nn, np, politique_reseau, torch, tp):
    def entrainer_dqn(n_pas=30_000, graine=0, gamma=0.99, taille_lot=64, lr=1e-3, tau=0.005, capacite=20_000):
        """DQN sur CartPole. Renvoie un dictionnaire : dernier réseau, meilleur réseau, historique, évaluations."""
        torch.manual_seed(graine)
        rng = np.random.default_rng(graine)
        env = tp.creer_cartpole()
        reseau, reseau_cible = ReseauQ(), ReseauQ()
        reseau_cible.load_state_dict(reseau.state_dict())        # le réseau cible démarre comme une copie
        optimiseur = torch.optim.Adam(reseau.parameters(), lr=lr)
        memoire = tp.MemoireRejeu(capacite)
        historique, evaluations, meilleur_score, meilleur_poids = [], [], -1.0, None
        obs, _ = env.reset(seed=graine)
        total = 0.0
        for pas in range(1, n_pas + 1):
            # 1. Agir, avec votre politique epsilon-gloutonne, et ranger la transition dans la mémoire
            epsilon = max(0.02, 1 - pas / (0.2 * n_pas))
            with torch.no_grad():
                q_valeurs = reseau(torch.as_tensor(obs)[None])[0].numpy()
            action = choisir_action(q_valeurs, epsilon, rng)
            suivant, recompense, termine, tronque, _ = env.step(action)
            memoire.ajouter(obs, action, recompense, suivant, termine)
            obs, total = suivant, total + recompense
            if termine or tronque:
                historique.append((pas, total))
                obs, _ = env.reset()
                total = 0.0
            # 2. Apprendre sur un lot tiré au hasard dans la mémoire
            if len(memoire) >= 1_000:
                etats, actions, recompenses, suivants, termines = memoire.tirer(taille_lot, rng)
                q_predits = reseau(etats).gather(1, actions[:, None])[:, 0]              # Q(s, a) de l'action jouée
                cibles = cibles_dqn(recompenses, reseau_cible(suivants), termines, gamma)   # votre TODO 3
                perte = nn.functional.smooth_l1_loss(q_predits, cibles)
                optimiseur.zero_grad()
                perte.backward()
                nn.utils.clip_grad_norm_(reseau.parameters(), 10)
                optimiseur.step()
                # 3. Le réseau cible suit lentement le réseau entraîné
                with torch.no_grad():
                    for p, p_cible in zip(reseau.parameters(), reseau_cible.parameters()):
                        p_cible.mul_(1 - tau).add_(tau * p)
            # 4. Évaluer régulièrement sans exploration, et conserver le meilleur réseau
            if pas % 2_500 == 0:
                score = tp.evaluer_politique(politique_reseau(reseau), 5, premiere_graine=500)
                evaluations.append((pas, score))
                if score >= meilleur_score:
                    meilleur_score, meilleur_poids = score, copy.deepcopy(reseau.state_dict())
        meilleur = ReseauQ()
        meilleur.load_state_dict(meilleur_poids)
        return {"dernier": reseau, "meilleur": meilleur, "historique": historique, "evaluations": evaluations}

    return (entrainer_dqn,)


@app.cell(hide_code=True)
def _(mo):
    bouton_dqn = mo.ui.run_button(label="Entraîner DQN (3 essais de 30 000 pas)")
    bouton_dqn
    return (bouton_dqn,)


@app.cell
def _(
    N_PAS,
    bouton_dqn,
    entrainer_dqn,
    mo,
    np,
    ok_todo1,
    ok_todo3,
    plt,
    politique_reseau,
    time,
    torch,
    tp,
):
    tp.bloquer_si(not (ok_todo1 and ok_todo3), mo, "Terminez les TODO 1 et 3 pour débloquer l'entraînement.")
    _cache = tp.charger_cache("tp5_dqn")
    mo.stop(not (tp.lance(bouton_dqn) or _cache is not None),
            mo.md("Cliquez pour lancer (environ 3 minutes). Pendant ce temps, relisez la boucle ci-dessus."))
    if tp.lance(bouton_dqn):
        _courbes, _evals, _dernier, _meilleur, _debut, _poids = [], [], [], [], time.perf_counter(), None
        for _graine in mo.status.progress_bar(range(3), title="DQN"):
            _r = entrainer_dqn(N_PAS, _graine)
            _courbes.append(tp.lisser_par_pas(_r["historique"], N_PAS))
            _evals.append([e[1] for e in _r["evaluations"]])
            _dernier.append(tp.evaluer_politique(politique_reseau(_r["dernier"])))
            _meilleur.append(tp.evaluer_politique(politique_reseau(_r["meilleur"])))
            if _meilleur[-1] >= max(_meilleur):
                _poids = _r["meilleur"].state_dict()
        torch.save(_poids, tp.SORTIES / "tp5_dqn.pt")            # le meilleur des trois essais, pour l'étape 5
        tp.sauver_cache("tp5_dqn", courbes=np.array(_courbes), evals=np.array(_evals), dernier=np.array(_dernier),
                        meilleur=np.array(_meilleur), duree=np.array((time.perf_counter() - _debut) / 3))
        _cache = tp.charger_cache("tp5_dqn")
    resultats_dqn = _cache
    _x = np.arange(1, _cache["evals"].shape[1] + 1) * 2_500
    _fig, _ax = plt.subplots(figsize=(6, 2.8))
    for _i, _c in enumerate(_cache["evals"]):
        _ax.plot(_x, _c, marker="o", markersize=3, label=f"essai {_i + 1}")
    _ax.set_xlabel("pas d'interaction avec l'environnement")
    _ax.set_ylabel("score sans exploration")
    _ax.set_ylim(0, 510)
    _ax.legend(fontsize=7)
    _fig.tight_layout()
    mo.vstack([
        _fig,
        tp.tableau([{"essai": i + 1, "score du dernier réseau": f"{_cache['dernier'][i]:.0f}",
                     "score du meilleur réseau conservé": f"{_cache['meilleur'][i]:.0f}"} for i in range(3)], mo),
        mo.md(f"Durée d'un essai : {float(_cache['duree']):.0f} s. Les scores du tableau sont mesurés sur 10 épisodes "
              "différents de ceux qui ont servi à choisir le meilleur réseau."),
    ])
    return (resultats_dqn,)


@app.cell
def _(mo):
    mo.md(r"""
    **Votre réponse :**

    1. Les courbes montent-elles régulièrement, comme la courbe d'un entraînement supervisé ? Décrivez ce que vous
       voyez, essai par essai.
    2. Lisez sur les courbes le score qu'aurait eu chaque essai si on l'avait arrêté à 20 000 pas, puis à
       25 000 pas. Pourquoi conserve-t-on le meilleur réseau évalué plutôt que le dernier ? Et pourquoi le
       tableau mesure-t-il les scores sur d'autres épisodes que ceux qui ont servi à choisir ce réseau ?
    3. Si vous n'aviez lancé qu'un seul essai, qu'auriez-vous conclu ? Que faut-il exiger d'un résultat publié
       en apprentissage par renforcement ?

    ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 5 — PPO avec une bibliothèque, et comparaison

    **PPO** (2017) est un algorithme acteur-critique. Il ne passe pas par $Q$ : il règle directement une politique
    qui donne la probabilité de chaque action. Il joue quelques milliers de pas avec la politique courante,
    rend plus probables les actions qui ont mieux rapporté que prévu, et **limite l'ampleur de chaque
    changement** (c'est le sens de *proximal*) pour ne pas détruire en une mise à jour ce qui fonctionnait.
    C'est aujourd'hui le choix par défaut quand on dispose d'un simulateur.

    En entreprise, on n'écrit pas ces algorithmes soi-même : on utilise une bibliothèque éprouvée.
    Avec **Stable-Baselines3**, l'entraînement tient en trois lignes (c'est ce que fait `tp.entrainer_ppo`) :

    ```python
    from stable_baselines3 import PPO
    modele = PPO("MlpPolicy", "CartPole-v1", seed=0)
    modele.learn(30_000)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bouton_ppo = mo.ui.run_button(label="Entraîner PPO (3 essais de 30 000 pas)")
    bouton_ppo
    return (bouton_ppo,)


@app.cell
def _(N_PAS, bouton_ppo, mo, np, time, tp):
    _cache = tp.charger_cache("tp5_ppo")
    mo.stop(not (tp.lance(bouton_ppo) or _cache is not None), mo.md("Cliquez pour lancer (environ 2 minutes)."))
    if tp.lance(bouton_ppo):
        _courbes, _scores, _debut = [], [], time.perf_counter()
        for _graine in mo.status.progress_bar(range(3), title="PPO"):
            _modele, _historique = tp.entrainer_ppo(N_PAS, _graine)
            _courbes.append(tp.lisser_par_pas(_historique, N_PAS))
            _scores.append(tp.evaluer_politique(tp.politique_ppo(_modele)))
            if _scores[-1] >= max(_scores):
                _modele.save(str(tp.SORTIES / "tp5_ppo"))
        tp.sauver_cache("tp5_ppo", courbes=np.array(_courbes), scores=np.array(_scores),
                        duree=np.array((time.perf_counter() - _debut) / 3))
        _cache = tp.charger_cache("tp5_ppo")
    resultats_ppo = _cache
    mo.md(f"Score final (politique sans tirage aléatoire, 10 épisodes) : **{', '.join(f'{s:.0f}' for s in _cache['scores'])}** "
          f"selon l'essai. Durée d'un essai : {float(_cache['duree']):.0f} s.")
    return (resultats_ppo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparaison

    Les courbes montrent le score des 10 derniers épisodes joués **pendant l'entraînement** (donc avec
    l'exploration) : le trait est la moyenne des 3 essais, la bande va du plus mauvais essai au meilleur.
    """)
    return


@app.cell(hide_code=True)
def _(
    np,
    plt,
    resultats_dqn,
    resultats_ppo,
    resultats_table,
    score_hasard,
    score_regle_anticipee,
):
    _fig, _ax = plt.subplots(figsize=(7, 3.4))
    for _nom, _r, _couleur in [("Q-learning tabulaire", resultats_table, "#7f8c8d"), ("DQN", resultats_dqn, "#2980b9"),
                               ("PPO", resultats_ppo, "#c0392b")]:
        _c = _r["courbes"]
        _x = np.arange(1, _c.shape[1] + 1) * 500
        _valide = ~np.isnan(_c).all(axis=0)
        _ax.plot(_x[_valide], np.nanmean(_c[:, _valide], axis=0), color=_couleur, label=_nom)
        _ax.fill_between(_x[_valide], np.nanmin(_c[:, _valide], axis=0), np.nanmax(_c[:, _valide], axis=0), color=_couleur, alpha=0.15)
    _ax.axhline(score_regle_anticipee, color="black", linestyle="--", linewidth=1, label="règle avec anticipation")
    _ax.axhline(score_hasard, color="black", linestyle=":", linewidth=1, label="hasard")
    _ax.set_xlabel("pas d'interaction avec l'environnement")
    _ax.set_ylabel("score pendant l'entraînement")
    _ax.set_ylim(0, 520)
    _ax.legend(fontsize=7, loc="center right")
    _fig.tight_layout()

    _fig
    return


@app.cell(hide_code=True)
def _(
    mo,
    np,
    resultats_dqn,
    resultats_ppo,
    resultats_table,
    score_hasard,
    score_regle_angle,
    score_regle_anticipee,
    tp,
):
    def _plage(v):
        return f"{np.mean(v):.0f} (de {np.min(v):.0f} à {np.max(v):.0f})"

    _n_table = resultats_table["courbes"].shape[1] * 500
    _n = resultats_dqn["courbes"].shape[1] * 500
    tableau_final = [
        {"méthode": "hasard", "score final": f"{score_hasard:.0f}", "interactions": 0, "entraînement": "aucun", "ce qu'il a fallu fournir": "rien"},
        {"méthode": "règle : pousser du côté où le mât penche", "score final": f"{score_regle_angle:.0f}", "interactions": 0,
         "entraînement": "aucun", "ce qu'il a fallu fournir": "une idée de la physique"},
        {"méthode": "règle avec anticipation", "score final": f"{score_regle_anticipee:.0f}", "interactions": 0,
         "entraînement": "aucun", "ce qu'il a fallu fournir": "une idée de la physique et un réglage"},
        {"méthode": "Q-learning tabulaire", "score final": _plage(resultats_table["scores"]), "interactions": _n_table,
         "entraînement": f"{float(resultats_table['duree']):.0f} s", "ce qu'il a fallu fournir": "la récompense et un découpage de l'état"},
        {"méthode": "DQN, dernier réseau", "score final": _plage(resultats_dqn["dernier"]), "interactions": _n,
         "entraînement": f"{float(resultats_dqn['duree']):.0f} s", "ce qu'il a fallu fournir": "la récompense"},
        {"méthode": "DQN, meilleur réseau conservé", "score final": _plage(resultats_dqn["meilleur"]), "interactions": _n,
         "entraînement": f"{float(resultats_dqn['duree']):.0f} s", "ce qu'il a fallu fournir": "la récompense"},
        {"méthode": "PPO (Stable-Baselines3)", "score final": _plage(resultats_ppo["scores"]), "interactions": _n,
         "entraînement": f"{float(resultats_ppo['duree']):.0f} s", "ce qu'il a fallu fournir": "la récompense"},
    ]
    mo.vstack([mo.md("Score final : moyenne des 3 essais (du plus mauvais au meilleur), sans exploration, maximum 500."),
               tp.tableau(tableau_final, mo)])
    return


@app.cell(hide_code=True)
def _(ReseauQ, mo, politique_reseau, resultats_dqn, resultats_ppo, torch, tp):
    # Les deux agents sauvegardés, sur le même épisode
    mo.stop(not ((tp.SORTIES / "tp5_dqn.pt").exists() and (tp.SORTIES / "tp5_ppo.zip").exists()),
            mo.md("Relancez les entraînements de DQN et de PPO pour voir les deux agents à l'œuvre."))
    _dqn = ReseauQ()
    _dqn.load_state_dict(torch.load(tp.SORTIES / "tp5_dqn.pt"))
    _total_dqn, _images_dqn = tp.jouer_episode(politique_reseau(_dqn), graine=3, max_images=200)
    _total_ppo, _images_ppo = tp.jouer_episode(tp.politique_ppo(tp.charger_ppo(tp.SORTIES / "tp5_ppo")), graine=3, max_images=200)
    mo.hstack([
        mo.vstack([mo.md(f"**DQN** (le meilleur des {len(resultats_dqn['meilleur'])} essais) : {int(_total_dqn)} pas"),
                   mo.Html(tp.balise_gif(_images_dqn))]),
        mo.vstack([mo.md(f"**PPO** (le meilleur des {len(resultats_ppo['scores'])} essais) : {int(_total_ppo)} pas"),
                   mo.Html(tp.balise_gif(_images_ppo))]),
    ], justify="start", gap=2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Votre réponse :**

    1. Classez les méthodes selon trois critères : score final, régularité d'un essai à l'autre, nombre
       d'interactions nécessaires.
    2. Regardez les deux animations : les agents tiennent-ils le mât de la même façon ? Le chariot reste-t-il au
       centre du rail ? La récompense demandait-elle qu'il y reste ?
    3. Une des lignes du tableau n'a demandé aucune interaction et aucun entraînement. Pourquoi ne s'en
       contente-t-on pas toujours ?

    ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 6 — Une limite à voir de ses yeux : la récompense rare

    Sur CartPole, la récompense arrive à chaque pas : l'agent sait tout de suite s'il fait bien.
    Changeons de problème. **MountainCar** : une voiture au fond d'une vallée doit atteindre le sommet de droite.
    Son moteur est trop faible pour monter directement : il faut d'abord reculer pour prendre de l'élan.
    Trois actions (pousser à gauche, ne rien faire, pousser à droite), un état de 2 nombres (position, vitesse),
    et une récompense de **−1 par pas** jusqu'à l'arrivée. L'épisode est coupé à 200 pas : un score de −200
    signifie que la voiture n'est **jamais** arrivée.

    On lance le même PPO, avec le même budget que sur CartPole.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bouton_mountain = mo.ui.run_button(label="Entraîner PPO sur MountainCar (30 000 pas)")
    bouton_mountain
    return (bouton_mountain,)


@app.cell
def _(N_PAS, bouton_mountain, mo, np, plt, tp):
    _cache = tp.charger_cache("tp5_mountaincar")
    mo.stop(not (tp.lance(bouton_mountain) or _cache is not None), mo.md("Cliquez pour lancer (moins d'une minute)."))
    if tp.lance(bouton_mountain):
        _modele, _historique = tp.entrainer_ppo(N_PAS, 0, environnement="MountainCar-v0")
        _score_ppo, _traj_ppo = tp.trajectoire("MountainCar-v0", tp.politique_ppo(_modele), graine=0)

        def _regle(obs):
            position, vitesse = obs
            return 2 if vitesse > 0 else 0            # pousser dans le sens du mouvement, pour amplifier l'élan

        _score_regle, _traj_regle = tp.trajectoire("MountainCar-v0", _regle, graine=0)
        tp.sauver_cache("tp5_mountaincar", scores_entrainement=np.array([h[1] for h in _historique]),
                        score_ppo=np.array(_score_ppo), score_regle=np.array(_score_regle),
                        traj_ppo=_traj_ppo[:, 0], traj_regle=_traj_regle[:, 0])
        _cache = tp.charger_cache("tp5_mountaincar")
    _fig, _ax = plt.subplots(figsize=(6, 2.8))
    _ax.plot(_cache["traj_ppo"], color="#c0392b", label="PPO après 30 000 pas")
    _ax.plot(_cache["traj_regle"], color="black", linestyle="--", label="règle : pousser dans le sens du mouvement")
    _ax.axhline(0.5, color="#27ae60", linewidth=1)
    _ax.text(2, 0.56, "sommet à atteindre", color="#27ae60", fontsize=7)
    _ax.set_ylim(-1.25, 0.75)
    _ax.set_xlabel("pas")
    _ax.set_ylabel("position de la voiture")
    _ax.legend(fontsize=7, loc="lower right")
    _fig.tight_layout()
    _s = _cache["scores_entrainement"]
    mo.vstack([
        _fig,
        mo.md(f"PPO : **{len(_s)}** épisodes joués pendant l'entraînement, dont **{int((_s > -200).sum())}** ont atteint le sommet. "
              f"Score de la politique apprise : **{float(_cache['score_ppo']):.0f}**. "
              f"Score de la règle d'une ligne : **{float(_cache['score_regle']):.0f}**."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Tant que l'agent n'a pas atteint le sommet une première fois **par hasard**, toutes ses tentatives reçoivent
    exactement la même note. Il n'a aucun signal pour distinguer une bonne action d'une mauvaise : il n'y a rien
    à apprendre. C'est le problème de la **récompense rare**, et il est fréquent : dans un problème réel, la
    récompense naturelle est souvent « réussi » ou « raté », à la fin.

    Les autres limites à connaître avant de proposer du renforcement en entreprise :

    - **le coût en interactions** : des dizaines de milliers de pas pour un mât en équilibre, des millions pour
      un jeu vidéo, davantage encore pour un robot simulé ;
    - **le besoin d'un simulateur**, et l'écart entre le simulateur et la réalité ;
    - **la conception de la récompense** : l'agent optimise ce qu'on a écrit, pas ce qu'on voulait dire ;
    - **l'instabilité** d'un essai à l'autre, que vous avez mesurée à l'étape 4 ;
    - **la sécurité** : explorer, c'est essayer des actions dont on ignore les conséquences.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Votre réponse :** proposez une modification de la récompense de MountainCar qui aiderait l'agent
    (indice : que sait la règle d'une ligne ?). Quel risque prend-on quand on modifie la récompense pour
    guider l'agent ?

    ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 7 — Fiche de décision et bilan de la journée
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    | Rubrique | Votre réponse |
    |---|---|
    | Problème | Maintenir un système en équilibre par une suite de décisions |
    | Baselines sans apprentissage et leurs scores | ... |
    | Baseline classique (table) et son score | ... |
    | Méthode profonde retenue et pourquoi | ... |
    | Régularité d'un essai à l'autre | ... |
    | Le deep learning est-il justifié sur CartPole ? | ... |
    | Dans quelle situation le serait-il clairement ? | ... |
    | Trois questions à poser avant de proposer du renforcement à un client | ... |

    ### Bilan de la journée

    Le Jour 5 ne fait pas partie du projet Usine 4.0 : ses deux TD sont rendus tels quels, complétés.
    Vérifiez que `tp5a_graphes.py` et `tp5b_renforcement.py` sont enregistrés, puis faites un commit.

    ### Pour aller plus loin (si vous avez terminé)

    - Relancez `entrainer_dqn` avec `tau=1.0` (le réseau cible est alors une copie exacte à chaque pas : il ne
      sert plus à rien), puis avec `capacite=500` (une mémoire très courte). Lancez plusieurs graines avant de
      conclure : vous savez maintenant qu'un seul essai ne prouve rien.
    - Modifiez la récompense de MountainCar comme vous l'avez proposé à l'étape 6 et vérifiez si PPO apprend.
      Aide : la classe `gymnasium.RewardWrapper` permet de transformer la récompense d'un environnement.
    - DQN surestime les valeurs, parce qu'il prend le maximum de valeurs bruitées. Le *Double DQN* corrige ce
      défaut : le réseau entraîné choisit l'action, le réseau cible lui donne sa valeur. Modifiez la boucle.
    """)
    return


if __name__ == "__main__":
    app.run()
