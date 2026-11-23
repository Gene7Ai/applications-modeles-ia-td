import marimo

__generated_with = "0.25.0"
app = marimo.App(
    width="medium",
    app_title="TD 5a - Graphes et recommandation",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _():
    import sys
    import time
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import torch
    import torch.nn as nn

    sys.path.insert(0, str(Path(mo.notebook_dir()) / "outils"))
    import tests_tp5 as tests
    import tp_utils as tp

    plt.rcParams["figure.dpi"] = 90
    return mo, nn, np, nx, plt, tests, torch, tp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TD 5a — Graphes et recommandation

    **Module Application au deep learning · ING3 IA · durée : 1h15 (fin de matinée)**

    Une image est une grille, une série est une ligne : dans les deux cas, chaque élément a des voisins évidents.
    Beaucoup de données n'ont pas cette régularité : un réseau social, des molécules, des produits achetés
    ensemble, des articles qui se citent. Ce sont des **graphes**. Ce matin, vous construisez un réseau de neurones
    sur graphe (GCN) à la main, vous le comparez à des méthodes qui ignorent soit le graphe, soit le contenu,
    puis vous passez de la classification à la recommandation.

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
    | 1 | Le cours en bref, avec démonstration (vu en cours : à relire si besoin) | — |
    | 2 | Le jeu de données Cora et deux baselines classiques | 15 min |
    | 3 | Construire un GCN | 25 min |
    | 4 | Comparer : MLP, GCN, GAT, et l'effet de la profondeur | 15 min |
    | 5 | Du graphe à la recommandation | 15 min |
    | 6 | Fiche de décision | 5 min |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 1 — Le cours en bref

    ### 1.1 Rappel : vocabulaire des graphes et attention

    Un graphe est un ensemble de **nœuds** reliés par des **arêtes**. On le décrit par sa **matrice d'adjacence**
    $A$ : $A_{ij} = 1$ si les nœuds $i$ et $j$ sont reliés, 0 sinon. Les nœuds reliés à $i$ sont ses **voisins**,
    et leur nombre est son **degré**. Chaque nœud porte aussi un vecteur de caractéristiques (les mots d'un
    article, le profil d'un utilisateur).

    L'**attention** (TD 1) pondère les éléments qu'on regarde. Sur un graphe, elle servira à pondérer les voisins.

    ### 1.2 L'idée commune à tous les GNN : le passage de messages

    À chaque couche, chaque nœud **agrège** les représentations de ses voisins et met à jour la sienne.
    Après une couche, un nœud connaît ses voisins ; après deux couches, les voisins de ses voisins ; et ainsi de suite.

    | Modèle | Comment un nœud agrège ses voisins |
    |---|---|
    | **GCN** | une moyenne pondérée par les degrés, identique pour tous les voisins |
    | **GAT** | une moyenne pondérée par un score d'**attention** appris : certains voisins comptent plus |
    | **GraphSAGE** | une agrégation sur un **échantillon** de voisins, ce qui permet de traiter des graphes de millions de nœuds par mini-lots et de calculer la représentation d'un nœud jamais vu |

    **Le fil conducteur : la recommandation.** Utilisateurs et produits forment un graphe dont les arêtes sont les
    interactions (achat, clic, note). Recommander, c'est prédire les arêtes manquantes. Pinterest a construit son
    système PinSage sur le principe de GraphSAGE, précisément parce qu'il passe à l'échelle.

    ### 1.3 Voir le passage de messages

    Sur ce petit réseau social (34 membres d'un club de karaté), on place une information sur un seul nœud et on
    applique plusieurs fois l'opération « chaque nœud prend la moyenne de ses voisins et de lui-même ».
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    demo_passes = mo.ui.slider(0, 8, value=0, label="Nombre de passes d'agrégation", show_value=True)
    demo_passes
    return (demo_passes,)


@app.cell(hide_code=True)
def _(demo_passes, mo, np, nx, plt):
    _g = nx.karate_club_graph()
    _a = nx.to_numpy_array(_g, weight=None) + np.eye(34)
    _p = _a / _a.sum(axis=1, keepdims=True)
    _signal = np.zeros(34)
    _signal[0] = 1.0
    for _ in range(demo_passes.value):
        _signal = _p @ _signal
    _fig, _ax = plt.subplots(figsize=(6, 3.6))
    nx.draw_networkx(_g, pos=nx.spring_layout(_g, seed=3), node_color=_signal, cmap="OrRd", vmin=0, vmax=max(0.15, _signal.max()),
                     node_size=180, font_size=6, edge_color="#bbbbbb", ax=_ax)
    _ax.axis("off")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"Après {demo_passes.value} passe(s), **{int((_signal > 1e-9).sum())}** nœud(s) sur 34 ont reçu "
                           "une partie de l'information partie du nœud 0. Avec beaucoup de passes, tous les nœuds "
                           "finissent par porter presque la même valeur : c'est le **sur-lissage**. Vous verrez à l'étape 4 "
                           "ce que devient un réseau quand on empile les couches.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 2 — Le jeu de données Cora et deux baselines classiques

    **Cora** : 2 708 articles de recherche en apprentissage automatique, reliés par 5 429 citations. Chaque article
    est décrit par la présence ou l'absence de 1 433 mots, et appartient à l'un de 7 thèmes. La tâche : retrouver
    le thème d'un article, en n'ayant le thème que de **20 articles par classe** (140 en tout). C'est le cas
    typique d'entreprise : beaucoup de données reliées entre elles, très peu d'étiquettes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bouton_cora = mo.ui.run_button(label="Charger Cora")
    bouton_cora
    return (bouton_cora,)


@app.cell(hide_code=True)
def _(bouton_cora, mo, tp):
    _deja_la = (tp.DONNEES / "cora" / "cora.content").exists()
    mo.stop(not (tp.lance(bouton_cora) or _deja_la), mo.md("Cliquez sur le bouton pour charger les données (8 Mo)."))
    cora = tp.charger_cora()
    return (cora,)


@app.cell(hide_code=True)
def _(cora, mo, np, nx, plt, tp):
    _g = nx.from_edgelist(cora["aretes"].tolist())
    _centre = 1358
    _voisinage = nx.ego_graph(_g, _centre, radius=2)
    _y = cora["y"].numpy()
    _fig, _ax = plt.subplots(figsize=(6, 3.8))
    nx.draw_networkx(_voisinage, pos=nx.spring_layout(_voisinage, seed=0), node_color=[_y[n] for n in _voisinage.nodes],
                     cmap="tab10", vmin=0, vmax=9, node_size=40, with_labels=False, edge_color="#cccccc", ax=_ax)
    _ax.set_title("Un article, ceux qu'il cite ou qui le citent, et leurs propres voisins (couleur = thème)", fontsize=8)
    _ax.axis("off")
    _fig.tight_layout()
    _deg = cora["A"].sum(1).numpy()
    _meme = float(np.mean(_y[cora["aretes"][:, 0]] == _y[cora["aretes"][:, 1]]))
    mo.vstack([
        mo.md(f"**{len(_y)}** articles, **{len(cora['aretes'])}** citations, {len(tp.CORA_CLASSES)} thèmes. "
              f"Degré moyen : {_deg.mean():.1f} (maximum {int(_deg.max())}). "
              f"**{_meme:.0%}** des citations relient deux articles du même thème."),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Deux méthodes classiques, chacune n'utilisant que **la moitié de l'information** :

    - une **régression logistique** sur les mots de chaque article, qui ignore le graphe ;
    - la **propagation d'étiquettes**, qui ignore les mots : les 140 thèmes connus se diffusent de proche en proche
      le long des citations, comme dans la démonstration de l'étape 1.
    """)
    return


@app.cell
def _(cora, mo, torch, tp):
    from sklearn.linear_model import LogisticRegression

    _X, _y = cora["X"].numpy(), cora["y"].numpy()
    _tr, _te = cora["train"].numpy(), cora["test"].numpy()
    # 1. Les mots seuls
    _acc_mots = float((LogisticRegression(max_iter=2000, C=100).fit(_X[_tr], _y[_tr]).predict(_X[_te]) == _y[_te]).mean())
    # 2. Le graphe seul : propagation d'étiquettes
    _a = cora["A"] + torch.eye(len(_y))
    _p = _a / _a.sum(dim=1, keepdim=True)
    _depart = torch.zeros(len(_y), 7)
    _depart[cora["train"], cora["y"][cora["train"]]] = 1
    _scores = _depart.clone()
    for _ in range(30):
        _scores = _p @ _scores
        _scores[cora["train"]] = _depart[cora["train"]]      # les étiquettes connues restent fixées
    _acc_graphe = float((_scores[cora["test"]].argmax(1) == cora["y"][cora["test"]]).float().mean())
    resultats_classiques = [
        {"méthode": "régression logistique (mots seuls)", "utilise": "le contenu", "accuracy test": round(_acc_mots, 3)},
        {"méthode": "propagation d'étiquettes (graphe seul)", "utilise": "les liens", "accuracy test": round(_acc_graphe, 3)},
    ]
    mo.vstack([mo.md("Hasard : 14 % (7 thèmes)."), tp.tableau(resultats_classiques, mo)])
    return (resultats_classiques,)


@app.cell
def _(mo):
    mo.md(r"""
    **Votre réponse :** laquelle des deux informations est la plus utile ici, le contenu ou les liens ?
    Reliez-le au pourcentage de citations entre articles du même thème affiché plus haut.

    ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 3 — Construire un GCN

    Un GCN utilise **les deux** informations. Une couche calcule :

    $$H' = \hat{A} \, H \, W \qquad \text{avec} \qquad \hat{A} = D^{-1/2} (A + I) \, D^{-1/2}$$

    - $H W$ : une transformation linéaire des caractéristiques de chaque nœud, comme dans un MLP ;
    - $\hat{A} \times$ : chaque nœud remplace sa représentation par une moyenne pondérée de celles de ses voisins.

    ### À compléter — TODO 1 : normaliser l'adjacence

    1. Ajoutez l'identité : $\tilde{A} = A + I$, pour que chaque nœud se compte parmi ses propres voisins.
    2. Calculez le degré de chaque nœud : la somme de chaque ligne de $\tilde{A}$.
    3. Divisez chaque coefficient $\tilde{A}_{ij}$ par $\sqrt{d_i \, d_j}$.

    Sans cette normalisation, les nœuds très connectés auraient des représentations énormes. Aide : avec `d` le
    vecteur des degrés, `d.sqrt()[:, None]` est une colonne et `d.sqrt()[None, :]` une ligne.
    """)
    return


@app.cell
def _(torch):
    def normaliser_adjacence(A):
        """A : (N, N) à valeurs 0/1. Renvoie D^(-1/2) (A + I) D^(-1/2), sans modifier A."""
        # TODO 1
        raise NotImplementedError("À compléter")
    return (normaliser_adjacence,)


@app.cell(hide_code=True)
def _(mo, normaliser_adjacence, tests, tp):
    ok_todo1, _affichage = tp.verifier(tests.tests_normalisation(normaliser_adjacence), mo)
    _affichage
    return (ok_todo1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### À compléter — TODO 2 : la couche GCN

    La transformation linéaire `self.lineaire` est déjà créée. Dans `forward`, appliquez-la à `x`, puis
    multipliez à gauche par `A_norm` (produit matriciel `@`).
    """)
    return


@app.cell
def _(nn):
    class CoucheGCN(nn.Module):
        def __init__(self, entree, sortie):
            super().__init__()
            self.lineaire = nn.Linear(entree, sortie)

        def forward(self, x, A_norm):  # x : (N, entree), A_norm : (N, N)
            """Chaque nœud transforme ses caractéristiques, puis agrège celles de ses voisins."""
            # TODO 2
            raise NotImplementedError("À compléter")
    return (CoucheGCN,)


@app.cell(hide_code=True)
def _(CoucheGCN, mo, tests, tp):
    ok_todo2, _affichage = tp.verifier(tests.tests_couche_gcn(CoucheGCN), mo)
    _affichage
    return (ok_todo2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Les trois modèles comparés (fournis)

    Les trois ont la même forme : une couche cachée, puis une couche de sortie à 7 dimensions.
    Seule change la façon dont un nœud utilise ses voisins.
    """)
    return


@app.cell
def _(CoucheGCN, nn, torch, tp):
    class MLP(nn.Module):
        """Aucun voisin : chaque article est classé à partir de ses seuls mots."""

        def __init__(self, cache=16):
            super().__init__()
            self.c1, self.c2, self.dropout = nn.Linear(1433, cache), nn.Linear(cache, 7), nn.Dropout(0.5)

        def forward(self, x, A_norm):
            return self.c2(self.dropout(torch.relu(self.c1(self.dropout(x)))))

    class GCN(nn.Module):
        """Vos couches GCN empilées. `n_couches` règle la profondeur, donc la distance à laquelle un nœud voit."""

        def __init__(self, n_couches=2, cache=16):
            super().__init__()
            tailles = [1433] + [cache] * (n_couches - 1) + [7]
            self.couches = nn.ModuleList([CoucheGCN(tailles[i], tailles[i + 1]) for i in range(n_couches)])
            self.dropout = nn.Dropout(0.5)

        def forward(self, x, A_norm, renvoyer_cache=False):
            for i, couche in enumerate(self.couches):
                x = couche(self.dropout(x), A_norm)
                if i < len(self.couches) - 1:
                    x = torch.relu(x)
                    cache = x
            return (x, cache) if renvoyer_cache else x

    class GAT(nn.Module):
        """Deux couches d'attention sur graphe (8 têtes d'attention, puis 1)."""

        def __init__(self):
            super().__init__()
            self.c1, self.c2 = tp.CoucheGAT(1433, 8, tetes=8), tp.CoucheGAT(64, 7, tetes=1, concatener=False)
            self.dropout = nn.Dropout(0.6)

        def forward(self, x, A_norm):
            return self.c2(self.dropout(torch.nn.functional.elu(self.c1(self.dropout(x), A_norm))), A_norm)

    return GAT, GCN, MLP


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 4 — Comparer : MLP, GCN, GAT

    Chaque modèle est entraîné 3 fois avec des initialisations différentes (200 époques). Comptez environ
    3 à 4 minutes sur CPU : profitez-en pour lire le code des trois modèles ci-dessus.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bouton_comparaison = mo.ui.run_button(label="Entraîner MLP, GCN et GAT (3 essais chacun)")
    bouton_comparaison
    return (bouton_comparaison,)


@app.cell
def _(
    GAT,
    GCN,
    MLP,
    bouton_comparaison,
    cora,
    mo,
    normaliser_adjacence,
    np,
    ok_todo1,
    ok_todo2,
    resultats_classiques,
    tp,
):
    tp.bloquer_si(not (ok_todo1 and ok_todo2), mo, "Terminez les TODO 1 et 2 pour débloquer l'entraînement.")
    A_norm = normaliser_adjacence(cora["A"])
    _cache = tp.charger_cache("tp5_comparaison")
    mo.stop(not (tp.lance(bouton_comparaison) or _cache is not None), mo.md("Cliquez sur le bouton pour lancer."))
    if tp.lance(bouton_comparaison):
        _acc, _duree = {}, {}
        for _nom, _classe in mo.status.progress_bar([("MLP", MLP), ("GCN", GCN), ("GAT", GAT)], title="Entraînement des trois modèles"):
            _essais = [tp.entrainer_noeuds(_classe(), cora["X"], A_norm, cora["y"], cora["train"], cora["val"], cora["test"], graine=g)
                       for g in range(3)]
            _acc[_nom], _duree[_nom] = [e[0] for e in _essais], float(np.mean([e[1] for e in _essais]))
        tp.sauver_cache("tp5_comparaison", **{f"acc_{n}": np.array(v) for n, v in _acc.items()},
                        **{f"duree_{n}": np.array(v) for n, v in _duree.items()})
        _cache = tp.charger_cache("tp5_comparaison")
    _info = {"MLP": "le contenu", "GCN": "le contenu et les liens", "GAT": "le contenu et les liens, pondérés par attention"}
    tableau_noeuds = list(resultats_classiques) + [
        {"méthode": n, "utilise": _info[n],
         "accuracy test": f"{_cache[f'acc_{n}'].mean():.3f} (de {_cache[f'acc_{n}'].min():.3f} à {_cache[f'acc_{n}'].max():.3f})",
         "entraînement (s)": round(float(_cache[f"duree_{n}"]), 1)}
        for n in ["MLP", "GCN", "GAT"]]
    tp.tableau(tableau_noeuds, mo)
    return (A_norm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### L'effet de la profondeur

    Empiler des couches permet de voir plus loin dans le graphe. Est-ce toujours mieux ?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bouton_profondeur = mo.ui.run_button(label="Tester des GCN de 1, 2, 4 et 8 couches")
    bouton_profondeur
    return (bouton_profondeur,)


@app.cell
def _(A_norm, GCN, bouton_profondeur, cora, mo, np, plt, tp):
    _cache = tp.charger_cache("tp5_profondeur")
    mo.stop(not (tp.lance(bouton_profondeur) or _cache is not None), mo.md("Cliquez pour lancer (environ 2 minutes)."))
    if tp.lance(bouton_profondeur):
        _prof = [1, 2, 4, 8]
        _acc = [tp.entrainer_noeuds(GCN(n_couches=n), cora["X"], A_norm, cora["y"], cora["train"], cora["val"], cora["test"])[0]
                for n in mo.status.progress_bar(_prof, title="GCN de profondeur croissante")]
        tp.sauver_cache("tp5_profondeur", prof=np.array(_prof), acc=np.array(_acc))
        _cache = tp.charger_cache("tp5_profondeur")
    _fig, _ax = plt.subplots(figsize=(5, 2.8))
    _ax.plot(_cache["prof"], _cache["acc"], marker="o")
    _ax.set_xticks(_cache["prof"])
    _ax.set_xlabel("nombre de couches GCN")
    _ax.set_ylabel("accuracy test")
    _ax.set_ylim(0, 1)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Questions**

    1. Que gagne-t-on à combiner le contenu et les liens, par rapport à chaque baseline de l'étape 2 ?
    2. L'attention de GAT apporte-t-elle quelque chose sur ce jeu ? Dans quel type de graphe pourrait-elle
       faire une vraie différence ?
    3. Comparez les GCN à 1, 2, 4 et 8 couches. Que manque-t-il au réseau le moins profond ? Qu'arrive-t-il au
       plus profond ? Deux pistes pour ce dernier : le sur-lissage (revoyez la démonstration de l'étape 1), et
       la difficulté d'entraîner un réseau profond sans connexions résiduelles (le problème que ResNet a
       résolu, au TD 1). Un seul essai par profondeur a été lancé : que faudrait-il faire avant de conclure ?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Votre réponse :**

    1. ...
    2. ...
    3. ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 5 — Du graphe à la recommandation

    Recommander, c'est proposer à un nœud des arêtes qui n'existent pas encore : ici, suggérer à l'auteur d'un
    article d'autres articles à citer. On procède comme au TD 4 : chaque article reçoit un **embedding**,
    et on recommande les articles dont l'embedding est le plus proche.

    Pour évaluer honnêtement, on **cache 500 citations** du graphe, puis on regarde si le système les retrouve
    parmi ses 20 premières suggestions. Quatre embeddings sont comparés :

    | Embedding d'un article | Utilise | Apprentissage |
    |---|---|---|
    | ses mots | le contenu | aucun |
    | sa ligne dans la matrice d'adjacence : deux articles sont proches s'ils ont des voisins communs | les liens | aucun |
    | ses mots **propagés** deux fois, $\hat{A} \hat{A} X$ : l'agrégation d'un GCN à 2 couches, sans aucun poids | le contenu et les liens | aucun |
    | la couche cachée d'un GCN entraîné à classer les thèmes (étape 4) | le contenu et les liens | 140 étiquettes de thème |

    Les deux premiers sont les méthodes classiques. Avant de lancer, notez votre pronostic : lequel sera le meilleur ?

    ### À compléter — TODO 3 : recommander

    Entrées : `embeddings` `(N, d)`, le numéro du `noeud`, `k`, et `deja_lies`, la liste des nœuds auxquels il
    est déjà relié. Renvoyez les indices des `k` nœuds les plus proches au sens de la similarité cosinus,
    du plus proche au moins proche, en excluant le nœud lui-même et ses voisins déjà connus.
    Aide : mettez la similarité des nœuds exclus à `-np.inf` avant de trier avec `np.argsort`.
    """)
    return


@app.cell
def _(np):
    def recommander(embeddings, noeud, k, deja_lies):
        """Indices des k nœuds les plus similaires à `noeud`, hors lui-même et hors `deja_lies`."""
        # TODO 3
        raise NotImplementedError("À compléter")
    return (recommander,)


@app.cell(hide_code=True)
def _(mo, recommander, tests, tp):
    ok_todo3, _affichage = tp.verifier(tests.tests_recommandation(recommander), mo)
    _affichage
    return (ok_todo3,)


@app.cell(hide_code=True)
def _(mo):
    bouton_reco = mo.ui.run_button(label="Évaluer la recommandation sur 500 citations cachées")
    bouton_reco
    return (bouton_reco,)


@app.cell
def _(
    GCN,
    bouton_reco,
    cora,
    mo,
    normaliser_adjacence,
    np,
    ok_todo1,
    ok_todo2,
    ok_todo3,
    recommander,
    torch,
    tp,
):
    tp.bloquer_si(not (ok_todo1 and ok_todo2 and ok_todo3), mo, "Terminez les TODO 1 à 3 d'abord.")
    mo.stop(not tp.lance(bouton_reco), mo.md("Cliquez pour lancer (environ 2 minutes)."))
    _rng = np.random.default_rng(0)
    _cachees = cora["aretes"][_rng.choice(len(cora["aretes"]), 500, replace=False)]
    _A = cora["A"].clone()
    _A[_cachees[:, 0], _cachees[:, 1]] = 0           # on retire ces citations du graphe
    _A[_cachees[:, 1], _cachees[:, 0]] = 0
    _A_norm = normaliser_adjacence(_A)
    _gcn = GCN(cache=64)                             # le GCN est entraîné sans les citations cachées
    tp.entrainer_noeuds(_gcn, cora["X"], _A_norm, cora["y"], cora["train"], cora["val"], cora["test"])
    _gcn.eval()
    with torch.no_grad():
        _emb_gcn = _gcn(cora["X"], _A_norm, renvoyer_cache=True)[1].numpy()
    _embeddings = [
        ("mots (contenu seul)", cora["X"].numpy()),
        ("lignes de la matrice d'adjacence (liens seuls)", (_A + torch.eye(len(_A))).numpy()),
        ("mots propagés deux fois, sans apprentissage", (_A_norm @ _A_norm @ cora["X"]).numpy()),
        ("couche cachée du GCN entraîné à classer les thèmes", _emb_gcn),
    ]
    _voisins = [torch.nonzero(_A[i]).flatten().tolist() for i in range(len(_A))]
    _lignes = []
    for _nom, _emb in mo.status.progress_bar(_embeddings, title="Évaluation des quatre embeddings"):
        _trouvees = [int(v in recommander(_emb, int(u), 20, _voisins[int(u)])) for u, v in _cachees]
        _lignes.append({"embedding": _nom, "dimension": _emb.shape[1],
                        "citations cachées retrouvées dans les 20 premières suggestions": f"{np.mean(_trouvees):.1%}"})
    mo.vstack([mo.md("Tirer 20 articles au hasard parmi 2 708 retrouverait la bonne citation dans 0,7 % des cas."),
               tp.tableau(_lignes, mo)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Votre réponse :**

    1. Votre pronostic était-il le bon ? Quel embedding gagne, et a-t-il demandé un entraînement ?
    2. Le GCN est le meilleur modèle pour classer les thèmes. Pourquoi sa couche cachée est-elle si mal placée
       ici ? Indice : de quelle information a-t-on besoin pour distinguer 7 thèmes, et de quelle information
       pour distinguer deux articles du même thème ?
    3. Transposez à un site de commerce en ligne : que seraient les nœuds, les arêtes, les caractéristiques, et
       qu'est-ce qu'une « citation cachée » ?

    ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **À retenir.** Un embedding est bon pour la tâche sur laquelle il a été entraîné. C'est pourquoi les systèmes
    industriels (PinSage chez Pinterest, par exemple) entraînent leur réseau sur graphe **directement sur la
    tâche de recommandation**, en lui apprenant à rapprocher les nœuds reliés et à éloigner les autres, et non
    sur une tâche de classification. Et avant d'entraîner quoi que ce soit, ils mesurent ce que donne la simple
    propagation, qui ne coûte rien.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 6 — Fiche de décision

    Remplissez-la pour les deux tâches de la matinée : elles n'appellent pas la même réponse.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    | Rubrique | Classer des articles avec 140 étiquettes | Recommander des citations |
    |---|---|---|
    | Baselines classiques et leurs scores | ... | ... |
    | Méthode retenue et pourquoi | ... | ... |
    | Score obtenu | ... | ... |
    | Le deep learning est-il justifié ? | ... | ... |
    | Limite principale pour un graphe industriel (millions de nœuds, nouveaux nœuds chaque jour) | ... | ... |

    ### Pour aller plus loin (si vous avez terminé)

    - Identifiez dans votre futur secteur un jeu de données qui est naturellement un graphe. Quels seraient les
      nœuds, les arêtes, la tâche ?
    - Remplacez la moyenne pondérée du GCN par une moyenne simple des voisins ($D^{-1}(A + I)$) : est-ce moins bon ?
    - Écrivez une couche GraphSAGE : pour chaque nœud, tirez au hasard 5 voisins, faites la moyenne de leurs
      représentations, concaténez-la à celle du nœud, puis appliquez une couche linéaire.
    - Entraînez un GCN sur la tâche de recommandation elle-même : les embeddings $Z$ en sortie doivent donner un
      produit scalaire $z_i \cdot z_j$ élevé pour les nœuds reliés et faible pour les autres (entropie croisée
      binaire sur la matrice d'adjacence). Fait-il mieux que la propagation sans apprentissage ?
    """)
    return


if __name__ == "__main__":
    app.run()
