import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="Usine 4.0 - Brique à nommer",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # Le dossier outils/ est à la racine du dépôt, un niveau au-dessus de projet/.
    sys.path.insert(0, str(Path(mo.notebook_dir()).parent / "outils"))
    import tp_utils as tp

    plt.rcParams["figure.dpi"] = 90
    return mo, np, tp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Projet Usine 4.0 — Brique 1 : classification des défauts d'acier

    **Groupe :** ... · **Membres :** Xinyu YU

    **Mode d'emploi de ce modèle**

    - Copiez ce fichier dans `projet/` sous le nom de la brique indiqué à la fin du TD
      (`brique1_classification.py`, `brique2_localisation.py`, `brique3_anomalies.py`, `brique4_defauts_rares.py`),
      puis ouvrez la copie : `marimo edit projet/brique1_classification.py`.
    - Les fonctions fournies restent accessibles par `tp.` (chargement des données, affichage, cache).
      Les fonctions que vous avez écrites dans un TD ne s'importent pas d'un notebook à l'autre : copiez dans
      une cellule de cette brique celles dont vous avez besoin.
    - Dans une cellule, une variable dont le nom commence par un tiret bas reste locale à la cellule.
    - Fixez les graines (`tp.fixer_graine(0)`) et gardez les résultats des calculs longs
      (`tp.sauver_cache`, `tp.charger_cache`) pour ne pas les relancer à chaque ouverture.
    - Ne réglez rien sur le jeu de test : gardez une partie des données pour la validation, et ne mesurez sur
      le test qu'une fois vos choix faits.
    - Remplacez les `...` par votre travail et supprimez ce mode d'emploi avant le rendu.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. La question et les données

    Quelle question métier cette brique traite-t-elle ? Quelles données, en quelle quantité, comment
    sont-elles découpées (entraînement, validation, test) ?

    L'usine veut savoir si un modèle peut reconnaître le type d'un défaut sur une photo de tôle, pour aider les opérateurs qui contrôlent les tôles à l'œil. Cette brique traite seulement le type de défaut (la localisation est dans la brique 2).

    Données : NEU-DET, 1800 images de tôles d'acier en niveaux de gris (200 x 200 pixels), 300 par type de défaut, six types : crazing, inclusion, patches, pitted_surface, rolled-in_scale et scratches. On utilise le même découpage pour tout le projet (tp.decoupage_neu()) : par type, 100 images de test, 20 de validation et 180 d'entraînement, soit 1080 images d'entraînement, 120 de validation et 600 de test. Tous les réglages se font sur la validation, et le test ne sert qu'une fois, à la fin.
    """)
    return


@app.cell
def _(mo, tp):
    # Découpage commun à tout le projet (le même que tp.decoupage_neu()) :
    # par type de défaut, 100 images de test, 20 de validation, et le reste (180) pour l'entraînement
    (images_train, y_train, images_val, y_val, images_test, y_test,
     noms) = tp.charger_neu(n_train_par_classe=180, n_val_par_classe=20, n_test_par_classe=100)
    mo.md(f"Entraînement : {len(images_train)} images, validation : {len(images_val)}, test : {len(images_test)}.")
    return images_train, images_val, y_train, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. La baseline classique

    Une méthode sans réseau de neurones, raisonnable pour ce problème : laquelle, pourquoi, quel score ?

    Méthode : un SVM (noyau RBF) sur deux descripteurs faits main. Le HOG décrit la forme des contours (rayures, plaques...) et un histogramme LBP décrit la texture (craquelures, piqûres, calamine). Par rapport au TD 1, on a remplacé les histogrammes de couleur par le LBP, car les images sont en niveaux de gris : les trois canaux sont identiques.

    Réglage sur la validation : le descripteur (HOG seul, LBP seul, ou les deux) et C parmi 0,1, 1, 10 et 100. Le meilleur est HOG + LBP avec C = 10 : 95,0 % sur la validation (HOG seul : 91,7 %, LBP seul : 94,2 %). C = 100 donne le même score, on garde le plus petit. Avec seulement 120 images de validation, un écart de un ou deux points n'est pas très significatif.

    Score sur le test : mesuré une seule fois, à la fin (partie 4).
    """)
    return


@app.cell
def _(np, tp):
    from skimage.feature import hog, local_binary_pattern


    def descripteur_hog(chemin):
        # HOG : décrit la forme des contours (comme au TD 1), sur l'image réduite à 128 x 128
        img = np.asarray(tp.Image.open(chemin).convert("L").resize((128, 128)))
        return hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))


    def descripteur_lbp(chemin):
        # LBP : décrit la texture. Les images sont en niveaux de gris, donc on remplace
        # les histogrammes de couleur du TD 1 par un histogramme de motifs LBP
        img = np.asarray(tp.Image.open(chemin).convert("L"))
        lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
        h, _ = np.histogram(lbp, bins=10, range=(0, 10))
        return h / h.sum()

    return descripteur_hog, descripteur_lbp


@app.cell
def _(
    descripteur_hog,
    descripteur_lbp,
    images_train,
    images_val,
    mo,
    np,
    tp,
    y_train,
    y_val,
):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    # les descripteurs ne sont calculés qu'une fois par image
    hog_train = np.stack([descripteur_hog(p) for p in images_train])
    hog_val = np.stack([descripteur_hog(p) for p in images_val])
    lbp_train = np.stack([descripteur_lbp(p) for p in images_train])
    lbp_val = np.stack([descripteur_lbp(p) for p in images_val])

    # réglage sur la validation : le descripteur utilisé et le paramètre C du SVM
    variantes = {
        "HOG": (hog_train, hog_val),
        "LBP": (lbp_train, lbp_val),
        "HOG + LBP": (np.hstack([hog_train, lbp_train]), np.hstack([hog_val, lbp_val])),
    }
    reglages_baseline = []
    for _nom, (_x_tr, _x_va) in variantes.items():
        for _C in [0.1, 1, 10, 100]:
            _svm = make_pipeline(StandardScaler(), SVC(C=_C)).fit(_x_tr, y_train)
            _acc = np.mean(_svm.predict(_x_va) == np.array(y_val))
            reglages_baseline.append({"descripteur": _nom, "C": _C, "accuracy validation": round(float(_acc), 3)})
    tp.tableau(reglages_baseline, mo)
    return


@app.cell
def _():
    # Votre baseline : remplacez cette ligne par votre code ; réglez-la sur la validation,
    # puis gardez son score sur le test, mesuré une seule fois.
    score_baseline = None
    return (score_baseline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. L'approche par deep learning

    Quel modèle, quelle stratégie d'adaptation, quels réglages essayés (comparés sur la validation) ?

    ...
    """)
    return


@app.cell
def _():
    # Votre modèle : remplacez cette ligne par votre code ; réglez-le sur la validation,
    # puis gardez son score sur le test, mesuré une seule fois.
    score_modele = None
    return (score_modele,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Résultats et analyse

    Tableau comparatif (mêmes données, même métrique pour toutes les méthodes), cas d'échec, seuil ou réglage
    retenu et sa justification en termes de coût métier.

    ...
    """)
    return


@app.cell
def _(mo, score_baseline, score_modele, tp):
    # Une ligne par méthode, la même métrique pour toutes ; ajoutez ensuite vos figures et vos cas d'échec.
    tableau_final = [
        {"méthode": "baseline classique", "score sur le test": score_baseline},
        {"méthode": "deep learning", "score sur le test": score_modele},
    ]
    tp.tableau(tableau_final, mo)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Fiche de décision

    | Rubrique | Votre réponse |
    |---|---|
    | Problème métier | ... |
    | Baseline classique et son score | ... |
    | Modèle retenu et pourquoi | ... |
    | Métrique principale et score obtenu | ... |
    | Seuil ou réglage retenu, justification métier | ... |
    | Coût (annotation, entraînement, inférence, matériel) | ... |
    | Principaux cas d'échec | ... |
    | Le deep learning est-il justifié ici ? | ... |
    | Déployable en l'état ? Sinon, que manque-t-il ? | ... |
    """)
    return


if __name__ == "__main__":
    app.run()
