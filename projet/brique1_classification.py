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
    return images_test, images_train, images_val, noms, y_test, y_train, y_val


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
    return SVC, StandardScaler, hog_train, lbp_train, make_pipeline


@app.cell
def _(
    SVC,
    StandardScaler,
    descripteur_hog,
    descripteur_lbp,
    hog_train,
    images_test,
    lbp_train,
    make_pipeline,
    np,
    y_test,
    y_train,
):
    # baseline retenue (HOG + LBP, C = 10) : entraînée sur l'entraînement, mesurée une seule fois sur le test
    _x_train = np.hstack([hog_train, lbp_train])
    _x_test = np.stack([np.concatenate([descripteur_hog(p), descripteur_lbp(p)]) for p in images_test])
    _svm = make_pipeline(StandardScaler(), SVC(C=10)).fit(_x_train, y_train)
    pred_baseline = _svm.predict(_x_test)
    score_baseline = round(float(np.mean(pred_baseline == np.array(y_test))), 3)
    return pred_baseline, score_baseline


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. L'approche par deep learning

    Quel modèle, quelle stratégie d'adaptation, quels réglages essayés (comparés sur la validation) ?

    Modèle : EfficientNet-B0 pré-entraîné sur ImageNet, utilisé en extraction de features : on garde le réseau tel quel et on entraîne une régression logistique sur ses embeddings. On a comparé la dernière couche (1280 valeurs) et des couches plus basses (features[:6], 112 valeurs), car NEU est loin d'ImageNet : c'est le cas « peu de données, domaine éloigné » du cours, où les couches basses peuvent mieux marcher.

    Réglage sur la validation : la couche utilisée et C parmi 0,01, 0,1, 1 et 10. Le meilleur est la dernière couche avec C = 0,1 : 99,2 % sur la validation (une seule erreur sur 120), contre 98,3 % pour features[:6] et 95,0 % pour la baseline.

    On n'a pas fait de fine-tuning. La feature extraction ne se trompe déjà que sur une image de validation : avec 120 images, un fine-tuning ne pourrait pas montrer de gain mesurable, alors qu'il demande un entraînement plus long sur CPU et des poids à sauvegarder.

    Score sur le test : mesuré une seule fois, à la fin (partie 4).
    """)
    return


@app.cell
def _(images_train, images_val, mo, tp, y_train, y_val):
    import torch.nn as nn

    # même prétraitement que pour ImageNet (images 224 x 224), sans augmentation
    _transfo = tp.transformation(224)
    ds_train = tp.DatasetImages(images_train, y_train, _transfo)
    ds_val = tp.DatasetImages(images_val, y_val, _transfo)

    # 1) EfficientNet-B0 pré-entraîné, sans sa tête : 1280 valeurs par image
    effnet = tp.charger_modele("efficientnet_b0")
    effnet.classifier = nn.Identity()
    emb_train = tp.extraire_embeddings(effnet, ds_train)
    emb_val = tp.extraire_embeddings(effnet, ds_val)

    # 2) seulement les premiers étages (features[:6]), moyennés : 112 valeurs par image.
    # NEU est loin d'ImageNet, les couches basses (textures) pourraient mieux marcher
    _effnet_complet = tp.charger_modele("efficientnet_b0")
    effnet_bas = nn.Sequential(_effnet_complet.features[:6], nn.AdaptiveAvgPool2d(1), nn.Flatten())
    emb_bas_train = tp.extraire_embeddings(effnet_bas, ds_train)
    emb_bas_val = tp.extraire_embeddings(effnet_bas, ds_val)

    mo.md(f"Embeddings : {emb_train.shape[1]} valeurs (dernière couche), {emb_bas_train.shape[1]} valeurs (features[:6]).")
    return effnet, emb_bas_train, emb_bas_val, emb_train, emb_val


@app.cell
def _(
    StandardScaler,
    emb_bas_train,
    emb_bas_val,
    emb_train,
    emb_val,
    make_pipeline,
    mo,
    np,
    tp,
    y_train,
    y_val,
):
    from sklearn.linear_model import LogisticRegression

    # régression logistique sur les embeddings, C réglé sur la validation
    reglages_deep = []
    for _nom, (_e_tr, _e_va) in {"EfficientNet-B0, dernière couche": (emb_train, emb_val),
                                  "EfficientNet-B0, features[:6]": (emb_bas_train, emb_bas_val)}.items():
        for _C in [0.01, 0.1, 1, 10]:
            _clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=_C)).fit(_e_tr, y_train)
            _acc = np.mean(_clf.predict(_e_va) == np.array(y_val))
            reglages_deep.append({"modèle": _nom, "C": _C, "accuracy validation": round(float(_acc), 3)})
    tp.tableau(reglages_deep, mo)
    return (LogisticRegression,)


@app.cell
def _(
    LogisticRegression,
    StandardScaler,
    effnet,
    emb_train,
    images_test,
    make_pipeline,
    np,
    tp,
    y_test,
    y_train,
):
    # modèle retenu (dernière couche d'EfficientNet-B0 + régression logistique, C = 0,1),
    # mesuré une seule fois sur le test
    _ds_test = tp.DatasetImages(images_test, y_test, tp.transformation(224))
    emb_test = tp.extraire_embeddings(effnet, _ds_test)
    _clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1)).fit(emb_train, y_train)
    pred_modele = _clf.predict(emb_test)
    score_modele = round(float(np.mean(pred_modele == np.array(y_test))), 3)
    return pred_modele, score_modele


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Résultats et analyse

    Tableau comparatif (mêmes données, même métrique pour toutes les méthodes), cas d'échec, seuil ou réglage
    retenu et sa justification en termes de coût métier.

    Les deux méthodes sont mesurées une seule fois, sur les mêmes 600 images de test et avec la même métrique (l'accuracy).

    - Baseline HOG + LBP + SVM : 93,5 % (± 2,0 points, intervalle de confiance à 95 %). Elle se trompe surtout sur patches (86 %) et scratches (91 %) : des plaques prises pour des craquelures ou des rayures, et des rayures prises pour des inclusions.
    - EfficientNet-B0 + régression logistique : 99,7 % (± 0,5 point), soit 2 erreurs sur 600. Les deux erreurs sont des inclusions (98 % sur ce type), prédites pitted_surface et scratches. Sur ces deux images, l'inclusion est une suite de petites taches allongées sur une ligne : elle ressemble à des piqûres ou à une rayure, l'erreur se comprend.

    Les scores du test sont proches de ceux de la validation (95,0 % et 99,2 %). L'écart entre les deux méthodes (6 points) est bien plus grand que l'incertitude.

    Seuil : pour reconnaître le type, il n'y a pas de seuil à régler, on garde le type le plus probable. Ici une erreur donne un défaut mal nommé, pas un défaut manqué. Le seuil d'alerte (défaut non vu contre fausse alerte) est choisi dans la brique 2, avec la localisation.
    """)
    return


@app.cell
def _(
    mo,
    noms,
    np,
    pred_baseline,
    pred_modele,
    score_baseline,
    score_modele,
    tp,
    y_test,
):
    # accuracy globale, puis type par type : une accuracy globale peut cacher un type mal reconnu
    _y = np.array(y_test)
    tableau_final = []
    for _nom, _pred, _score in [("HOG + LBP + SVM (baseline)", pred_baseline, score_baseline),
                                ("EfficientNet-B0 + régression logistique", pred_modele, score_modele)]:
        _ligne = {"méthode": _nom, "accuracy test": _score}
        for _k, _n in enumerate(noms):
            _ligne[_n] = round(float(np.mean(_pred[_y == _k] == _k)), 2)
        tableau_final.append(_ligne)
    tp.tableau(tableau_final, mo)
    return


@app.cell
def _(images_test, mo, noms, np, pred_baseline, pred_modele, tp, y_test):
    # matrices de confusion sur le test, puis les images où le modèle retenu se trompe
    _, _fig_baseline = tp.matrice_confusion(y_test, pred_baseline, noms, taille=4)
    _, _fig_modele = tp.matrice_confusion(y_test, pred_modele, noms, taille=4)
    _erreurs = np.flatnonzero(pred_modele != np.array(y_test))
    _images = [tp.Image.open(images_test[i]) for i in _erreurs]
    _titres = [f"réel {noms[y_test[i]]}\nprédit {noms[pred_modele[i]]}" for i in _erreurs]
    mo.vstack([
        mo.hstack([mo.vstack([mo.md("**Baseline**"), _fig_baseline]),
                   mo.vstack([mo.md("**EfficientNet-B0**"), _fig_modele])]),
        mo.md(f"**Erreurs du modèle retenu sur le test : {len(_erreurs)}**"),
        tp.grille(_images, _titres, n_col=4, taille=2.2, cmap="gray"),
    ])
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
