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
    # Projet Usine 4.0 — Brique 2 : localisation des défauts d'acier

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

    L'usine veut savoir où se trouve le défaut sur la tôle, en plus de son type (brique 1). Cette brique cherche donc des boîtes autour des défauts.

    Données : NEU-DET avec ses annotations (.xml), converties au format YOLO. Une image peut contenir plusieurs défauts du même type. On garde exactement le même découpage que la brique 1 (tp.decoupage_neu()) : 1080 images d'entraînement (2488 boîtes), 120 de validation (274 boîtes) et 600 de test (1427 boîtes). Ainsi, une image n'est jamais dans l'entraînement d'un modèle et dans le test de l'autre. Les tailles des défauts sont très différentes : une inclusion couvre en général environ 4 % de l'image, une zone de piqûres (pitted_surface) plus de la moitié.
    """)
    return


@app.cell
def _(mo, tp):
    # même découpage que la brique 1 (tp.decoupage_neu()) : chaque image est dans la même partie pour les deux modèles.
    # Le jeu YOLO est écrit dans un nouveau dossier : celui du TD 2 ne découpe pas les images de la même façon
    dossier_yolo = tp.DONNEES / "neu-yolo-projet"
    if (dossier_yolo / "config.yaml").exists():
        config_neu = dossier_yolo / "config.yaml"
    else:
        config_neu = tp.voc_vers_yolo(tp.DONNEES / "NEU", dossier_yolo, tp.NEU_CLASSES, parties=tp.decoupage_neu())
    images_train = tp.lister_images(dossier_yolo, "train")
    images_val = tp.lister_images(dossier_yolo, "val")
    images_test = tp.lister_images(dossier_yolo, "test")
    mo.md(f"Entraînement : {len(images_train)} images, validation : {len(images_val)}, test : {len(images_test)}.")
    return config_neu, images_test, images_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. La baseline classique

    Une méthode sans réseau de neurones, raisonnable pour ce problème : laquelle, pourquoi, quel score ?

    Méthode sans apprentissage, reprise de l'étape 11 du TD 2 : un défaut est une zone plus claire ou plus sombre que le reste de la tôle. On compare chaque pixel au fond local (l'image très floutée), on garde les pixels qui s'en écartent de plus de k écarts-types, on relie les morceaux proches (fermeture), puis on met une boîte autour de chaque région assez grande. Elle ne donne pas le type du défaut.

    Une boîte est jugée juste si son IoU avec un défaut annoté atteint 0,3 (sans regarder le type) : les défauts ont des contours flous, deux annotateurs ne les placeraient pas au pixel près. Ce critère est fixé à l'avance et le même pour toutes les méthodes.

    Réglage sur la validation (64 combinaisons, meilleur F1) : taille du flou sigma (5, 10, 25, 50), seuil k (1 à 2,5 écarts-types) et aire minimale (150 à 1500 pixels). Une première grille plus petite donnait son meilleur score au bord (aire 400, la plus grande essayée), on l'a donc élargie. Meilleur réglage : sigma = 25, k = 1,5, aire minimale 800 pixels, avec F1 = 0,443 (précision 53,1 %, rappel 38,0 %) sur la validation.

    Score sur le test : mesuré une seule fois, à la fin (partie 4).
    """)
    return


@app.cell
def _(np):
    import cv2


    # fonctions recopiées du TD 2
    def iou(boite_a, boite_b):
        """IoU entre deux boîtes [x1, y1, x2, y2]."""
        x1 = max(boite_a[0], boite_b[0])
        y1 = max(boite_a[1], boite_b[1])
        x2 = min(boite_a[2], boite_b[2])
        y2 = min(boite_a[3], boite_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        aire_a = (boite_a[2] - boite_a[0]) * (boite_a[3] - boite_a[1])
        aire_b = (boite_b[2] - boite_b[0]) * (boite_b[3] - boite_b[1])
        union = aire_a + aire_b - inter
        return float(inter / union)


    def boites_depuis_masque(masque, aire_min):
        """Une boîte [x1, y1, x2, y2] par composante connexe d'au moins aire_min pixels."""
        n, etiquettes, stats, centres = cv2.connectedComponentsWithStats(masque.astype(np.uint8))
        boites = []
        for i in range(1, n):
            x, y, largeur, hauteur, aire = stats[i]
            if aire >= aire_min:
                boites.append([x, y, x + largeur, y + hauteur])
        return boites


    def evaluer_detections(predictions, verites, seuil_iou=0.5):
        """predictions : liste de (boite, score) ; verites : liste de boîtes. Renvoie tp, fp, fn, precision, rappel."""
        predictions = sorted(predictions, key=lambda p: p[1], reverse=True)
        associee = [False] * len(verites)
        tp, fp = 0, 0
        for boite, score in predictions:
            meilleure_iou = 0
            meilleur_j = -1
            for j in range(len(verites)):
                if not associee[j]:
                    v = iou(boite, verites[j])
                    if v > meilleure_iou:
                        meilleure_iou = v
                        meilleur_j = j
            if meilleur_j >= 0 and meilleure_iou >= seuil_iou:
                tp += 1
                associee[meilleur_j] = True
            else:
                fp += 1
        fn = len(verites) - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        rappel = tp / (tp + fn) if tp + fn > 0 else 0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "rappel": rappel}

    return boites_depuis_masque, cv2, evaluer_detections


@app.cell
def _(compter, detecter_classique, images_val, mo, tp):
    # réglage sur la validation : taille du flou, seuil (en écarts-types) et aire minimale, on garde le meilleur F1
    reglages_baseline = []
    for _sigma in [5, 10, 25, 50]:
        for _k in [1.0, 1.5, 2.0, 2.5]:
            for _aire in [150, 400, 800, 1500]:
                _preds = [detecter_classique(p, _sigma, _k, _aire) for p in images_val]
                _tp, _fp, _fn = compter(images_val, _preds)
                _p = _tp / max(1, _tp + _fp)
                _r = _tp / max(1, _tp + _fn)
                _f1 = 2 * _p * _r / (_p + _r) if _p + _r > 0 else 0
                reglages_baseline.append({"sigma": _sigma, "k": _k, "aire min": _aire, "précision": round(_p, 3),
                                          "rappel": round(_r, 3), "F1": round(_f1, 3)})
    reglages_baseline.sort(key=lambda r: r["F1"], reverse=True)
    tp.tableau(reglages_baseline[:10], mo)
    return


@app.cell
def _(boites_depuis_masque, cv2, evaluer_detections, np, tp):
    # Baseline sans apprentissage (point de départ : étape 11 du TD 2) : on garde les pixels qui s'écartent
    # du fond local de la tôle (l'image très floutée), puis une boîte par région assez grande
    def detecter_classique(chemin, sigma, k, aire):
        gris = cv2.cvtColor(tp.lire_image_rgb(chemin), cv2.COLOR_RGB2GRAY).astype(np.float32)
        ecart = np.abs(gris - cv2.GaussianBlur(gris, (0, 0), sigma))
        masque = ecart > ecart.mean() + k * ecart.std()
        # fermeture : relie les morceaux proches d'un même défaut
        masque = cv2.morphologyEx(masque.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        return [(b, 1.0) for b in boites_depuis_masque(masque, aire)]


    def compter(images, predictions, seuil_iou=0.3):
        # vrais positifs, fausses alertes et oublis additionnés sur toutes les images.
        # Une boîte est juste si son IoU avec un défaut annoté atteint 0,3, sans regarder le type
        total_tp, total_fp, total_fn = 0, 0, 0
        for chemin, preds in zip(images, predictions):
            r = evaluer_detections(preds, list(tp.lire_annotations(chemin)[0]), seuil_iou)
            total_tp += r["tp"]
            total_fp += r["fp"]
            total_fn += r["fn"]
        return total_tp, total_fp, total_fn

    return compter, detecter_classique


@app.cell
def _(compter, detecter_classique, images_test):
    # baseline retenue (sigma = 25, k = 1,5, aire 800), mesurée une seule fois sur le test
    preds_baseline_test = [detecter_classique(p, 25, 1.5, 800) for p in images_test]
    _tp, _fp, _fn = compter(images_test, preds_baseline_test)
    score_baseline = {"vrais positifs": _tp, "fausses alertes": _fp, "défauts non vus": _fn}
    return (score_baseline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. L'approche par deep learning

    Quel modèle, quelle stratégie d'adaptation, quels réglages essayés (comparés sur la validation) ?

    Modèle : YOLOv8n pré-entraîné sur COCO, fine-tuné sur les 1080 images d'entraînement (50 époques, images 320 px, 36 minutes sur CPU). C'est le détecteur du TD 2 : un seul passage du réseau donne la boîte et le type du défaut, et c'est le plus petit modèle de la famille, adapté à un PC sans carte graphique. Les poids sont sauvegardés dans projet/poids/yolo_neu.pt.

    Sur la validation, mAP@0.5 = 0,734. Par type : patches 0,949, scratches 0,849, pitted_surface 0,841, inclusion 0,802, rolled-in_scale 0,584 et crazing 0,377. Les craquelures (crazing) et la calamine (rolled-in_scale) sont les plus difficiles : ce sont des textures réparties sur une grande partie de l'image, sans bord net, donc il est difficile de retrouver la boîte annotée.

    Hypothèses de coût : une fausse alerte coûte 1 (un opérateur vérifie une tôle pour rien, environ une minute) et un défaut non vu coûte 20 (une tôle défectueuse part chez le client : retour, réclamation). Le seuil de confiance est choisi sur la validation pour avoir le coût total le plus bas, avec le même critère que la baseline (IoU au moins 0,3, sans regarder le type). Seuil retenu : 0,05, coût 785 (260 défauts trouvés sur 274, 505 fausses alertes, 14 défauts non vus). La première grille commençait à 0,05, on a ajouté 0,01 à 0,03 pour vérifier que le minimum n'était pas au bord : en dessous, le coût remonte. Avec le même critère et les mêmes coûts, la baseline a un coût de 3492 (170 défauts non vus sur 274).

    Avec ces coûts, le seuil est très bas et donne beaucoup de fausses alertes (environ 4 par image). Si l'usine juge les fausses alertes plus chères, le seuil monte : par exemple 0,1 donne 251 fausses alertes et 28 défauts non vus.
    """)
    return


@app.cell
def _(config_neu, mo, tp):
    import shutil
    import time
    from ultralytics import YOLO

    # YOLOv8n pré-entraîné sur COCO, fine-tuné sur les 6 types de défauts (comme au TD 2).
    # L'entraînement dure environ 30 minutes sur CPU, donc les poids sont gardés dans projet/poids/ :
    # il n'est relancé que si le fichier n'existe pas
    dossier_poids = tp.RACINE / "projet" / "poids"
    chemin_poids = dossier_poids / "yolo_neu.pt"
    if not chemin_poids.exists():
        tp.fixer_graine(0)
        _debut = time.perf_counter()
        _modele = tp.charger_yolo("yolov8n.pt")
        _modele.train(data=str(config_neu), epochs=50, imgsz=320, batch=16, workers=0, plots=False, amp=False,
                      seed=0, project=str(tp.SORTIES / "yolo"), name="neu_projet", exist_ok=True, verbose=False)
        dossier_poids.mkdir(exist_ok=True)
        shutil.copy(tp.SORTIES / "yolo" / "neu_projet" / "weights" / "best.pt", chemin_poids)
        (dossier_poids / "yolo_neu_duree.txt").write_text(f"{time.perf_counter() - _debut:.0f}")
    duree_yolo = float((dossier_poids / "yolo_neu_duree.txt").read_text())
    yolo = YOLO(str(chemin_poids))
    mo.md(f"YOLOv8n fine-tuné : 50 époques, images 320 px, entraînement en {duree_yolo / 60:.0f} minutes sur CPU.")
    return (yolo,)


@app.cell
def _(config_neu, mo, tp, yolo):
    # mAP@0.5 par type de défaut sur la validation (ici le type compte)
    _r = yolo.val(data=str(config_neu), split="val", imgsz=320, batch=16, workers=0, plots=False, verbose=False)
    _lignes = [{"type de défaut": tp.NEU_CLASSES[int(c)], "AP@0.5": round(float(_r.box.ap50[k]), 3)}
               for k, c in enumerate(_r.box.ap_class_index)]
    mo.vstack([mo.md(f"mAP@0.5 sur la validation : **{_r.box.map50:.3f}**"), tp.tableau(_lignes, mo)])
    return


@app.cell
def _(compter, detecter_classique, images_val, mo, tp, yolo):
    # hypothèses de coût : une fausse alerte = un opérateur vérifie une tôle pour rien (environ une minute),
    # un défaut non vu = une tôle défectueuse part chez le client (retour, réclamation)
    cout_fausse_alerte = 1
    cout_oubli = 20

    # prédictions sur la validation avec un seuil très bas, puis on essaie plusieurs seuils de confiance.
    # NMS agnostique et IoU de 0,3 sans regarder le type : même critère que pour la baseline
    _brutes = tp.predictions_yolo(yolo, images_val, conf=0.01, taille=320, agnostique=True)
    choix_seuil = []
    for _s in [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]:
        _preds = [[(b, sc) for b, sc in zip(_bx, _sc) if sc >= _s] for _bx, _sc, _ in _brutes]
        _tp, _fp, _fn = compter(images_val, _preds)
        choix_seuil.append({"seuil": _s, "vrais positifs": _tp, "fausses alertes": _fp, "défauts non vus": _fn,
                            "précision": round(_tp / max(1, _tp + _fp), 3), "rappel": round(_tp / max(1, _tp + _fn), 3),
                            "coût total": cout_fausse_alerte * _fp + cout_oubli * _fn})
    seuil_retenu = min(choix_seuil, key=lambda r: r["coût total"])["seuil"]

    # la baseline réglée (sigma = 25, k = 1,5, aire 800), avec le même critère et les mêmes coûts
    _tp, _fp, _fn = compter(images_val, [detecter_classique(p, 25, 1.5, 800) for p in images_val])
    mo.vstack([
        tp.tableau(choix_seuil, mo),
        mo.md(f"Seuil de confiance retenu (coût minimal sur la validation) : **{seuil_retenu}**"),
        mo.md(f"Baseline sur la validation, même critère : {_tp} vrais positifs, {_fp} fausses alertes, "
              f"{_fn} défauts non vus, coût total **{cout_fausse_alerte * _fp + cout_oubli * _fn}**."),
    ])
    return cout_fausse_alerte, cout_oubli, seuil_retenu


@app.cell
def _(compter, config_neu, images_test, seuil_retenu, tp, yolo):
    # YOLO au seuil choisi sur la validation, mesuré une seule fois sur le test (même critère que la baseline)
    _brutes = tp.predictions_yolo(yolo, images_test, conf=0.01, taille=320, agnostique=True)
    preds_yolo_test = [[(b, sc) for b, sc in zip(_bx, _sc) if sc >= seuil_retenu] for _bx, _sc, _ in _brutes]
    _tp, _fp, _fn = compter(images_test, preds_yolo_test)
    score_modele = {"vrais positifs": _tp, "fausses alertes": _fp, "défauts non vus": _fn}

    # mAP@0.5 par type sur le test (ici le type compte)
    _r = yolo.val(data=str(config_neu), split="test", imgsz=320, batch=16, workers=0, plots=False, verbose=False)
    map50_test = round(float(_r.box.map50), 3)
    ap50_test = {tp.NEU_CLASSES[int(c)]: round(float(_r.box.ap50[k]), 3) for k, c in enumerate(_r.box.ap_class_index)}
    return ap50_test, map50_test, preds_yolo_test, score_modele


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Résultats et analyse

    Tableau comparatif (mêmes données, même métrique pour toutes les méthodes), cas d'échec, seuil ou réglage
    retenu et sa justification en termes de coût métier.

    Les deux méthodes sont mesurées une seule fois sur les 600 images de test (1427 défauts annotés), avec le même critère (IoU au moins 0,3, sans regarder le type) et les mêmes coûts (fausse alerte 1, défaut non vu 20).

    - Baseline (écart au fond local) : 475 défauts trouvés, 952 non vus et 462 fausses alertes, soit un rappel de 33,3 %, une précision de 50,7 % et un coût total de 19 502. Elle rate deux défauts sur trois.
    - YOLOv8n au seuil 0,05 : 1356 défauts trouvés, 71 non vus et 2735 fausses alertes, soit un rappel de 95,0 % (± 1,1 point), une précision de 33,1 % et un coût total de 4155, presque cinq fois moins que la baseline. Ces chiffres sont proches de ceux de la validation (rappel 94,9 %, précision 34 %).
    - En tenant compte du type, YOLO obtient un mAP@0.5 de 0,741 sur le test (0,734 sur la validation). Les types les plus difficiles restent crazing (0,427) et rolled-in_scale (0,542).

    Défauts non vus par type : crazing 19 sur 231 (8 %), inclusion 14 sur 306, rolled-in_scale 11 sur 202, pitted_surface 9 sur 165, patches 10 sur 307 et scratches 8 sur 216 (moins de 6 % pour tous ces types). Les exemples montrés sont des craquelures : la texture couvre une grande partie de la tôle et les boîtes annotées la découpent de façon un peu arbitraire. YOLO met des boîtes sur la bonne zone, mais pas au même endroit que l'annotation. Sur ces exemples, une partie des fausses alertes tombe aussi sur la texture du défaut : le comptage par boîte est sévère pour ce type.

    Limite : au seuil choisi, il y a environ 4,6 fausses alertes par image. C'est la conséquence de l'hypothèse « un défaut non vu coûte 20 fois une fausse alerte ». Si l'usine ne peut pas traiter autant d'alertes, il faudra revoir ces coûts et choisir à nouveau le seuil sur la validation.
    """)
    return


@app.cell
def _(
    ap50_test,
    cout_fausse_alerte,
    cout_oubli,
    map50_test,
    mo,
    score_baseline,
    score_modele,
    seuil_retenu,
    tp,
):
    # même critère (IoU au moins 0,3, sans regarder le type) et mêmes coûts pour les deux méthodes
    tableau_final = []
    for _nom, _sc in [("baseline (écart au fond local)", score_baseline), (f"YOLOv8n, seuil {seuil_retenu}", score_modele)]:
        _vp, _fa, _nv = _sc["vrais positifs"], _sc["fausses alertes"], _sc["défauts non vus"]
        tableau_final.append({"méthode": _nom, "vrais positifs": _vp, "fausses alertes": _fa, "défauts non vus": _nv,
                              "précision": round(_vp / max(1, _vp + _fa), 3), "rappel": round(_vp / max(1, _vp + _nv), 3),
                              "coût total": cout_fausse_alerte * _fa + cout_oubli * _nv})
    mo.vstack([
        tp.tableau(tableau_final, mo),
        mo.md(f"YOLO sur le test, en tenant compte du type : mAP@0.5 = **{map50_test}**"),
        tp.tableau([{"type de défaut": _k, "AP@0.5": _v} for _k, _v in ap50_test.items()], mo),
    ])
    return


@app.cell
def _(evaluer_detections, images_test, mo, preds_yolo_test, tp):
    # défauts non vus par type (chaque image NEU ne contient qu'un type de défaut), et quelques images avec des oublis
    oublis_par_type = {n: 0 for n in tp.NEU_CLASSES}
    defauts_par_type = {n: 0 for n in tp.NEU_CLASSES}
    _exemples, _titres = [], []
    for _chemin, _preds in zip(images_test, preds_yolo_test):
        _vb, _vc = tp.lire_annotations(_chemin)
        _res = evaluer_detections(_preds, list(_vb), 0.3)
        _type = tp.NEU_CLASSES[_vc[0]]
        oublis_par_type[_type] += _res["fn"]
        defauts_par_type[_type] += len(_vb)
        if _res["fn"] > 0 and len(_exemples) < 8:
            _img = tp.dessiner_boites(tp.Image.open(_chemin).convert("RGB"), _vb, classes=[1] * len(_vb))
            _img = tp.dessiner_boites(_img, [b for b, _ in _preds], classes=[0] * len(_preds), epaisseur=1)
            _exemples.append(_img)
            _titres.append(f"{_type} : {_res['fn']} oubli(s)")
    _lignes = [{"type de défaut": n, "défauts annotés": defauts_par_type[n], "non vus": oublis_par_type[n]}
               for n in tp.NEU_CLASSES]
    mo.vstack([
        tp.tableau(_lignes, mo),
        mo.md("Images avec des défauts non vus (en vert les défauts annotés, en rouge les boîtes de YOLO) :"),
        tp.grille(_exemples, _titres, n_col=4, taille=2.4),
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
