"""Partie 1 du projet : inspection des tôles (NEU-DET).

1. Reconnaître le type du défaut : baseline HOG + LBP + SVM, contre EfficientNet-B0 (extraction de
   features) + régression logistique.
2. Localiser les défauts : baseline « écart au fond local », contre YOLOv8n fine-tuné.

Pour les deux, les réglages sont choisis sur la validation et le test n'est mesuré qu'une fois, à la fin.
Le découpage est le même partout : tp.decoupage_neu() (100 images de test et 20 de validation par type).
Les résultats sont écrits dans projet/resultats/inspection.md.
"""
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch.nn as nn
from skimage.feature import hog, local_binary_pattern
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from ultralytics import YOLO

DOSSIER_PROJET = Path(__file__).resolve().parent
sys.path.insert(0, str(DOSSIER_PROJET.parent / "outils"))
import tp_utils as tp

DOSSIER_RESULTATS = DOSSIER_PROJET / "resultats"
CHEMIN_POIDS_YOLO = DOSSIER_PROJET / "poids" / "yolo_neu.pt"

# Hypothèses de coût pour la détection (expliquées dans le rapport) :
# une fausse alerte = un opérateur vérifie une tôle pour rien (environ une minute),
# un défaut non vu = une tôle défectueuse part chez le client (retour, réclamation)
COUT_FAUSSE_ALERTE = 1
COUT_OUBLI = 20


def tableau_md(lignes):
    """Transforme une liste de dictionnaires en tableau markdown (pour le fichier de résultats)."""
    colonnes = list(lignes[0])
    texte = "| " + " | ".join(colonnes) + " |\n|" + "---|" * len(colonnes) + "\n"
    for ligne in lignes:
        texte += "| " + " | ".join(str(ligne[c]) for c in colonnes) + " |\n"
    return texte


# ---------------------------------------------------------------------------
# 1. Classification du type de défaut
# ---------------------------------------------------------------------------
def descripteur_hog(chemin):
    # HOG : forme des contours (comme au TD 1), sur l'image réduite à 128 x 128
    img = np.asarray(tp.Image.open(chemin).convert("L").resize((128, 128)))
    return hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))


def descripteur_lbp(chemin):
    # LBP : texture. Les images sont en niveaux de gris, donc on remplace les histogrammes
    # de couleur du TD 1 par un histogramme de motifs LBP
    img = np.asarray(tp.Image.open(chemin).convert("L"))
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    h, _ = np.histogram(lbp, bins=10, range=(0, 10))
    return h / h.sum()


def accuracy_par_type(pred, y, noms):
    return {n: round(float(np.mean(pred[y == k] == k)), 2) for k, n in enumerate(noms)}


def classification(texte):
    print("1. Classification du type de défaut")
    (images_train, y_train, images_val, y_val, images_test, y_test,
     noms) = tp.charger_neu(n_train_par_classe=180, n_val_par_classe=20, n_test_par_classe=100)
    y_train, y_val, y_test = np.array(y_train), np.array(y_val), np.array(y_test)
    texte.append(f"## 1. Classification du type de défaut\n\n{len(images_train)} images d'entraînement, "
                 f"{len(images_val)} de validation, {len(images_test)} de test.\n")

    # --- baseline : on calcule les descripteurs une seule fois par image
    hog_train = np.stack([descripteur_hog(p) for p in images_train])
    hog_val = np.stack([descripteur_hog(p) for p in images_val])
    hog_test = np.stack([descripteur_hog(p) for p in images_test])
    lbp_train = np.stack([descripteur_lbp(p) for p in images_train])
    lbp_val = np.stack([descripteur_lbp(p) for p in images_val])
    lbp_test = np.stack([descripteur_lbp(p) for p in images_test])
    variantes = {
        "HOG": (hog_train, hog_val, hog_test),
        "LBP": (lbp_train, lbp_val, lbp_test),
        "HOG + LBP": (np.hstack([hog_train, lbp_train]), np.hstack([hog_val, lbp_val]), np.hstack([hog_test, lbp_test])),
    }
    # réglage sur la validation : le descripteur et C (à égalité, on garde le premier, donc le plus petit C)
    reglages = []
    meilleur = None
    for nom, (x_tr, x_va, _) in variantes.items():
        for C in [0.1, 1, 10, 100]:
            svm = make_pipeline(StandardScaler(), SVC(C=C)).fit(x_tr, y_train)
            acc = float(np.mean(svm.predict(x_va) == y_val))
            reglages.append({"descripteur": nom, "C": C, "accuracy validation": round(acc, 3)})
            if meilleur is None or acc > meilleur[0]:
                meilleur = (acc, nom, C)
    _, nom_baseline, C_baseline = meilleur
    texte.append(f"### Baseline : réglage sur la validation\n\n{tableau_md(reglages)}\n"
                 f"Retenu : {nom_baseline}, C = {C_baseline}.\n")

    # --- deep learning : EfficientNet-B0 pré-entraîné, sans sa tête
    transfo = tp.transformation(224)
    effnet = tp.charger_modele("efficientnet_b0")
    effnet.classifier = nn.Identity()
    effnet_complet = tp.charger_modele("efficientnet_b0")
    effnet_bas = nn.Sequential(effnet_complet.features[:6], nn.AdaptiveAvgPool2d(1), nn.Flatten())
    embeddings = {}
    for nom_couche, modele in [("dernière couche", effnet), ("features[:6]", effnet_bas)]:
        embeddings[nom_couche] = [tp.extraire_embeddings(modele, tp.DatasetImages(images, y, transfo))
                                  for images, y in [(images_train, y_train), (images_val, y_val), (images_test, y_test)]]
    # réglage sur la validation : la couche utilisée et C
    reglages = []
    meilleur = None
    for nom_couche, (e_tr, e_va, _) in embeddings.items():
        for C in [0.01, 0.1, 1, 10]:
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C)).fit(e_tr, y_train)
            acc = float(np.mean(clf.predict(e_va) == y_val))
            reglages.append({"couche": nom_couche, "C": C, "accuracy validation": round(acc, 3)})
            if meilleur is None or acc > meilleur[0]:
                meilleur = (acc, nom_couche, C)
    _, couche_modele, C_modele = meilleur
    texte.append(f"### EfficientNet-B0 : réglage sur la validation\n\n{tableau_md(reglages)}\n"
                 f"Retenu : {couche_modele}, C = {C_modele}.\n")

    # --- mesure finale sur le test, une seule fois, avec les réglages retenus
    x_tr, _, x_te = variantes[nom_baseline]
    svm = make_pipeline(StandardScaler(), SVC(C=C_baseline)).fit(x_tr, y_train)
    pred_baseline = svm.predict(x_te)
    e_tr, _, e_te = embeddings[couche_modele]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C_modele)).fit(e_tr, y_train)
    pred_modele = clf.predict(e_te)
    lignes = []
    for nom, pred in [(f"{nom_baseline} + SVM (baseline)", pred_baseline),
                      (f"EfficientNet-B0 ({couche_modele}) + régression logistique", pred_modele)]:
        ligne = {"méthode": nom, "accuracy test": round(float(np.mean(pred == y_test)), 3)}
        ligne.update(accuracy_par_type(pred, y_test, noms))
        lignes.append(ligne)
    erreurs = [f"{images_test[i].name} (réel {noms[y_test[i]]}, prédit {noms[pred_modele[i]]})"
               for i in np.flatnonzero(pred_modele != y_test)]
    texte.append(f"### Test (mesuré une fois)\n\n{tableau_md(lignes)}\n"
                 f"Erreurs du modèle retenu ({len(erreurs)}) : {', '.join(erreurs)}\n")

    # matrices de confusion, pour le rapport
    for nom_fichier, pred in [("confusion_baseline.png", pred_baseline), ("confusion_effnet.png", pred_modele)]:
        _, fig = tp.matrice_confusion(y_test, pred, noms, taille=4)
        fig.savefig(DOSSIER_RESULTATS / nom_fichier)


# ---------------------------------------------------------------------------
# 2. Localisation des défauts
# ---------------------------------------------------------------------------
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
    """predictions : liste de (boite, score) ; verites : liste de boîtes. Renvoie tp, fp, fn."""
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
    return {"tp": tp, "fp": fp, "fn": len(verites) - tp}


def detecter_classique(chemin, sigma, k, aire):
    # baseline sans apprentissage (étape 11 du TD 2) : on garde les pixels qui s'écartent du fond local
    # de la tôle (l'image très floutée), puis une boîte par région assez grande
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


def ligne_detection(nom, vp, fa, nv):
    return {"méthode": nom, "vrais positifs": vp, "fausses alertes": fa, "défauts non vus": nv,
            "précision": round(vp / max(1, vp + fa), 3), "rappel": round(vp / max(1, vp + nv), 3),
            "coût total": COUT_FAUSSE_ALERTE * fa + COUT_OUBLI * nv}


def localisation(texte, entrainer):
    print("2. Localisation des défauts")
    # même découpage que la classification, écrit au format YOLO dans un nouveau dossier
    dossier_yolo = tp.DONNEES / "neu-yolo-projet"
    if (dossier_yolo / "config.yaml").exists():
        config = dossier_yolo / "config.yaml"
    else:
        config = tp.voc_vers_yolo(tp.DONNEES / "NEU", dossier_yolo, tp.NEU_CLASSES, parties=tp.decoupage_neu())
    images_val = tp.lister_images(dossier_yolo, "val")
    images_test = tp.lister_images(dossier_yolo, "test")
    texte.append("## 2. Localisation des défauts\n\nUne boîte est juste si son IoU avec un défaut annoté atteint 0,3 "
                 f"(sans regarder le type). Coûts : fausse alerte = {COUT_FAUSSE_ALERTE}, défaut non vu = {COUT_OUBLI}.\n")

    # --- baseline : réglage sur la validation (taille du flou, seuil, aire minimale), meilleur F1
    reglages = []
    meilleur = None
    for sigma in [5, 10, 25, 50]:
        for k in [1.0, 1.5, 2.0, 2.5]:
            for aire in [150, 400, 800, 1500]:
                vp, fa, nv = compter(images_val, [detecter_classique(p, sigma, k, aire) for p in images_val])
                p_ = vp / max(1, vp + fa)
                r_ = vp / max(1, vp + nv)
                f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ > 0 else 0
                reglages.append({"sigma": sigma, "k": k, "aire min": aire, "F1 validation": round(f1, 3)})
                if meilleur is None or f1 > meilleur[0]:
                    meilleur = (f1, sigma, k, aire)
    _, sigma, k, aire = meilleur
    reglages.sort(key=lambda r: r["F1 validation"], reverse=True)
    texte.append(f"### Baseline : les 5 meilleurs réglages sur la validation\n\n{tableau_md(reglages[:5])}\n"
                 f"Retenu : sigma = {sigma}, k = {k}, aire minimale = {aire}.\n")

    # --- YOLOv8n fine-tuné. L'entraînement dure environ 35 minutes sur CPU : les poids sont gardés dans
    # projet/poids/, et on ne réentraîne que si le fichier manque ou si on lance le pipeline avec --entrainer
    if entrainer or not CHEMIN_POIDS_YOLO.exists():
        tp.fixer_graine(0)
        debut = time.perf_counter()
        modele = tp.charger_yolo("yolov8n.pt")
        modele.train(data=str(config), epochs=50, imgsz=320, batch=16, workers=0, plots=False, amp=False,
                     seed=0, project=str(tp.SORTIES / "yolo"), name="neu_projet", exist_ok=True, verbose=False)
        CHEMIN_POIDS_YOLO.parent.mkdir(exist_ok=True)
        shutil.copy(tp.SORTIES / "yolo" / "neu_projet" / "weights" / "best.pt", CHEMIN_POIDS_YOLO)
        (CHEMIN_POIDS_YOLO.parent / "yolo_neu_duree.txt").write_text(f"{time.perf_counter() - debut:.0f}")
    yolo = YOLO(str(CHEMIN_POIDS_YOLO))
    duree = float((CHEMIN_POIDS_YOLO.parent / "yolo_neu_duree.txt").read_text())
    r = yolo.val(data=str(config), split="val", imgsz=320, batch=16, workers=0, plots=False, verbose=False)
    ap_val = [{"type": tp.NEU_CLASSES[int(c)], "AP@0.5 validation": round(float(r.box.ap50[i]), 3)}
              for i, c in enumerate(r.box.ap_class_index)]
    texte.append(f"### YOLOv8n (50 époques, 320 px, {duree / 60:.0f} minutes d'entraînement sur CPU)\n\n"
                 f"mAP@0.5 sur la validation : {r.box.map50:.3f}\n\n{tableau_md(ap_val)}\n")

    # choix du seuil de confiance sur la validation : celui qui donne le coût total le plus bas
    brutes = tp.predictions_yolo(yolo, images_val, conf=0.01, taille=320, agnostique=True)
    choix = []
    for s in [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]:
        preds = [[(b, sc) for b, sc in zip(bx, scs) if sc >= s] for bx, scs, _ in brutes]
        ligne = ligne_detection(f"seuil {s}", *compter(images_val, preds))
        ligne["seuil"] = s
        choix.append(ligne)
    seuil = min(choix, key=lambda l: l["coût total"])["seuil"]
    for ligne in choix:
        del ligne["seuil"]
    texte.append(f"### Choix du seuil de confiance sur la validation\n\n{tableau_md(choix)}\n"
                 f"Seuil retenu (coût minimal) : {seuil}.\n")

    # --- mesure finale sur le test, une seule fois, mêmes critères et mêmes coûts pour les deux méthodes
    preds_baseline = [detecter_classique(p, sigma, k, aire) for p in images_test]
    brutes = tp.predictions_yolo(yolo, images_test, conf=0.01, taille=320, agnostique=True)
    preds_yolo = [[(b, sc) for b, sc in zip(bx, scs) if sc >= seuil] for bx, scs, _ in brutes]
    lignes = [ligne_detection("baseline (écart au fond local)", *compter(images_test, preds_baseline)),
              ligne_detection(f"YOLOv8n, seuil {seuil}", *compter(images_test, preds_yolo))]
    r = yolo.val(data=str(config), split="test", imgsz=320, batch=16, workers=0, plots=False, verbose=False)
    ap_test = [{"type": tp.NEU_CLASSES[int(c)], "AP@0.5 test": round(float(r.box.ap50[i]), 3)}
               for i, c in enumerate(r.box.ap_class_index)]
    # défauts non vus par type (chaque image NEU ne contient qu'un type de défaut)
    oublis = {n: [0, 0] for n in tp.NEU_CLASSES}
    for chemin, preds in zip(images_test, preds_yolo):
        verites, classes = tp.lire_annotations(chemin)
        res = evaluer_detections(preds, list(verites), 0.3)
        oublis[tp.NEU_CLASSES[classes[0]]][0] += len(verites)
        oublis[tp.NEU_CLASSES[classes[0]]][1] += res["fn"]
    lignes_oublis = [{"type": n, "défauts annotés": v[0], "non vus": v[1]} for n, v in oublis.items()]
    texte.append(f"### Test (mesuré une fois)\n\n{tableau_md(lignes)}\n"
                 f"YOLO en tenant compte du type : mAP@0.5 = {r.box.map50:.3f}\n\n{tableau_md(ap_test)}\n"
                 f"Défauts non vus par YOLO, par type :\n\n{tableau_md(lignes_oublis)}\n")


def lancer(entrainer=False):
    DOSSIER_RESULTATS.mkdir(exist_ok=True)
    tp.fixer_graine(0)
    texte = ["# Partie 1 : inspection des tôles\n"]
    classification(texte)
    localisation(texte, entrainer)
    (DOSSIER_RESULTATS / "inspection.md").write_text("\n".join(texte), encoding="utf-8")
    print("Résultats écrits dans", DOSSIER_RESULTATS / "inspection.md")


if __name__ == "__main__":
    lancer(entrainer="--entrainer" in sys.argv)
