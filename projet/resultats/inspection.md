# Partie 1 : inspection des tôles

## 1. Classification du type de défaut

1080 images d'entraînement, 120 de validation, 600 de test.

### Baseline : réglage sur la validation

| descripteur | C | accuracy validation |
|---|---|---|
| HOG | 0.1 | 0.8 |
| HOG | 1 | 0.9 |
| HOG | 10 | 0.917 |
| HOG | 100 | 0.917 |
| LBP | 0.1 | 0.808 |
| LBP | 1 | 0.908 |
| LBP | 10 | 0.925 |
| LBP | 100 | 0.942 |
| HOG + LBP | 0.1 | 0.808 |
| HOG + LBP | 1 | 0.933 |
| HOG + LBP | 10 | 0.95 |
| HOG + LBP | 100 | 0.95 |

Retenu : HOG + LBP, C = 10.

### EfficientNet-B0 : réglage sur la validation

| couche | C | accuracy validation |
|---|---|---|
| dernière couche | 0.01 | 0.983 |
| dernière couche | 0.1 | 0.992 |
| dernière couche | 1 | 0.992 |
| dernière couche | 10 | 0.992 |
| features[:6] | 0.01 | 0.975 |
| features[:6] | 0.1 | 0.975 |
| features[:6] | 1 | 0.983 |
| features[:6] | 10 | 0.983 |

Retenu : dernière couche, C = 0.1.

### Test (mesuré une fois)

| méthode | accuracy test | crazing | inclusion | patches | pitted_surface | rolled-in_scale | scratches |
|---|---|---|---|---|---|---|---|
| HOG + LBP + SVM (baseline) | 0.935 | 0.98 | 0.93 | 0.86 | 0.93 | 1.0 | 0.91 |
| EfficientNet-B0 (dernière couche) + régression logistique | 0.997 | 1.0 | 0.98 | 1.0 | 1.0 | 1.0 | 1.0 |

Erreurs du modèle retenu (2) : inclusion_7.jpg (réel inclusion, prédit pitted_surface), inclusion_8.jpg (réel inclusion, prédit scratches)

## 2. Localisation des défauts

Une boîte est juste si son IoU avec un défaut annoté atteint 0,3 (sans regarder le type). Coûts : fausse alerte = 1, défaut non vu = 20.

### Baseline : les 5 meilleurs réglages sur la validation

| sigma | k | aire min | F1 validation |
|---|---|---|---|
| 25 | 1.5 | 800 | 0.443 |
| 25 | 1.5 | 400 | 0.438 |
| 5 | 2.0 | 400 | 0.421 |
| 10 | 2.0 | 400 | 0.415 |
| 50 | 1.5 | 800 | 0.405 |

Retenu : sigma = 25, k = 1.5, aire minimale = 800.

### YOLOv8n (50 époques, 320 px, 36 minutes d'entraînement sur CPU)

mAP@0.5 sur la validation : 0.734

| type | AP@0.5 validation |
|---|---|
| crazing | 0.377 |
| inclusion | 0.802 |
| patches | 0.949 |
| pitted_surface | 0.841 |
| rolled-in_scale | 0.584 |
| scratches | 0.849 |


### Choix du seuil de confiance sur la validation

| méthode | vrais positifs | fausses alertes | défauts non vus | précision | rappel | coût total |
|---|---|---|---|---|---|---|
| seuil 0.01 | 268 | 1843 | 6 | 0.127 | 0.978 | 1963 |
| seuil 0.02 | 266 | 1154 | 8 | 0.187 | 0.971 | 1314 |
| seuil 0.03 | 264 | 824 | 10 | 0.243 | 0.964 | 1024 |
| seuil 0.05 | 260 | 505 | 14 | 0.34 | 0.949 | 785 |
| seuil 0.1 | 246 | 251 | 28 | 0.495 | 0.898 | 811 |
| seuil 0.15 | 232 | 153 | 42 | 0.603 | 0.847 | 993 |
| seuil 0.2 | 213 | 91 | 61 | 0.701 | 0.777 | 1311 |
| seuil 0.25 | 197 | 64 | 77 | 0.755 | 0.719 | 1604 |
| seuil 0.3 | 188 | 49 | 86 | 0.793 | 0.686 | 1769 |
| seuil 0.4 | 164 | 21 | 110 | 0.886 | 0.599 | 2221 |
| seuil 0.5 | 146 | 10 | 128 | 0.936 | 0.533 | 2570 |
| seuil 0.6 | 120 | 2 | 154 | 0.984 | 0.438 | 3082 |

Seuil retenu (coût minimal) : 0.05.

### Test (mesuré une fois)

| méthode | vrais positifs | fausses alertes | défauts non vus | précision | rappel | coût total |
|---|---|---|---|---|---|---|
| baseline (écart au fond local) | 475 | 462 | 952 | 0.507 | 0.333 | 19502 |
| YOLOv8n, seuil 0.05 | 1356 | 2735 | 71 | 0.331 | 0.95 | 4155 |

YOLO en tenant compte du type : mAP@0.5 = 0.741

| type | AP@0.5 test |
|---|---|
| crazing | 0.427 |
| inclusion | 0.802 |
| patches | 0.931 |
| pitted_surface | 0.83 |
| rolled-in_scale | 0.542 |
| scratches | 0.916 |

Défauts non vus par YOLO, par type :

| type | défauts annotés | non vus |
|---|---|---|
| crazing | 231 | 19 |
| inclusion | 306 | 14 |
| patches | 307 | 10 |
| pitted_surface | 165 | 9 |
| rolled-in_scale | 202 | 11 |
| scratches | 216 | 8 |

