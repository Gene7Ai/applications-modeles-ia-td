"""Lance tout le projet Usine 4.0, des données jusqu'aux résultats sur le test.

    uv run python projet/pipeline.py              (utilise les poids sauvegardés dans projet/poids/)
    uv run python projet/pipeline.py --entrainer  (refait aussi les entraînements longs)

Les résultats sont écrits dans projet/resultats/.
"""
import sys
import time

import inspection

debut = time.perf_counter()
entrainer = "--entrainer" in sys.argv

inspection.lancer(entrainer)
# à venir : defauts_rares.lancer(entrainer) et pompes.lancer(entrainer)

print(f"Pipeline terminé en {(time.perf_counter() - debut) / 60:.1f} minutes")
