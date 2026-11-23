"""Tests automatiques du TD 5 (graphes et apprentissage par renforcement)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ================================================================ TD 5a : GRAPHES
def _petit_graphe():
    # chemin 0 - 1 - 2, plus un nœud isolé 3
    A = torch.zeros(4, 4)
    A[0, 1] = A[1, 0] = A[1, 2] = A[2, 1] = 1
    return A


# ---------------------------------------------------------------- TD 5a, TODO 1
def tests_normalisation(normaliser_adjacence):
    def forme_et_symetrie():
        N = normaliser_adjacence(_petit_graphe())
        assert tuple(N.shape) == (4, 4), f"forme attendue (4, 4), obtenue {tuple(N.shape)}"
        assert torch.allclose(N, N.T, atol=1e-6), "la matrice normalisée doit rester symétrique"

    def boucles():
        N = normaliser_adjacence(_petit_graphe())
        assert (torch.diagonal(N) > 0).all(), "chaque nœud doit se voir lui-même : ajoutez l'identité (A + I) avant de normaliser"
        assert abs(N[3, 3].item() - 1) < 1e-6, "un nœud isolé n'a que lui-même comme voisin : son coefficient vaut 1"

    def valeurs():
        N = normaliser_adjacence(_petit_graphe())
        # degrés avec boucle : 2, 3, 2, 1
        assert abs(N[0, 0].item() - 1 / 2) < 1e-5, f"N[0, 0] = 1 / racine(2 x 2) = 0.5, obtenu {N[0, 0].item():.3f}"
        assert abs(N[0, 1].item() - 1 / np.sqrt(6)) < 1e-5, f"N[0, 1] = 1 / racine(2 x 3) = 0.408, obtenu {N[0, 1].item():.3f}"
        assert N[0, 2].item() == 0, "deux nœuds non reliés gardent un coefficient nul"

    def entree_intacte():
        A = _petit_graphe()
        copie = A.clone()
        normaliser_adjacence(A)
        assert torch.equal(A, copie), "la fonction ne doit pas modifier la matrice A reçue (travaillez sur une copie)"

    return [("forme et symétrie", forme_et_symetrie), ("boucles sur chaque nœud", boucles),
            ("valeurs sur un petit graphe", valeurs), ("l'entrée n'est pas modifiée", entree_intacte)]


# ---------------------------------------------------------------- TD 5a, TODO 2
def tests_couche_gcn(CoucheGCN):
    def forme():
        torch.manual_seed(0)
        c = CoucheGCN(5, 3)
        y = c(torch.randn(4, 5), torch.eye(4))
        assert tuple(y.shape) == (4, 3), f"forme attendue (4 nœuds, 3 dimensions), obtenue {tuple(y.shape)}"

    def identite():
        torch.manual_seed(0)
        c = CoucheGCN(5, 3)
        x = torch.randn(4, 5)
        assert torch.allclose(c(x, torch.eye(4)), c.lineaire(x), atol=1e-6), \
            "avec une adjacence identité (aucun voisin), la couche doit se réduire à la transformation linéaire"

    def agregation():
        torch.manual_seed(0)
        c = CoucheGCN(5, 3)
        x = torch.randn(4, 5)
        N = torch.full((4, 4), 0.25)  # chaque nœud fait la moyenne de tous
        y = c(x, N)
        assert torch.allclose(y, c.lineaire(x).mean(0, keepdim=True).expand(4, 3), atol=1e-5), \
            "la couche doit calculer A_norm @ (X W) : chaque nœud agrège les représentations de ses voisins"

    def voisinage():
        torch.manual_seed(0)
        c = CoucheGCN(2, 2)
        x = torch.randn(4, 2)
        N = torch.eye(4)
        N[0, 1] = 1.0
        x2 = x.clone()
        x2[1] += 5
        d = (c(x2, N) - c(x, N)).abs().sum(1)
        assert d[0] > 1e-4 and d[2] < 1e-6 and d[3] < 1e-6, "modifier le nœud 1 ne doit changer que lui-même et le nœud 0, qui l'a pour voisin"

    return [("forme de sortie", forme), ("sans voisin : simple couche linéaire", identite),
            ("agrégation des voisins", agregation), ("l'information ne passe que par les arêtes", voisinage)]


# ---------------------------------------------------------------- TD 5a, TODO 3
def tests_recommandation(recommander):
    emb = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.3], [0.0, 1.0], [-1.0, 0.0], [5.0, 0.2]])

    def ordre():
        r = [int(i) for i in recommander(emb, 0, k=3, deja_lies=[])]
        assert r == [5, 1, 2], f"les plus proches du nœud 0 (similarité cosinus) sont, dans l'ordre, [5, 1, 2]. Obtenu {r}"

    def exclusions():
        r = [int(i) for i in recommander(emb, 0, k=3, deja_lies=[1, 5])]
        assert 0 not in r, "on ne recommande pas un article à lui-même"
        assert 1 not in r and 5 not in r, "on ne recommande pas un article déjà cité (deja_lies)"
        assert r == [2, 3, 4], f"attendu [2, 3, 4], obtenu {r}"

    def nombre():
        assert len(recommander(emb, 3, k=2, deja_lies=[])) == 2, "la fonction doit renvoyer exactement k indices"

    return [("classement par similarité cosinus", ordre), ("exclut le nœud lui-même et ses voisins", exclusions), ("k résultats", nombre)]


# ================================================================ TD 5b : RENFORCEMENT
# ---------------------------------------------------------------- TD 5b, TODO 1
def tests_epsilon(choisir_action):
    def glouton():
        rng = np.random.default_rng(0)
        assert all(choisir_action(np.array([0.1, 0.9]), 0.0, rng) == 1 for _ in range(50)), \
            "avec epsilon = 0, on choisit toujours l'action de plus grande valeur"
        assert all(choisir_action(np.array([2.0, -1.0]), 0.0, rng) == 0 for _ in range(50)), "avec epsilon = 0 et Q = [2, -1], l'action 0"

    def aleatoire():
        rng = np.random.default_rng(0)
        a = [choisir_action(np.array([0.1, 0.9]), 1.0, rng) for _ in range(2000)]
        p = np.mean(a)
        assert 0.44 < p < 0.56, f"avec epsilon = 1, les deux actions sont tirées à parts égales : action 1 choisie {p:.0%} du temps"

    def intermediaire():
        rng = np.random.default_rng(0)
        a = [choisir_action(np.array([0.1, 0.9]), 0.2, rng) for _ in range(5000)]
        p = np.mean(a)
        assert 0.87 < p < 0.93, f"avec epsilon = 0.2, la meilleure action est choisie environ 90 % du temps (80 % + la moitié des 20 %), obtenu {p:.0%}"

    def type_retour():
        a = choisir_action(np.array([0.1, 0.9]), 0.0, np.random.default_rng(0))
        assert isinstance(a, (int, np.integer)), f"la fonction doit renvoyer un entier, obtenu {type(a).__name__}"

    return [("epsilon = 0 : toujours la meilleure action", glouton), ("epsilon = 1 : action au hasard", aleatoire),
            ("epsilon = 0.2 : mélange des deux", intermediaire), ("renvoie un entier", type_retour)]


# ---------------------------------------------------------------- TD 5b, TODO 2
def tests_q_learning(mise_a_jour_q):
    def cas_general():
        Q = np.zeros((3, 2))
        Q[1] = [2.0, 5.0]
        mise_a_jour_q(Q, 0, 1, recompense=1.0, etat_suivant=1, termine=False, alpha=0.5, gamma=0.9)
        assert np.isclose(Q[0, 1], 0.5 * (1 + 0.9 * 5)), f"Q[0, 1] = 0 + 0.5 x (1 + 0.9 x 5 - 0) = 2.75, obtenu {Q[0, 1]}"

    def etat_terminal():
        Q = np.zeros((3, 2))
        Q[1] = [2.0, 5.0]
        mise_a_jour_q(Q, 0, 0, recompense=1.0, etat_suivant=1, termine=True, alpha=0.5, gamma=0.9)
        assert np.isclose(Q[0, 0], 0.5), f"si l'épisode est terminé, il n'y a pas de futur : cible = récompense. Attendu 0.5, obtenu {Q[0, 0]}"

    def seule_la_case():
        Q = np.ones((3, 2))
        avant = Q.copy()
        mise_a_jour_q(Q, 2, 1, 1.0, 0, False, 0.1, 0.99)
        avant[2, 1] = Q[2, 1]
        assert np.array_equal(Q, avant), "seule la case Q[etat, action] doit être modifiée"

    def etats_tuples():
        Q = np.zeros((2, 2, 2))
        Q[1, 1] = [3.0, 1.0]
        mise_a_jour_q(Q, (0, 1), 1, 2.0, (1, 1), False, 1.0, 0.5)
        assert np.isclose(Q[0, 1, 1], 2 + 0.5 * 3), ("l'état peut être un tuple d'indices (c'est le cas pour CartPole) : "
                                                     "accédez à la case par Q[etat][action] et aux valeurs suivantes par Q[etat_suivant]")

    return [("formule de mise à jour", cas_general), ("état terminal", etat_terminal),
            ("une seule case modifiée", seule_la_case), ("états sous forme de tuple", etats_tuples)]


# ---------------------------------------------------------------- TD 5b, TODO 3
def tests_cibles(cibles_dqn):
    r = torch.tensor([1.0, 1.0, 0.0])
    q_suiv = torch.tensor([[2.0, 4.0], [3.0, 1.0], [5.0, 6.0]])
    fini = torch.tensor([0.0, 1.0, 0.0])

    def valeurs():
        c = cibles_dqn(r, q_suiv, fini, 0.9)
        attendu = torch.tensor([1 + 0.9 * 4, 1.0, 0.9 * 6])
        assert torch.allclose(c, attendu), f"cibles attendues {attendu.tolist()}, obtenues {[round(float(v), 3) for v in c]}"

    def forme():
        c = cibles_dqn(r, q_suiv, fini, 0.9)
        assert tuple(c.shape) == (3,), f"une cible par transition : forme attendue (3,), obtenue {tuple(c.shape)}"

    def terminal():
        c = cibles_dqn(r, q_suiv, torch.ones(3), 0.9)
        assert torch.allclose(c, r), "quand l'épisode est terminé, la cible est la récompense seule"

    def sans_gradient():
        q = q_suiv.clone().requires_grad_(True)
        c = cibles_dqn(r, q, fini, 0.9)
        assert not c.requires_grad, "la cible ne doit pas transporter de gradient : utilisez .detach() (ou torch.no_grad())"

    return [("valeurs sur un exemple", valeurs), ("une cible par transition", forme),
            ("transitions terminales", terminal), ("pas de gradient dans la cible", sans_gradient)]
