import copy
import random

import matplotlib.pyplot as plt
import numpy as np

H = 10
L = 10
dico = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
P_G = 20
S_G = 100
N = -5

alpha = 0.05
gamma = 0.95
epsilon_debut = 0.7
epsilon_fin = 0.05
episode = 100000

tab_val = [
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, 0, 1, 0, 0, -1, 0, 0, 0, -1],
    [10, 0, -1, 0, 0, -1, 0, 0, 0, 1],
    [-1, 0, -1, 1, -1, -1, -1, 1, 0, -1],
    [-1, 0, -1, 0, 0, 0, 0, 0, 0, -1],
    [-1, -1, -1, -1, -1, 0, -1, -1, -1, 1],
    [1, 0, 0, 0, -1, 0, -1, 0, 0, 0],
    [-1, -1, -1, -1, 1, -1, -1, -1, -1, -1],
    [-1, 0, 0, 0, 0, -1, 0, 0, 0, 10],
    [-1, -1, -1, 1, -1, -1, -1, -1, -1, -1],
]


def nombre_pastilles(tab_val):
    nbr_pastilles = 0
    for i in range(H):
        for j in range(L):
            if tab_val[i][j] == 1 or tab_val[i][j] == 10:
                nbr_pastilles += 1
    return nbr_pastilles


nbr_pastilles = nombre_pastilles(tab_val)

tab_caractère = np.array(
    [
        ["N", "N", "N", "N", "N", "N", "N", "N", "N", "N"],
        ["N", 0, "P_G", 0, 0, "N", 0, 0, 0, "N"],
        ["S_G", 0, "N", 0, 0, "N", 0, 0, 0, "P_G"],
        ["N", 0, "N", "P_G", "N", "N", "N", "P_G", 0, "N"],
        ["N", 0, "N", 0, 0, 0, 0, 0, 0, "N"],
        ["N", "N", "N", "N", "N", 0, "N", "N", "N", "P_G"],
        ["P_G", 0, 0, 0, "N", 0, "N", 0, 0, 0],
        ["N", "N", "N", "N", "P_G", "N", "N", "N", "N", "N"],
        ["N", 0, 0, 0, 0, "N", 0, 0, 0, "S_G"],
        ["N", "N", "N", "P_G", "N", "N", "N", "N", "N", "N"],
    ]
)


def associe_cara_a_val(tab_cara):
    tab = np.zeros((H, L))
    tab[tab_cara == "P_G"] = P_G
    tab[tab_cara == "S_G"] = S_G
    tab[tab_cara == "N"] = N
    return tab


tab = associe_cara_a_val(tab_caractère)


def liste_coordonnees_pastilles(tab):
    mask = (tab == P_G) | (tab == S_G)
    indices = np.argwhere(mask)
    return [tuple(coord) for coord in indices]


liste_coo_pastilles = liste_coordonnees_pastilles(tab)


def position_possible(tab):
    mask = tab != 0
    indices = np.argwhere(mask)
    return [tuple(coord) for coord in indices]


liste_position_possible = position_possible(tab)


def dico_ind_pastilles(liste_coo_pastilles):
    d = {}
    n = len(liste_coo_pastilles)
    for i in range(n):
        d[liste_coo_pastilles[i]] = i
    return d


dico_indice_pastilles = dico_ind_pastilles(liste_coo_pastilles)


def Q_init():
    Q = np.random.dirichlet(np.ones(4), (H, L, 2**nbr_pastilles))
    return Q


def pastilles_oui_ou_non_init(liste_coo_pastilles):
    n = len(liste_coo_pastilles)
    return [1 for _ in range(n)]


def tab_est_finale(tab_episode):
    return not np.any((tab_episode == P_G) | (tab_episode == S_G))


def model(s, a, tab_episode):
    action = dico[a]
    (i, j) = s
    ai, aj = action[0], action[1]
    if i + ai < 0 or i + ai > H - 1 or j + aj < 0 or j + aj > L - 1:
        s_autre = (i, j)
        tab_episode[i, j] = N
        R = tab_episode[i, j]
    elif tab_episode[i, j] != 0 and tab_episode[i + ai, j + aj] == 0:
        s_autre = (i, j)
        tab_episode[i, j] = N
        R = tab_episode[i, j]
    else:
        s_autre = (i + ai, j + aj)
        R = tab_episode[i + ai, j + aj]
        tab_episode[i + ai, j + aj] = N
    return (s_autre, R, tab_est_finale(tab_episode))


def Q_greedy(Q, s, pastilles_oui_ou_non, epsilon=0):
    if random.random() < epsilon:
        return random.randint(0, 3)
    else:
        val_binaire = "".join(map(str, pastilles_oui_ou_non))
        val_decimale = int(val_binaire, 2)
        return np.argmax(Q[s[0], s[1], val_decimale])


def Qlearning(alpha, gamma, epsilon_debut, epsilon_fin, episode):
    Q = Q_init()

    meilleur_score = 0
    pastilles_mangees_sur_1000_episodes = []
    moyenne_sur_1000_episodes = 0

    for i in range(episode):
        pastilles_oui_ou_non = pastilles_oui_ou_non_init(liste_coo_pastilles)

        epsilon = epsilon_debut - (epsilon_debut - epsilon_fin) * (i / episode)

        S = random.choice(liste_position_possible)

        tab_episode = copy.deepcopy(tab)

        terminal = False
        compteur = 0
        pastilles_mangees = 0
        while not terminal:
            A = Q_greedy(Q, S, pastilles_oui_ou_non, epsilon)
            S_autre, R, terminal = model(S, A, tab_episode)

            val_binaire = "".join(map(str, pastilles_oui_ou_non))
            val_decimale = int(val_binaire, 2)

            if S_autre in liste_coo_pastilles:
                index = dico_indice_pastilles[S_autre]
                pastilles_oui_ou_non[index] = 0

            val_binaire_autre = "".join(map(str, pastilles_oui_ou_non))
            val_decimale_autre = int(val_binaire_autre, 2)

            if R > 0:
                pastilles_mangees += 1

            Amax = Q_greedy(Q, S_autre, pastilles_oui_ou_non)
            Q[S[0], S[1], val_decimale, A] += alpha * (
                R
                + gamma * Q[S_autre[0], S_autre[1], val_decimale_autre, Amax]
                - Q[S[0], S[1], val_decimale, A]
            )

            S = (S_autre[0], S_autre[1])
            compteur += 1
            if compteur > 100:
                terminal = True
        pastilles_mangees_sur_1000_episodes.append(pastilles_mangees)
        if pastilles_mangees > meilleur_score:
            meilleur_score = pastilles_mangees

        if i % 1000 == 0:
            moyenne_sur_1000_episodes = sum(pastilles_mangees_sur_1000_episodes) / len(
                pastilles_mangees_sur_1000_episodes
            )
            pastilles_mangees_sur_1000_episodes = []
            print(
                f"Épisode {i}/{episode}, Meilleur score: {meilleur_score}, Moyenne: {moyenne_sur_1000_episodes}"
            )

    S = (7, 5)
    tab_exploitation = copy.deepcopy(tab)
    terminal = False
    chemin_optimal = []

    pastilles_oui_ou_non = pastilles_oui_ou_non_init(liste_coo_pastilles)
    compteur = 0

    while not terminal and compteur < 100:
        A = Q_greedy(Q, S, pastilles_oui_ou_non)
        S_autre, R, terminal = model(S, A, tab_exploitation)
        chemin_optimal.append(S_autre)
        if S_autre in liste_coo_pastilles:
            index = dico_indice_pastilles[S_autre]
            pastilles_oui_ou_non[index] = 0
        S = S_autre
        compteur += 1

    return (Q, chemin_optimal, terminal)


grid = np.zeros((H, L))
grid_p = np.zeros((H, L))

for i in range(H):
    for j in range(L):
        if tab[i, j] != 0:
            grid[i, j] = 1

for i in range(H):
    for j in range(L):
        grid_p[i, j] = tab[i, j]


def points(grid_p_actuel, i, j, positions_mangées):
    positions_mangées.append((i, j))
    for i in range(H):
        for j in range(L):
            if grid_p_actuel[i, j] == P_G and (i, j) not in positions_mangées:
                plt.scatter(j + 0.5, i + 0.5, color="black", s=8)
            elif grid_p_actuel[i, j] == S_G and (i, j) not in positions_mangées:
                plt.scatter(j + 0.5, i + 0.5, color="black", s=40)


def affichage(grid_p_actuel, i, j, positions_mangées):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="gray", extent=[0, L, H, 0])
    points(grid_p_actuel, i, j, positions_mangées)
    plt.scatter(j + 0.5, i + 0.5, color="red", s=60)
    plt.grid(False)
    plt.xticks(np.arange(L) + 0.5, labels=np.arange(L))
    plt.yticks(np.arange(H) + 0.5, labels=np.arange(H))
    plt.gca().set_xticks(np.arange(L + 1), minor=True)
    plt.gca().set_yticks(np.arange(H + 1), minor=True)
    plt.grid(which="minor", color="black", linewidth=0.5)
    plt.pause(0.001)
    plt.close()


def deplacement_Pac_Man(liste_nouvelles_position):
    Reward = 0
    Episode = 0
    print(Episode, Reward)
    i = 7
    j = 5
    positions_mangées_liste = []
    grid_p_actuel = np.copy(grid_p)
    n = len(liste_nouvelles_position)
    affichage(grid_p_actuel, i, j, positions_mangées_liste)
    for k in range(n):
        grid_p_actuel[i, j] = N
        i = liste_nouvelles_position[k][0]
        j = liste_nouvelles_position[k][1]
        if grid_p_actuel[i, j] == P_G:
            Reward += P_G
        if grid_p_actuel[i, j] == S_G:
            Reward += S_G
        if grid_p_actuel[i, j] == N:
            Reward += N
        grid_p_actuel[i, j] = N
        affichage(grid_p_actuel, i, j, positions_mangées_liste)
        Episode += 1
        print(Episode, Reward)


deplacement_Pac_Man(Qlearning(alpha, gamma, epsilon_debut, epsilon_fin, episode)[1])
