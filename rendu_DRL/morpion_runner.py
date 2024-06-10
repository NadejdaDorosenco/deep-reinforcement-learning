import json
import matplotlib.pyplot as plt
from random import choice, choices, random
import tkinter as tk
from tkinter import ttk

import pygame

# Importation des modules du jeu et des méthodes de renforcement à partir de la bibliothèque drl_lib
from drl_lib.to_do.game import TicTacToeEnv
from drl_lib.to_do.monte_carlo_methods import (
    monte_carlo_es_on_tic_tac_toe_solo,
    off_policy_monte_carlo_control_on_tic_tac_toe_solo,
    on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo,
    
)
from drl_lib.to_do.temporal_difference_learning import ( 
    q_learning_on_tic_tac_toe_solo, 
    sarsa_on_tic_tac_toe_solo,
    deep_q_learning_on_tic_tac_toe_solo,
    expected_sarsa_on_tic_tac_toe_solo
    )

# Initialisation de pygame
pygame.init()

# Définition des constantes pour l'affichage du jeu
largeur_fenetre, hauteur_fenetre = 470, 470
largeur_ligne = 10
rayon_cercle = 55
epaisseur_cercle = 15
epaisseur_croix = 15

# Création de la fenêtre de jeu
fenetre = pygame.display.set_mode((largeur_fenetre, hauteur_fenetre))
pygame.display.set_caption("Morpion")

# Définition des couleurs utilisées pour le jeu
blanc = (255, 255, 255)
noir = (0, 0, 0)
vert = (124, 252, 0)
rouge = (255, 0, 0)
couleur_fond = (242, 235, 211)
couleur_ligne = (173, 65, 54)
couleur_croix = (84, 84, 84)
couleur_cercle = (124, 252, 0)

# Définition des dimensions des blocs de la grille
hauteur_bloc = 150
largeur_bloc = 150

# Initialisation de l'horloge de jeu
horloge = pygame.time.Clock()

# Fonction pour dessiner la grille de jeu
def dessiner_grille():
    for i in range(1, 3):
        pygame.draw.line(fenetre, couleur_ligne, (0, i * hauteur_bloc), (largeur_fenetre, i * hauteur_bloc), largeur_ligne)
        pygame.draw.line(fenetre, couleur_ligne, (i * largeur_bloc, 0), (i * largeur_bloc, hauteur_fenetre), largeur_ligne)

# Fonction pour dessiner les figures (cercles et croix) sur la grille
def dessiner_figures(grille, taille):
    for ligne in range(taille):
        for colonne in range(taille):
            if grille[ligne][colonne] == 1:
                x = colonne * largeur_bloc + largeur_bloc // 2
                y = ligne * hauteur_bloc + hauteur_bloc // 2
                pygame.draw.circle(fenetre, couleur_cercle, (x, y), rayon_cercle, epaisseur_cercle)
            if grille[ligne][colonne] == 2:
                eps = 35
                pygame.draw.line(fenetre, couleur_croix, (colonne * largeur_bloc + eps, ligne * hauteur_bloc + hauteur_bloc - eps),
                                 (colonne * largeur_bloc + largeur_bloc - eps, ligne * largeur_bloc + eps), epaisseur_croix)
                pygame.draw.line(fenetre, couleur_croix, (colonne * largeur_bloc + eps, ligne * hauteur_bloc + eps),
                                 (colonne * largeur_bloc + largeur_bloc - eps, ligne * largeur_bloc + largeur_bloc - eps), epaisseur_croix)

# Fonction pour dessiner la fenêtre de jeu, y compris la grille et les figures
def dessiner_fenetre(grille, taille):
    fenetre.fill(couleur_fond)
    horloge.tick(3)
    dessiner_grille()
    dessiner_figures(grille, taille)
    pygame.display.flip()

# Fonction pour sauvegarder la politique dans un fichier JSON
def sauvegarder_pi(pi, nom_fichier):
    with open(nom_fichier, 'w') as fp:
        json.dump(pi, fp)

# Fonction pour trouver l'action correspondante à partir des coordonnées x et y
def trouver_action(x, y, actions, taille_grille):
    for action in actions:
        if x == action % taille_grille and y == action // taille_grille:
            return action
    return None

# Fonction pour sélectionner la méthode d'apprentissage pour chaque joueur
def select_method():
    window = tk.Tk()
    window.title('Select Method')

    # Fonction pour sélectionner la méthode d'apprentissage et fermer la fenêtre tkinter
    def select():
        global pi_et_q1, pi_et_q2, human_vs_ai
        selection1 = combo1.get()
        selection2 = combo2.get()
        window.destroy()

        human_vs_ai = selection2 == 'Human'
        
        # Assignation de la méthode d'apprentissage pour le joueur 1
        if selection1 == 'Monte Carlo ES':
            pi_et_q1 = monte_carlo_es_on_tic_tac_toe_solo()
        elif selection1 == 'Off-policy Monte Carlo':
            pi_et_q1 = off_policy_monte_carlo_control_on_tic_tac_toe_solo()
        elif selection1 == 'On-policy first-visit Monte Carlo':
            pi_et_q1 = on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo()
        elif selection1 == 'Sarsa':
            pi_et_q1 = sarsa_on_tic_tac_toe_solo()
        elif selection1 == 'Expected Sarsa':
            pi_et_q1 = sarsa_on_tic_tac_toe_solo()
        else:
            pi_et_q1 = q_learning_on_tic_tac_toe_solo()

        # Assignation de la méthode d'apprentissage pour le joueur 2
        if not human_vs_ai:
            if selection2 == 'Monte Carlo ES':
                pi_et_q2 = monte_carlo_es_on_tic_tac_toe_solo()
            elif selection2 == 'Off-policy Monte Carlo':
                pi_et_q2 = off_policy_monte_carlo_control_on_tic_tac_toe_solo()
            elif selection2 == 'On-policy first-visit Monte Carlo':
                pi_et_q2 = expected_sarsa_on_tic_tac_toe_solo()
            elif selection1 == 'Sarsa':
                pi_et_q1 = sarsa_on_tic_tac_toe_solo()
            elif selection1 == 'Expected Sarsa':
                pi_et_q1 = sarsa_on_tic_tac_toe_solo()
            else:
                pi_et_q2 = q_learning_on_tic_tac_toe_solo()

    # Création des combobox pour sélectionner la méthode d'apprentissage pour chaque joueur
    options = ['Monte Carlo ES', 'Off-policy Monte Carlo', 'On-policy first-visit Monte Carlo','Sarsa','Expected Sarsa', 'Q-learning', 'Human']

    combo1 = ttk.Combobox(window, values=options)
    combo1.grid(column=0, row=0)
    combo2 = ttk.Combobox(window, values=options)
    combo2.grid(column=0, row=1)

    # Bouton pour valider la sélection et fermer la fenêtre
    button = ttk.Button(window, text="OK", command=select)
    button.grid(column=1, row=0)

    # Exécution de la boucle tkinter
    window.mainloop()

# Fonction principale pour exécuter le jeu
def main():
    global pi_et_q1, pi_et_q2, human_vs_ai
    en_cours = True
    env = TicTacToeEnv()

    select_method()

    toutes_les_cles1 = pi_et_q1.pi.keys()
    toutes_les_cles2 = None if human_vs_ai else pi_et_q2.pi.keys()

    current_player = 1

    cpt = 0
    nb_parties = 100
    while en_cours:
        debut = True
        for evenement in pygame.event.get():
            if evenement.type == pygame.QUIT:
                en_cours = False
        while not env.is_game_over():
            etat = env.state_id()
            if debut:
                action = choice(env.available_actions_ids())
                debut = False
            else:
                if current_player == 1:
                    if etat not in toutes_les_cles1:
                        pi_et_q1.pi[etat] = {action: random() for action in env.available_actions_ids()}
                    probabilites = [pi_et_q1.pi[etat][action] for action in env.available_actions_ids()]
                    model = pi_et_q1
                else:  # Second player's turn
                    if human_vs_ai:
                        action = None
                        while action is None:
                            for evenement in pygame.event.get():
                                if evenement.type == pygame.MOUSEBUTTONDOWN:
                                    colonne = evenement.pos[0] // largeur_bloc
                                    ligne = evenement.pos[1] // hauteur_bloc
                                    action = trouver_action(colonne, ligne, env.available_actions_ids(), env.size)
                        env.act_with_action_id(current_player, action)
                        current_player = 3 - current_player
                        continue
                    else:
                        if etat not in toutes_les_cles2:
                            pi_et_q2.pi[etat] = {action: random() for action in env.available_actions_ids()}
                        probabilites = [pi_et_q2.pi[etat][action] for action in env.available_actions_ids()]
                        model = pi_et_q2

                if sum(probabilites) <= 0.0:
                    action = choice(env.available_actions_ids())
                else:
                    action = choices(env.available_actions_ids(), weights=probabilites)[0]

            env.act_with_action_id(current_player, action)
            current_player = 3 - current_player
            dessiner_fenetre(env.board, env.size)

        env.is_game_over()

        if env.players[1].is_winner:
            cpt += 1
            print("Le joeur 1 a gagné")
        elif env.players[0].is_winner:
            print("Le joueur 2 a gagné")
        else:
            print("vous etes nul tous les deux")

        env.reset()
        dessiner_fenetre(env.board, env.size)
        nb_parties -= 1
        if nb_parties == 0:
            en_cours = False

        print("Pourcentage de réussite :", cpt, "%")
        

    pygame.quit()

if __name__ == "__main__":
    main()