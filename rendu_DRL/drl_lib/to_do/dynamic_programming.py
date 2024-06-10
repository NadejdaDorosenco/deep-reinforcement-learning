from ..do_not_touch.mdp_env_wrapper import Env1
from ..do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction

import numpy as np

# Definition du MDP pour le Line World
S_Line_World = [0, 1, 2, 3, 4, 5, 6]
A_Line_World = [0, 1] # Gauche, Droite
R_Line_World = [-1.0, 0.0, 1.0]

#création des contrats liés à un MDP
def p_line_world(s, a, s_p, r):
    assert(s >= 0 and s <= 6)
    assert(s_p >= 0 and s_p <= 6)
    assert(a >= 0 and a <= 1)
    assert(r >= 0 and r <= 2)
    if s == 0 or s == 6:
        return 0.0
    if s + 1 == s_p and a == 1 and r == 1 and s != 5:
        return 1.0
    if s + 1 == s_p and a == 1 and r == 2 and s == 5:
        return 1.0
    if s - 1 == s_p and a == 0 and r == 1 and s != 1:
        return 1.0
    if s - 1 == s_p and a == 0 and r == 0 and s == 1:
        return 1.0
    return 0.0

def pi_random_line_world(s, a):
    if s == 0 or s == 6:
        return 0.0
    return 0.5

# Policy Evaluation
def policy_evaluation(S, A, R, p, pi, theta: float = 0.0000001) -> ValueFunction:
    V = {s: 0.0 for s in S}
    while True:
        delta = 0.0
        for s in S:
            old_v = V[s]
            total = 0.0
            for a in A:
                total_inter = 0.0
                for s_p in S:
                    for r in range(len(R)):            
                        total_inter += p(s, a, s_p, r) * (R[r] + 0.99999 * V[s_p])
                total_inter = pi(s, a) * total_inter
                total += total_inter
            V[s] = total
            delta = max(delta, np.abs(V[s] - old_v))
        if delta < theta:  
             return V 

def policy_evaluation_on_line_world() -> ValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    # TODO
    return policy_evaluation(S_Line_World, A_Line_World, R_Line_World, p_line_world, pi_random_line_world)

# Policy Iteration
def policy_iteration(S, A, R, p, theta=0.0000001, gamma=0.99999):
    # Initialisation aléatoire de la politique
    pi = {s: {a: 1/len(A) for a in A} for s in S}
    V = policy_evaluation(S, A, R, p, lambda s, a: pi[s][a], theta)

    while True:
        policy_stable = True
        for s in S:
            old_a = max(pi[s], key=pi[s].get) #retourne la clé de pi[s] dont la valeur est la plus grande.

            ## Calcul de q pour toutes les actions
            q_sa = [sum([p(s, a, s_p, r) * (R[r] + gamma * V[s_p]) for s_p in S for r in range(len(R))]) for a in A]
            # Mettre à jour la politique pour s avec la meilleure action
            best_a = A[np.argmax(q_sa)]
            
            pi[s] = {a: 1 if a == best_a else 0 for a in A}

            # Vérifier si la politique a changé                    
            if old_a != best_a:
                policy_stable = False

        if policy_stable:
            break
        else: # Si la politique a changé, mettre à jour V avec la nouvelle politique
            V = policy_evaluation(S, A, R, p, lambda s, a: pi[s][a], theta)

    return PolicyAndValueFunction(pi, V) 

def policy_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    return policy_iteration(S_Line_World, A_Line_World, R_Line_World, p_line_world)


# Value Iteration
def value_iteration(S, A, R, p, theta=0.0000001, gamma=0.99999):
    V = {s: 0 for s in S}  # initialiser V arbitrairement
    while True:
        # Policy Evaluation
        delta = 0
        for s in S:
            v = V[s]
            # Calculer la valeur maximale pour toutes les actions
            max_value = max([sum([p(s, a, s_p, r) * (R[r] + gamma * V[s_p]) for s_p in S for r in range(len(R))]) for a in A])
            V[s] = max_value
            delta = max(delta, np.abs(v - V[s]))

        if delta < theta:  # la valeur a suffisamment convergé
            break

    # Policy Improvement
    pi = {s: {a: 0 for a in A} for s in S}
    for s in S:
        # Calculer la valeur de q pour toutes les actions
        q_sa = [sum([p(s, a, s_p, r) * (R[r] + gamma * V[s_p]) for s_p in S for r in range(len(R))]) for a in A]
        # Mettre à jour la politique pour s avec la meilleure action
        # np.argmax trouve l'action qui donne la valeur maximale de q(s, a)
        best_a = np.argmax(q_sa)
        pi[s][best_a] = 1

    return PolicyAndValueFunction(pi, V)  # Pas besoin de convertir le tableau numpy en dict car V est déjà un dict

def value_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    return value_iteration(S_Line_World, A_Line_World, R_Line_World, p_line_world)


S_Grid_World = [(i, j) for i in range(5) for j in range(5)]  # The states are tuples representing the coordinates in the grid
A_Grid_World = [0, 1, 2, 3]  # Gauche, Droite, Haut, Bas
R_Grid_World = [-1.0, 0.0, 1.0]  # Rewards for the terminal states and the other states

# Uniform random policy for the grid world
def pi_random_grid_world(s, a):
    if s == (4, 4) or s == (0, 4):  # Etats terminaux
        return 0.0
    return 0.25  # 4 actions possible (haut, bas, droite, gauche) donc 1 chance sur 4 de faire l'action

# Probability transition function for the grid world
def p_grid_world(s, a, s_p, r):
    assert(s[0] >= 0 and s[0] <= 4 and s[1] >= 0 and s[1] <= 4)
    assert(s_p[0] >= 0 and s_p[0] <= 4 and s_p[1] >= 0 and s_p[1] <= 4)
    assert(a >= 0 and a <= 3)
    assert(r >= 0 and r <= 2)

    # If it's terminal state
    if s == (0, 4) or s == (4, 4):
        return 0.0

    if (s[0], s[1] - 1) == s_p and a == 0 and r == 1:  # gauche
        return 1.0
    if (s[0], s[1] + 1) == s_p and a == 1 : # droite
        if r == 1 and s != (0,3) and s != (3,4):
            return 1.0
        if r == 0 and s == (0,3):
            return 1.0
        if r == 2 and s == (4,3):
            return 1.0
    if  (s[0]-1, s[1]) == s_p and a == 2 : # haut
        if r == 1 and s != (1,4):
            return 1.0
        if r == 0 and s == (1,4):
            return 1.0
    if  (s[0]+1, s[1]) == s_p and a == 3 : # bas
        if r == 1 and s != (3,4):
            return 1.0
        if r == 2 and s == (3,4):
            return 1.0
    return 0.0

def policy_evaluation_on_grid_world() -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    # TODO
    return policy_evaluation(S_Grid_World, A_Grid_World, R_Grid_World, p_grid_world, pi_random_grid_world)

def policy_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # TODO
    return policy_iteration(S_Grid_World, A_Grid_World, R_Grid_World, p_grid_world)


def value_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # TODO
    return value_iteration(S_Grid_World, A_Grid_World, R_Grid_World, p_grid_world)

def pi_random_secret_env(s, a):
    num_actions = len(env.actions())
    if env.is_state_terminal(s):  
        return 0.0
    return 1.0 / num_actions  # Uniform random policy

def p_secret_env(s, a, s_p, r):
    return env.transition_probability(s, a, s_p, r)

def policy_evaluation_on_secret_env1() -> ValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    global env
    env = Env1()
    S = env.states()
    A = env.actions()
    R = env.rewards()
    
    return policy_evaluation(S, A, R, p_secret_env, pi_random_secret_env)



def policy_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    S = env.states().tolist() 
    A = env.actions().tolist() 
    R = env.rewards().tolist()
    p = env.transition_probability
    theta = 0.0000001  # Convergence threshold
    gamma = 0.99999  # Discount factor

    pi_and_v = policy_iteration(S, A, R, p, theta, gamma)

    return pi_and_v


def value_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Prints the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    S = env.states().tolist() 
    A = env.actions().tolist() 
    R = env.rewards().tolist()
    p = env.transition_probability
    return value_iteration(S, A, R, p)

def demo():

    print("policy_evaluation_on_line_world : " )
    print(policy_evaluation_on_line_world())
    print("\n")

    print("policy_iteration_on_line_world : " )
    print(policy_iteration_on_line_world())
    print("\n")

    print("value_iteration_on_line_world : " )
    print(value_iteration_on_line_world())
    print("\n")

    print("policy_evaluation_on_grid_world : " )
    print(policy_evaluation_on_grid_world())
    print("\n")

    print("policy_iteration_on_grid_world : " )
    print(policy_iteration_on_grid_world())
    print("\n")

    print("value_iteration_on_grid_world : " )
    print(value_iteration_on_grid_world())
    print("\n")

    print("policy_evaluation_on_secret_env1 : " )
    print(policy_evaluation_on_secret_env1())
    print("\n")

    print("policy_iteration_on_secret_env1 : " )
    print(policy_iteration_on_secret_env1())
    print("\n")

    print("value_iteration_on_secret_env1 : " )
    print(value_iteration_on_secret_env1())
    print("\n")
