from collections import defaultdict
from random import random, choice, choices

from ..to_do.game import TicTacToeEnv
from ..do_not_touch.result_structures import PolicyAndActionValueFunction
from ..do_not_touch.single_agent_env_wrapper import Env2
import numpy as np

def monte_carlo_es_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:

    env = TicTacToeEnv()
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    actions = env.available_actions_ids()
    Q = defaultdict(lambda: {a:random() for a in actions})
    pi = defaultdict(lambda: {a:random() for a in actions})
    num_episodes = 70000

    for i in range(num_episodes):
        env.reset()
        s0 = env.state_id()
        a0 = choice(env.available_actions_ids())

        # faire jouer player[1]
        env.act_with_action_id(env.players[1].sign,a0)

        # faire jouer player[0]
        rand_action = env.players[0].play(env.available_actions_ids())
        env.act_with_action_id(env.players[0].sign,rand_action)
        
        s_history = [s0]
        a_history = [a0]
        s_p_history = [env.state_id()]
        r_history = [env.score()]

        while not env.is_game_over():
            s = env.state_id()
            pis = [pi[s][a] for a in env.available_actions_ids()]
            av_actions = env.available_actions_ids()
            if max(pis) == 0.0:
                a = choice(av_actions)
            else:
                a = choices(av_actions, weights=pis)[0]


            # faire jouer player[1]
            env.act_with_action_id(env.players[1].sign,a)
            
            if not env.is_game_over():
                # faire jouer player[0]
                rand_action = env.players[0].play(env.available_actions_ids())
                env.act_with_action_id(env.players[0].sign,rand_action)
            game_over = env.is_game_over()
            s_history.append(s)
            a_history.append(a)
            s_p_history.append(env.state_id())
            r_history.append(env.score())

        G = 0
        discount = 0.999
        for t in reversed(range(len(s_history))):
            G = discount * G + r_history[t]
            s_t = s_history[t]
            a_t = a_history[t]

            appear = False
            for t_p in range(t - 1):
                if s_history[t_p] == s_t and a_history[t_p] == a_t:
                    appear = True
                    break
            if appear:
                continue

            returns_sum[(s_t,a_t)] += G
            returns_count[(s_t,a_t)] += 1.0
            Q[s_t][a_t] = returns_sum[(s_t,a_t)]/returns_count[(s_t,a_t)]
            pi[s_t] = {a:0.0 for a in Q[s_t].keys()}
            best_action = max(Q[s_t],key=Q[s_t].get)
            pi[s_t][best_action] = 1.0

        pi_and_Q = PolicyAndActionValueFunction(pi,Q)

    return pi_and_Q


def on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:

    def epsilon_greedy_policy(env, Q, epsilon, state, A):
        rand = random()
        if rand > epsilon:
            best_action = max(Q[state], key=Q[state].get)
            for a in env.available_actions_ids():
                A[state][a] = 0
            A[state][best_action] = 1.0
        else:
            best_action = choice(env.available_actions_ids())

        return A[state]

    epsilon = 0.3
    num_episodes = 50000

    env = TicTacToeEnv()

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    Q = defaultdict(lambda: {a: random() for a in env.available_actions_ids()})
    pi = defaultdict(lambda: {a: random() for a in env.available_actions_ids()})
    for i_episode in range(1, num_episodes + 1):
        if i_episode % (num_episodes / 5) == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes))

        env.reset()
        pair_history = []
        s_history = []
        a_history = []
        s_p_history = []
        r_history = []
        while not env.is_game_over():
            state = env.state_id()
            pi[state] = epsilon_greedy_policy(env, Q, epsilon, state, pi)
            keys = []
            for i in pi[state].keys():
                keys.append(i)
            if len(keys) > 1:
                a = choices(keys, weights=pi[state])[0]
            else:
                a = choice(keys)

            env.act_with_action_id(env.players[1].sign, a)

            if not env.is_game_over():
                rand_action = env.players[0].play(env.available_actions_ids())
                env.act_with_action_id(env.players[0].sign, rand_action)

            game_over = env.is_game_over()
            r = env.score()

            s_history.append(state)
            a_history.append(a)
            s_p_history.append(env.state_id())
            r_history.append(r)
            pair_history.append(((state, a), r))

        G = 0
        for ((s, a), r) in pair_history:
            first_occurence_idx = next(
                i for i, (s_a, r) in enumerate(pair_history) if s_a == (s, a))
            G = sum([r for ((s, a), r) in pair_history[first_occurence_idx:]])

            returns_sum[(s, a)] += G
            returns_count[(s, a)] += 1.0
            Q[s][a] = returns_sum[(s, a)] / returns_count[(s, a)]

        pi_and_Q = PolicyAndActionValueFunction(pi, Q)
    return pi_and_Q

def create_target_policy(Q):
    
    def policy_fn(state):
        A = {a:0.0 for a in Q[state].keys()}
        best_action = max(Q[state],key=Q[state].get)
        A[best_action] = 1.0
        return A
    return policy_fn

    
def off_policy_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:

    env = TicTacToeEnv()

    actions = env.available_actions_ids()
    Q = defaultdict(lambda: {a:random() for a in actions})
    C = defaultdict(lambda: {a:0.0 for a in actions})

    pi = defaultdict(lambda: {a:random() for a in actions})
    target_policy = create_target_policy(Q)
    num_episodes = 80000

    
    for i_episode in range(1, num_episodes+1):

        env.reset()
        s0 = env.state_id()
        pis = [pi[s0][a] for a in env.available_actions_ids()]
        a0 = choices(env.available_actions_ids(), weights=pis)[0]

        # faire jouer player[1]
        env.act_with_action_id(env.players[1].sign,a0)

        # faire jouer player[0]
        rand_action = env.players[0].play(env.available_actions_ids())
        env.act_with_action_id(env.players[0].sign,rand_action)

        s_history = [s0]
        a_history = [a0]
        s_p_history= [env.state_id()]
        r_history= [env.score()]
        
        while(not env.is_game_over()):
            s = env.state_id()
            pis = [pi[s][a] for a in env.available_actions_ids()]
            av_actions = env.available_actions_ids()
            if max(pis) == 0.0:
                a = choice(av_actions)
            else:
                a = choices(av_actions, weights=pis)[0]

            # faire jouer player[1]
            env.act_with_action_id(env.players[1].sign,a)
            
            if not env.is_game_over():
                # faire jouer player[0]
                rand_action = env.players[0].play(env.available_actions_ids())
                env.act_with_action_id(env.players[0].sign,rand_action)
            env.is_game_over()
            s_history.append(s)
            a_history.append(a)
            s_p_history.append(env.state_id())
            r_history.append(env.score())
            
        G = 0.0
        W = 1.0
        discount=0.999
        
        for t in range(len(s_history))[::-1]:
            state, action, reward = s_history[t],a_history[t],r_history[t]
            G = discount*G + reward
            C[state][action] += W
            Q[state][action] += (W/C[state][action]) * (G - Q[state][action])
            best_action = max(Q[state],key=Q[state].get)

            if action != best_action:
                break
                
            W = W * (target_policy(state)[action]/pi[state][action])
    
    final_policy = {state:target_policy(state) for state in Q.keys()}
        
    return PolicyAndActionValueFunction(final_policy,Q)

def monte_carlo_es_on_secret_env2() -> PolicyAndActionValueFunction:

    env = Env2()
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    actions = env.available_actions_ids()
    Q = defaultdict(lambda: {a:0.0 for a in actions})
    pi = defaultdict(lambda: {a:random() for a in actions})

    num_episodes = 10000
    for i in range(num_episodes):
        env.reset()
        s0 = env.state_id()
        a0 = choice(env.available_actions_ids())

        env.act_with_action_id(a0)
        s_history = [s0]
        a_history = [a0]
        s_p_history = [env.state_id()]
        r_history = [env.score()]

        while not env.is_game_over():
            s = env.state_id()
            pis = [pi[s][a] for a in env.available_actions_ids()]
            a = choices(env.available_actions_ids(), weights=pis)[0]
            env.act_with_action_id(a)
            s_history.append(s)
            a_history.append(a)
            s_p_history.append(env.state_id())
            r_history.append(env.score())
        G = 0
        for t in reversed(range(len(s_history))):
            G = 0.999 * G + r_history[t]
            s_t = s_history[t]
            a_t = a_history[t]

            appear = False
            for t_p in range(t - 1):
                if s_history[t_p] == s_t and a_history[t_p] == a_t:
                    appear = True
                    break
            if appear:
                continue

            returns_sum[(s_t,a_t)] += G
            returns_count[(s_t,a_t)] += 1.0

            Q[s_t][a_t] = returns_sum[(s_t,a_t)]/returns_count[(s_t,a_t)]
            pi[s_t]={a:0.0 for a in actions}
            best_action = max(Q[s_t],key=Q[s_t].get)
            pi[s_t][best_action] = 1.0
            
            pi_and_Q = PolicyAndActionValueFunction(pi,Q)

        return pi_and_Q


def on_policy_first_visit_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:

    env = Env2()
    def epsilon_greedy_policy(env, Q, epsilon, state, A):
        rand = random()
        if rand > epsilon:
            best_action = max(Q[state], key=Q[state].get)
            for a in env.available_actions_ids():
                A[state][a] = 0
            A[state][best_action] = 1.0
        else:
            best_action = choice(env.available_actions_ids())

        return A[state]

    epsilon = 0.1
    num_episodes = 10000

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    Q = defaultdict(lambda: {a: 0.0 for a in env.available_actions_ids()})
    pi = defaultdict(lambda: {a : random() for a in env.available_actions_ids()})
    final_policy={}

    for i_episode in range(1, num_episodes + 1):

        env.reset()
        pair_history = []
        s_history = []
        while not env.is_game_over():
            state = env.state_id()
            pi[state] = epsilon_greedy_policy(env, Q, epsilon, state, pi)
            keys = []
            for i in pi[state].keys():
                keys.append(i)
            if len(keys) > 1:
                a = choices(keys, weights=pi[state])[0]
            else:
                a = choice(keys)

            env.act_with_action_id(a)
            r = env.score()
            pair_history.append(((state, a), r))
            s_history.append(state)
        G = 0
        for ((s, a), r) in pair_history:
            first_occurence_idx = next(
                i for i, (s_a, r) in enumerate(pair_history) if s_a == (s, a))
            G = sum([r for ((s, a), r) in pair_history[first_occurence_idx:]])

            returns_sum[(s, a)] += G
            returns_count[(s, a)] += 1.0
            Q[s][a] = returns_sum[(s, a)] / returns_count[(s, a)]

        for state in Q.keys():
            final_policy[state] = {a:0.0 for a in Q[state].keys()}
            best_action = max(Q[state],key=Q[state].get)
            final_policy[state][best_action] = 1.0

        pi_and_Q = PolicyAndActionValueFunction(final_policy, Q)

    return pi_and_Q

def create_behaviour_policy(actions):
    def policy_fn(observation):
        A = {a:1.0/len(actions) for a in actions}
        return A

    return policy_fn

def off_policy_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:   

    env = Env2()                                                                                                                                        
    actions = env.available_actions_ids()                                              
    Q = defaultdict(lambda: {a: random() for a in actions})                            
    C = defaultdict(lambda: {a: 0.0 for a in actions})   
    pi = defaultdict(lambda: {a:0.0 for a in actions})                             
    policy_behaviour = create_behaviour_policy(actions)            
    target_policy = create_target_policy(Q)                                                                                                                         
    num_episodes = 10000                                                               
    for i_episode in range(1, num_episodes + 1):                                       

        env.reset()                                                                    
        s0 = env.state_id()                                                                                                                                                                             
        pi[s0] = policy_behaviour(s0)
        print("pi = " ,pi)
        pis = [pi[s0][a] for a in env.available_actions_ids()]                                                                                                                                                                     
        a0 = choices(env.available_actions_ids(),pis)[0]                                   
        env.act_with_action_id(a0)                                                                                                                             
        s_history = []                                                                 
        a_history = []                                                                 
        s_p_history = []                                                               
        r_history = []   

        while(not env.is_game_over()):                                                 
            s = env.state_id()                                                         
            pi[s] = policy_behaviour(s)
            pis = [pi[s][a] for a in env.available_actions_ids()]                                                                                           
            a = choices(env.available_actions_ids(),pis)[0]    

            # faire jouer player[1]                                                    
            env.act_with_action_id(a)                                                  

            env.is_game_over()

            s_history.append(s)                                                                                                                         
            a_history.append(a)                                                        
            s_p_history.append(env.state_id())                                         
            r_history.append(env.score())                                              

        G = 0.0                                                                        
        W = 1.0                                                                        
        discount = 0.999                                                               
        for t in range(len(s_history))[::-1]:                                          
            state, action, reward = s_history[t], a_history[t], r_history[t]           
            G = discount * G + reward                                                                                                
            C[state][action] += W                                                      
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])        
            best_action = max(Q[state], key=Q[state].get)                              
            if action != best_action:                                                  
                break                                                                  

            W = W * (target_policy(state)[action] / policy_behaviour(state)[action])   
    final_policy = {state: target_policy(state) for state in Q.keys()}                 

    return PolicyAndActionValueFunction(final_policy,Q) 


def demo():
    #print("monte_carlo_es_on_tic_tac_toe")
    #print(monte_carlo_es_on_tic_tac_toe_solo())
    #print("on_policy_first_visit_monte_carlo_control_on_tic_tac_toe")
    #print(on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo())
    #print("off_policy_first_visit_monte_carlo_control_on_tic_tac_toe")
    #print(off_policy_monte_carlo_control_on_tic_tac_toe_solo())

    #print("secret part ")
    #print("secret env 2:monte_carlo_es_on_secret_env2")
    #print(monte_carlo_es_on_secret_env2())
    print("secret env 2:on_policy_first_visit_monte_carlo_control_on_secret_env2")
    print(on_policy_first_visit_monte_carlo_control_on_secret_env2())
    #print("secret env 2: off_policy_monte_carlo_control_on_secret_env2")
    #print(off_policy_monte_carlo_control_on_secret_env2())