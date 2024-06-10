import itertools
from collections import defaultdict
from random import random, choice, choices

from ..do_not_touch.result_structures import PolicyAndActionValueFunction
from ..do_not_touch.single_agent_env_wrapper import Env3
from ..do_not_touch.result_structures import PolicyAndActionValueFunction


from ..to_do.game import TicTacToeEnv

from collections import defaultdict
from random import random, choice, choices
import numpy as np

import matplotlib.pyplot as plt

def sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    env = TicTacToeEnv()

    epsilon = 0.9
    num_episodes = 50000
    max_steps = 100
    alpha = 0.05
    gamma = 0.95

    Q = defaultdict(lambda: {a: random() for a in env.available_actions_ids()})
    final_policy = {}

    def choose_action(env):
        s = env.state_id()
        if random() < epsilon:
            action = choice(env.available_actions_ids())
        else:
            action = max(Q[s], key=Q[s].get)
        return action

    reward = 0

    for i_episode in range(1, num_episodes + 1):
        env.reset()
        pred_state = 0
        pred_action = 0
        state1 = env.state_id()
        action1 = choose_action(env)
        Q[state1]
        t = 0
        while t < max_steps:
            env.act_with_action_id(env.players[1].sign, action1)

            if not env.is_game_over():
                rand_action = env.players[0].play(env.available_actions_ids())
                env.act_with_action_id(env.players[0].sign, rand_action)

            if env.is_game_over():
                reward = env.score()
                prediction = Q[pred_state][pred_action]
                target = reward + gamma * Q[state2][action2]
                Q[state1][action1] = Q[state1][action1] + alpha * (target - prediction)
                break
            else:
                state2 = env.state_id()
                action2 = choose_action(env)
                reward = env.score()

            prediction = Q[state1][action1]
            target = reward + gamma * Q[state2][action2]
            Q[state1][action1] = Q[state1][action1] + alpha * (target - prediction)
            pred_state = state1
            state1 = state2
            pred_action = action2
            action1 = action2

            if env.is_game_over():
                break
    for state in Q.keys():
        actions = np.array(list(Q[state].keys()))
        final_policy[state] = {a:0.0 for a in actions}
        best_action = max(Q[state], key=Q[state].get)
        final_policy[state][best_action]=1.0

    return PolicyAndActionValueFunction(final_policy,Q)


def epsilon_greedy_policy(current_actions, Q, epsilon, state):
    A = defaultdict(
        lambda: {
            a: 1 * epsilon / len(current_actions)
            for a in current_actions
        }
    )
    # Only consider actions in the current action space
    best_action = max(current_actions, key=lambda a: Q[state].get(a, 0))  # Use 0 if action is not found in Q[state]
    A[state][best_action] += (1.0 - epsilon)
    return A[state]




def q_learning_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = TicTacToeEnv()
    num_episodes = 70000
    discount_factor = 1.0
    epsilon = 0.1
    alpha = 0.6

    actions = env.available_actions_ids()
    Q = defaultdict(lambda: {a:0.0 for a in actions})
    final_policy = {}

    for ith_episode in range(num_episodes):

        env.reset()

        for t in itertools.count():
            state = env.state_id()
            pi= epsilon_greedy_policy(actions, Q, epsilon, env.state_id())
            pis = [pi[a] for a in env.available_actions_ids()]

            if max(pis) == 0.0:
                action = choice(env.available_actions_ids())
            else:
                action = choices(env.available_actions_ids(), weights=pis)[0]

            env.act_with_action_id(env.players[1].sign,action)

            if not env.is_game_over():

                # faire jouer player[0]
                rand_action = env.players[0].play(env.available_actions_ids())
                env.act_with_action_id(env.players[0].sign,rand_action)

            best_action = max(Q[state],key=Q[state].get)
            env.is_game_over()

            td_target = env.score() + discount_factor * Q[env.state_id()][best_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if env.is_game_over():
                break

    for state in Q.keys():
        actions = np.array(list(Q[state].keys()))
        final_policy[state] = epsilon_greedy_policy(actions,Q,epsilon,state)

    pi_and_Q = PolicyAndActionValueFunction(final_policy, Q)
    return pi_and_Q


def deep_q_learning_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = TicTacToeEnv()
    num_episodes = 70000
    discount_factor = 1.0
    epsilon = 0.1
    alpha = 0.6

    actions = env.available_actions_ids()
    Q = defaultdict(lambda: {a: 0.0 for a in actions})
    final_policy = {}

    for ith_episode in range(num_episodes):

        env.reset()

        for t in itertools.count():
            state = env.state_id()
            pi = epsilon_greedy_policy(actions, Q, epsilon, env.state_id())
            pis = [pi[a] for a in env.available_actions_ids()]

            if max(pis) == 0.0:
                action = choice(env.available_actions_ids())
            else:
                action = choices(env.available_actions_ids(), weights=pis)[0]

            env.act_with_action_id(env.players[1].sign, action)

            if not env.is_game_over():
                # faire jouer player[0]
                rand_action = env.players[0].play(env.available_actions_ids())
                env.act_with_action_id(env.players[0].sign, rand_action)

            best_action = max(Q[state], key=Q[state].get)
            env.is_game_over()

            td_target = env.score() + discount_factor * Q[env.state_id()][best_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if env.is_game_over():
                break

    for state in Q.keys():
        actions = np.array(list(Q[state].keys()))
        final_policy[state] = epsilon_greedy_policy(actions, Q, epsilon, state)

    pi_and_Q = PolicyAndActionValueFunction(final_policy, Q)
    return pi_and_Q


def expected_sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = TicTacToeEnv()
    epsilon = 0.1
    num_episodes = 50000
    alpha = 0.5
    gamma = 0.95

    Q = defaultdict(lambda: {a: 0.0 for a in env.available_actions_ids()})
    final_policy = {}

    for i_episode in range(num_episodes):
        env.reset()
        state = env.state_id()
        action = max(Q[state], key=Q[state].get)

        for t in itertools.count():
            env.act_with_action_id(env.players[1].sign, action)
            
            next_state = env.state_id()
            next_action = max(Q[next_state], key=Q[next_state].get)

            # Check if the game is over
            if env.is_game_over():
                Q[state][action] += alpha * (env.score() - Q[state][action])
                break

            # Compute the expected Q value
            expected_q = sum(pi * Q[next_state][a] for a, pi in epsilon_greedy_policy(env.available_actions_ids(), Q, epsilon, next_state).items())

            # Update Q value
            Q[state][action] += alpha * (env.score() + gamma * expected_q - Q[state][action])

            state = next_state
            action = next_action

    # Update final policy
    for state in Q.keys():
        actions = np.array(list(Q[state].keys()))
        final_policy[state] = epsilon_greedy_policy(actions,Q,epsilon,state)

    pi_and_Q = PolicyAndActionValueFunction(final_policy, Q)
    return pi_and_Q


def sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    epsilon = 0.9
    num_episodes = 50000
    max_steps = 100
    alpha = 0.05
    gamma = 0.95

    episode_lengths = defaultdict(float)
    episode_rewards = defaultdict(float)

    Q = defaultdict(lambda: {a: 0.0 for a in env.available_actions_ids()})

    def choose_action(state, env, Q):
        if np.random.uniform(0, 1) < epsilon:
            action = choice(env.available_actions_ids())
        else:
            action = max(Q[state], key=Q[state].get)
        return action

    for i_episode in range(1, num_episodes + 1):
        if i_episode % (num_episodes/5) == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes))
        env.reset()
        state1 = env.state_id()
        action1 = choose_action(state1, env, Q)
        Q[env.state_id()]
        t=0
        while t < max_steps:
            env.act_with_action_id(action1)
            state2 = env.state_id()
            reward = env.score()
            action2 = choose_action(state2, env, Q)
            Q[env.state_id()]

            # Learning the Q-value
            Q[state1][action1] += alpha * (reward + gamma * Q[state2][action2] - Q[state1][action1])

            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = t

            state1 = state2
            action1 = action2
            t += 1

            if env.is_game_over():
                break

    return Q

def q_learning_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    num_episodes = 50000
    discount_factor = 1.0
    epsilon = 0.1
    alpha = 0.6

    Q = defaultdict(lambda: {a: 0.0 for a in env.available_actions_ids()})
    final_policy = {}

    for ith_episode in range(num_episodes):
        env.reset()
        for t in itertools.count():
            state = env.state_id()

            # Get the available actions for the current state
            actions = env.available_actions_ids()

            pi = epsilon_greedy_policy(actions, Q, epsilon, state)
            pis = [pi[a] for a in actions]
            action = choices(actions, weights=pis)[0]

            env.act_with_action_id(action)

            env.is_game_over()

            # Check if the new state is in Q, if not add it
            if env.state_id() not in Q:
                Q[env.state_id()] = {a: 0.0 for a in env.available_actions_ids()}

            best_action = max(Q[env.state_id()], key=Q[env.state_id()].get)  # Compute best_action for the new state

            td_target = env.score() + discount_factor * Q[env.state_id()][best_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if env.is_game_over():
                break
            
    for state in Q.keys():
        final_policy[state] = epsilon_greedy_policy(actions,Q,epsilon,state)
    pi_and_Q = PolicyAndActionValueFunction(final_policy, Q)
    return pi_and_Q



def expected_sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    num_episodes = 70000
    discount_factor = 1.0
    epsilon = 0.1
    alpha = 0.6

    actions = env.available_actions_ids()
    Q = defaultdict(lambda: {a: 0.0 for a in actions})
    final_policy = {}

    for ith_episode in range(num_episodes):
        env.reset()
        state = env.state_id()

        while not env.is_game_over():
            # Get the available actions for the current state
            actions = env.available_actions_ids()

            pi = epsilon_greedy_policy(actions, Q, epsilon, state)
            action_probs = [pi.get(a, 0) for a in actions]
            action = choices(actions, weights=action_probs)[0]

            env.act_with_action_id(action)

            next_state = env.state_id()

            # Check if the new state is in Q, if not add it
            if next_state not in Q:
                Q[next_state] = {a: 0.0 for a in actions}

            # Compute the expected value of the next state-action pairs
            expected_value = sum([pi.get(a, 0) * Q[next_state].get(a, 0) for a in actions])

            # Check if the current state-action pair exists in Q, if not add it
            if state not in Q:
                Q[state] = {a: 0.0 for a in actions}
            if action not in Q[state]:
                Q[state][action] = 0.0

            # Update the Q-value of the current state-action pair
            td_target = env.score() + discount_factor * expected_value
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            state = next_state

    # Construct the final policy
    for state in Q.keys():
        final_policy[state] = epsilon_greedy_policy(actions, Q, epsilon, state)
    
    pi_and_Q = PolicyAndActionValueFunction(final_policy, Q)
    return pi_and_Q
    


def demo():
    print(deep_q_learning_on_tic_tac_toe_solo())
    print(sarsa_on_tic_tac_toe_solo())
    print(q_learning_on_tic_tac_toe_solo())
    print(expected_sarsa_on_tic_tac_toe_solo())

    print(sarsa_on_secret_env3()[0])
    print(q_learning_on_secret_env3())
    print(expected_sarsa_on_secret_env3())