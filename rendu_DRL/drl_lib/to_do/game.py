from typing import Optional
from os import access
import numpy as np


class Player():
    def __init__(self,sign:int,type:str) -> None:
        self.sign = sign
        self.is_winner = False
        self.type = type # 'H':human player, 'R': Random player, 'A'

    def play(self,available_actions,state_id=None,policy=None) -> Optional[int]:
        action_id=None
        if self.type == 'H':
            while True:
                action_id = input("Please enter your action id: ")
                if action_id in available_actions:
                    break
        elif self.type == 'R':
            action_id = np.random.choice(available_actions)
        else:
            if state_id is not None and policy is not None:
                action_id = max(policy[state_id],key=policy[state_id].get)
        return action_id

class TicTacToeEnv():
    def __init__(self,size=3) -> None:
        self.size=size
        self.board = np.zeros((size, size))
        self.actions=np.arange(size*size)
        self.players = np.array([Player(1,'R'),Player(2,'A')])

    
    def state_id(self) ->int:
        state = 0
        for i in range(self.size):
            for j in range(self.size):
                state += self.board[i][j] * pow(self.size, i * self.size + j)
        return int(state)

    def is_game_over(self) -> bool:
        term = False
        if len(self.available_actions_ids())==0:
            term = True
            # Check diagonals
        if (self.board[0][0]==self.board[1][1] == self.board[2][2]) and self.board[0][0] !=0:
            if self.players[0].sign == self.board[0][0]:
                self.players[0].is_winner = True
            else:
                self.players[1].is_winner = True
            return True
        elif (self.board[2][0]==self.board[1][1] == self.board[0][2]) and self.board[2][0] !=0:
            if self.players[0].sign == self.board[2][0]:
                self.players[0].is_winner = True
            else:
                self.players[1].is_winner = True
            return True
        else:
            for i in range(self.size):
                # Check horizontals
                if (self.board[i][0]==self.board[i][1] == self.board[i][2]) and self.board[i][0] !=0:
                    if self.players[0].sign == self.board[i][0]:
                        self.players[0].is_winner = True
                    else:
                        self.players[1].is_winner = True
                    return True
                # Check verticals
                elif (self.board[0][i]==self.board[1][i] == self.board[2][i]) and self.board[0][i] !=0:
                    if self.players[0].sign == self.board[0][i]:
                        self.players[0].is_winner = True
                    else:
                        self.players[1].is_winner = True
                    return True
        return term

    def act_with_action_id(self, player_sign:int,action_id: int):
        i= action_id // self.size
        j = action_id % self.size
        self.board[i][j]=player_sign

    def score(self) -> float:
        score = 0
        nb_coups = (self.board == 2).sum()
        if self.players[1].is_winner:
            score = 1
            if nb_coups<=3:
                score+=2
        if self.players[0].is_winner:
            score = -20
        
        return score

    def available_actions_ids(self):
        positions = []
        cpt =0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0:
                    positions.append(cpt)  
                cpt+=1
        return np.array(positions)

    def reset(self):
        self.board = np.zeros((self.size, self.size))
        self.players[0].is_winner = False
        self.players[1].is_winner = False

    def convertStateToBoard(self,state, b=3):
        if state == 0:
            return  np.zeros((self.size, self.size))
        digits = []
        while len(digits) < self.size*self.size:
            digits.append(int(state % b))
            state //= b
        digits = np.array(digits)
        return digits.reshape(self.size, self.size)