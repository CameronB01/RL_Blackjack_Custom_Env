import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random



class BlackEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self):
        super().__init__()
        self.deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4
        self.player_hand = []
        self.dealer_hand = []
        self.action_space = spaces.Discrete(2)  # Hit or Stand
        self.observation_space = spaces.Dict({
            'player_score': spaces.Discrete(32),  # Player's current sum (0-31)
            'dealer_card': spaces.Discrete(13),  # Dealer's face up card (1-10)
            'usable_ace': spaces.Discrete(2)    # Usable ace (0: no, 1: yes)
        })

    def deal_initial(self):
        random.shuffle(self.deck)
        self.player_hand = [self.deck.pop(), self.deck.pop()]
        self.dealer_hand = [self.deck.pop(), self.deck.pop()]

    def calculate_score(self, hand):
        score = sum(hand)
        if score > 21 and 11 in hand:
            hand.remove(11)
            hand.append(1)
            score = sum(hand)
        return score

    def player_turn(self, action):
        if action == 1:  # Hit
            self.player_hand.append(self.deck.pop())
            player_score = self.calculate_score(self.player_hand)
            if player_score > 21:
                return player_score, False, False  # Player busted
            else:
                return player_score, True, False
        elif action == 0:  # Stand
            return self.calculate_score(self.player_hand), False, True

    def dealer_turn(self):
        # if self.calculate_score(self.player_hand) < self.calculate_score(self.dealer_hand):
        #     return self.calculate_score(self.dealer_hand)
        
        # while self.calculate_score(self.dealer_hand) < self.calculate_score(self.player_hand):
        while self.calculate_score(self.dealer_hand) < 17:
            self.dealer_hand.append(self.deck.pop())
        dealer_score = self.calculate_score(self.dealer_hand)
        return dealer_score


    def step(self, action):
        assert self.action_space.contains(action)
        player_score, self.new_action, self.done = self.player_turn(action)

        if self.new_action:
            self.render()
            reward = 0.0000001
            return self.get_observation(), reward, False, False, {}
        elif self.done:
            dealer_score = self.dealer_turn()
            if dealer_score > 21 or player_score > dealer_score:
                reward = 1
            elif player_score < dealer_score:
                reward = -1
            else:
                reward = 0
            self.render()
            return self.get_observation(), reward, True, False, {}
        else:
            self.render()
            reward = -1
            return self.get_observation(), reward, True, False, {}

    def reset(self, seed=None):
        self.deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4
        self.player_hand = []
        self.dealer_hand = []
        self.deal_initial()
        return self.get_observation(), {}

    def get_observation(self):
        observation_dict = {
            'player_score': self.calculate_score(self.player_hand),
            'dealer_card': self.dealer_hand[0],
            'usable_ace': 1 if 11 in self.player_hand else 0
        }
        return observation_dict

    def render(self, mode='human'):
        print(f"Player's Hand: {self.player_hand}, Score: {self.calculate_score(self.player_hand)}")
        print(f"Dealer's Hand: {self.dealer_hand}, Score: {self.calculate_score(self.dealer_hand)}")

    def close(self):
        pass
