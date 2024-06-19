# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 10:49:10 2023

@author: ZhaoAidong
"""

import numpy as np


class GradientBandits():
    
    def __init__(self,numActions = 2, alpha = 0.3):
        self.numActions = numActions
        self.H = np.zeros(numActions)
        self.alpha = alpha
        self.R_bar = 0
        self.cnt = 0
        
        
    def draw(self):
        self.distr()
        choice = np.random.uniform(0, 1)
        choiceIndex = 0
        for prob in self.probabilityDistribution:
            choice -= prob
            if choice <= 0:
                self.Action = choiceIndex
                return choiceIndex
            choiceIndex += 1
    
    def distr(self):
        expH = np.exp(self.H)
        self.probabilityDistribution = expH / expH.sum()
        return self.probabilityDistribution
    
    def update_reward(self,reward):
        H_Act_old = self.H[self.Action]
        self.H = self.H - self.alpha * (reward-self.R_bar) * self.probabilityDistribution
        self.H[self.Action] = H_Act_old + self.alpha * (reward-self.R_bar) * (1-self.probabilityDistribution[self.Action])
        self.cnt += 1
        self.R_bar = (1-1./self.cnt)*self.R_bar + reward / self.cnt 
        return 
        
    
    
if __name__ == "__main__":
    import random
    numActions = 2
    numRounds = 100
    biases = [1.0 / k for k in range(2,4)]
    rewardVector = [[1 if random.random() < bias else 0 for bias in biases] for _ in range(numRounds)]
    rewards = lambda choice, t: rewardVector[t][choice]
    cumulativeReward = 0
    bandit = GradientBandits()
    for j in range(numRounds):
        bandit.distr()
        Action = bandit.draw()
        reward = rewardVector[j][Action]
        print('Action = ',Action)
        cumulativeReward += reward
        bandit.update_reward(reward)

    