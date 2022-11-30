#!/usr/bin/env python
# coding: utf-8



#------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   A simple, bare bones, implementation of differential evolution with Python
#   August, 2017
#
#------------------------------------------------------------------------------+

#--- IMPORT DEPENDENCIES ------------------------------------------------------+

from random import random
from random import sample
from random import uniform
import numpy as np
from tqdm import tqdm
#--- FUNCTIONS ----------------------------------------------------------------+




#--- MAIN ---------------------------------------------------------------------+
class DE:
    def __init__(self, bounds, popsize, mutate, recombination, cost_func):
        self.bounds = bounds
        self.popsize = popsize
        self.mutate = mutate
        self.recombination = recombination
        self.cost_func = cost_func

    def minimize(self, population=None):
        gen_scores = [] # score keeping
        for j in range(0, self.popsize):
            candidates = list(range(0, self.popsize))
            candidates.remove(j)
            random_index = sample(candidates, 3)#随机采样
            x_1 = np.array(population[random_index[0]])
            x_2 = np.array(population[random_index[1]])
            x_3 = np.array(population[random_index[2]])
            x_t = np.array(population[j]).copy()  # target individual
            x_diff = np.array(x_2) - np.array(x_3)
            v_donor = np.clip(np.array(x_diff) * self.mutate + x_1, -self.bounds, self.bounds)
            idx = np.random.choice(np.arange(0, len(x_1)), size=int(len(x_1) * self.recombination), replace=False) 
            noidx = np.delete(np.arange(0, len(x_1)), idx)  # 不需要重组的下标
            v_trial = np.arange(0, len(x_1),dtype=float)
            v_trial[idx.tolist()] = v_donor[idx.tolist()]
            v_trial[noidx.tolist()] = x_t[noidx.tolist()]
            score,which_solution  = self.cost_func(v_trial,x_t)
            gen_scores.append(score)
            if which_solution==1:
                population[j] = v_trial.copy()
            else:
                population[j] = x_t.copy()
        gen_best = min(gen_scores)                                  # fitness of best individual
        gen_sol = population[gen_scores.index(min(gen_scores))]     # solution of best individual
        bestidx = gen_scores.index(min(gen_scores))
        return gen_sol, gen_best, population, bestidx
