# -*- coding: utf-8 -*-

import random

from scipy import stats
from mendel import *
from entropy import *
import numpy

def factorial(n):
    result = 1
    if n > 1:
        for i in range(2, n+1):
            result *= i
    return result

def multi_coef(n_seq):
    n = numpy.sum(n_seq)
    result = float(factorial(n))
    for i in range(len(n_seq)):
        result = result / float(factorial(n_seq[i]))
    return result

# http://en.wikipedia.org/wiki/Multinomial_distribution 
class Multinomial(stats.rv_discrete):

    def __init__(self, p_seq):
        self.p_seq = p_seq

    def rvs(self, num_obs):
        obs = []
        for i in range(num_obs):
            multi = numpy.random.multinomial(1, self.p_seq)
            for i in range(len(multi)):
                if multi[i] > 0:
                    obs.append(i)
                    break
        return obs

    def pmf(self, obs):
        if hasattr(obs,"__iter__"):
            return map(lambda x: self.p_seq[x], obs)
        return self.p_seq[obs]


def multiset(list_):
    """Returns a multiset (a dictionary) from the input iterable list_."""
    mset = dict()
    for elem in list_:
        try:
            mset[elem] += 1
        except KeyError:
            mset[elem] = 1
    return mset

def determine_color(plant):
    obs = plant.rvs(20)
    mean = numpy.average(obs)
    if mean > 5:
        return 'purple'
    return 'white'

# to construct a homozygous purple plant, use
# PeaPlant(genome=PeaPlant.purple_genome)

class PeaPlant(object):
    white_allele = RecessiveAllele("wh", stats.norm(0,1))
    purple_allele = DominantAllele("pu", stats.norm(10,1))
    white_chromosome = Chromosome([(0.5, white_allele)])
    purple_chromosome = Chromosome([(0.5, purple_allele)])
    white_genome = DiploidGenome({1:(white_chromosome, white_chromosome)})
    purple_genome = DiploidGenome({1:(purple_chromosome, purple_chromosome)})

    def __init__(self, name=None, genome=None):
        if genome is None:
            genome = random.choice([self.white_genome, self.purple_genome])
        else:
            self.genome = genome
        if name is not None:
            self.name = name
        else:
            self.name = ""

    def __mul__(self, mate):
        """Reproduce!"""
        new_genome = self.genome * mate.genome
        return PeaPlant(genome=new_genome)

    def rvs(self, n=1):
        """Observe n times."""
        return genome.get_phenotypes()[0].rvs(n)

