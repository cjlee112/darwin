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
    """Forms a multinomial from a probability dictionary, e.g. {'Pu':0.9, 'Wh': 0.1}"""

    def __init__(self, p_dict):
        self.p_dict = p_dict

    def rvs(self, num_obs):
        obs = []
        keys = self.p_dict.keys()
        values = [self.p_dict[key] for key in keys]
        for i in range(num_obs):
            multi = numpy.random.multinomial(1, values)
            for i in range(len(multi)):
                if multi[i] > 0:
                    obs.append(keys[i])
                    break
        return obs

    def pmf(self, obs):
        if hasattr(obs,"__iter__"):
            results = []
            for x in obs:
                try:
                    results.append(self.p_dict[x])
                except KeyError:
                    results.append(0)
            return results
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
        return self.genome.get_phenotypes()[0].rvs(n)

class SpeciesCrossModel(object):
    'species model for mating observations'
    def __init__(self, species, priors, pHybrid=0., pFail=0.001):
        '''species: a list of species models (must have pdf() method)
        priors: list of prior probabilities for the corresponding species
        pHybrid: probability that two different species will yield progeny
        pFail: probability that a mating between male & female of same
        species will produce no progeny.'''
        self.species = species
        self.priors = priors
        self.pHybrid = pHybrid
        self.pFail = pFail

    def p_obs(self, parent1, parent2, progeny):
        '''compute mating-success likelihood for parent1 obs to
        be emitted by one species and parent2 obs to be emitted by
        another, and progeny (True/False) based on whether they
        are the same species.
        Assumes p(progeny) only depends on whether the two species
        are the same (i.e. a constant for all non-matching pairs),
        and also that parent1 & parent2 are otherwise independent.

        Returns p(parent1), p(parent2|parent1), p(progeny|parent1, parent2)'''
        pMatch = 0.
        for i,sp in enumerate(self.species):
            p1 = priors[i] * sp.pdf(parent1)
            p2 = priors[i] * sp.pdf(parent2)
            pMatch += p1 * p2 # compute diagonal
            pSum1 += p1
            pSum2 += p2
        pMismatch = pSum1 * pSum2 - pMatch # subtract diagonal
        if progeny: # fixed probability for inter-species progeny
            pMismatch *= self.pHybrid
            pMatch *= (1. - self.pFail)
        else: # no progeny
            pMismatch *= (1. - self.pHybrid)
            pMatch *= self.pFail
        return pSum1, pSum2, (pMatch + pMismatch) / (pSum1 * pSum2)
