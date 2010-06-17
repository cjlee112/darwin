# -*- coding: utf-8 -*-

import random

from scipy import stats
from mendel import *
from entropy import *
import model
import numpy

# http://en.wikipedia.org/wiki/Multinomial_distribution 
# this is really more of a categorical distribution
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
        if hasattr(obs, "__iter__"):
            results = []
            for x in obs:
                try:
                    results.append(self.p_dict[x])
                except KeyError:
                    results.append(0)
            return results
        return self.p_seq[obs]

    def __repr__(self):
        return 'Multinomial(%s)' % self.p_dict

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
    if mean > 8:
        return 'purple'
    if mean > 4:
        return 'lavender'
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

    lavender_allele = DominantAllele("la", stats.norm(5,1))
    lavender_chromosome = Chromosome([(0.5, lavender_allele)])
    lavender_genome = DiploidGenome({1:(lavender_chromosome, lavender_chromosome)})

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

    def __repr__(self):
        return determine_color(self)


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
        pMatch = pSum1 = pSum2 = pSum3 = 0.
        for i,sp in enumerate(self.species):
            p1 = self.priors[i] * sp.pdf(parent1)
            p2 = self.priors[i] * sp.pdf(parent2)
            pSum1 += p1
            pSum2 += p2
            if progeny is None: # no progeny
                pMatch += p1 * p2 # compute diagonal
            else:
                p3 = self.priors[i] * sp.pdf(progeny)
                pMatch += p1 * p2 * p3 # compute diagonal
                pSum3 += p3
        if progeny is None: # no progeny
            pMismatch = pSum1 * pSum2 - pMatch # subtract diagonal
            pMismatch *= (1. - self.pHybrid)
            pMatch *= self.pFail
        else: # fixed probability for inter-species progeny
            pMismatch = pSum1 * pSum2 * pSum3 - pMatch # subtract diagonal
            pMismatch *= self.pHybrid
            pMatch *= (1. - self.pFail)
        return pSum1, pSum2, (pMatch + pMismatch) / (pSum1 * pSum2)

    def pdf(self, obs):
        '''provide scipy.stats style pdf() interface.
        Expects (parent1, parent2, progeny) observation vector.'''
        return self.p_obs(obs[0], obs[1], obs[2])

noneState = model.VarFilterState('no-progeny', model.EmissionDict({None:1.}))

        
class SpeciesCrossTransition(object):
    'state-graph for conditioning on both parents'
    def __init__(self, pHybrid=0., pFail=0.001, noneState=noneState):
        self.pHybrid = pHybrid
        self.pFail = pFail
        self.noneState = noneState

    def __call__(self, sources, targetVar, state=None, parent=None):
        mom, dad = sources # raise ValueError if wrong number of sources
        if targetVar.obsLabel is not None:
            obsLabel = targetVar.obsLabel
        else:
            obsLabel = mom.var.obsLabel
        noProgeny = self.noneState(mom, targetVar, obsLabel, 1., parent)
        childMom = mom.state(mom, targetVar, obsLabel, 1., parent)
        childDad = dad.state(dad, targetVar, obsLabel, 1., parent)
        if mom.state == dad.state: # species match
            return {childMom:1. - self.pFail, noProgeny:self.pFail}
        else:
            return {childMom:self.pHybrid, childDad:self.pHybrid,
                    noProgeny:1. - 2. * self.pHybrid}
