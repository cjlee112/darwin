# -*- coding: utf-8 -*-

import random

from scipy import stats
from mendel import *
from entropy import *
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

