# -*- coding: utf-8 -*-

from scipy import stats
from mendel import *
from entropy import *
import numpy

def multiset(list_):
    """Returns a multiset (a dictionary) from the input iterable list_."""
    mset = dict()
    for elem in list_:
        try:
            mset[elem] += 1
        except KeyError:
            mset[elem] = 1
    return mset

Wh = RecessiveAllele("Wh", stats.norm(0,1))
Pu = DominantAllele("Pu", stats.norm(10,1))

chrWh = Chromosome([(0.5,Wh)])
chrPu = Chromosome([(0.5,Pu)])

plantWh = DiploidGenome({1:(chrWh,chrWh)})
plantPu = DiploidGenome({1:(chrPu,chrPu)})

def determine_color(plant):
    obs = plant.get_phenotypes()[0].rvs(20)
    mean = numpy.average(obs)
    if mean > 5:
        return 'purple'
    return 'white'
