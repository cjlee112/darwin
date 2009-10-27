# -*- coding: utf-8 -*-

import random
import sys

from scipy import stats
from mendel import *
from entropy import *

Wh = RecessiveAllele("Wh", stats.norm(0,1))
Pu = DominantAllele("Pu", stats.norm(10,1))

def multiset(list_):
    """return a multiset from the input iterable list_"""
    mset = dict()
    for elem in list_:
        try:
            mset[elem] += 1
        except KeyError:
            mset[elem] = 1
    return mset

class PeaPlant(object):
    def __init__(self, name=None, loci=None):
        if loci is None:
            genes = {"white": Wh, "purple": Pu }
            self.loci = list()
            for i in range(0,2):
                key = random.choice(genes.keys())
                self.loci.append(genes[key])
        else:
            self.loci = loci
        if name is not None:
            self.name = name
        else:
            self.name = ""

    def __mul__(self, mate):
        """Reproduce!"""
        return PeaPlant(loci=[random.choice(self.loci), random.choice(mate.loci)] )
        
    def rvs(self, n=1):
        """Observe n times."""
        locus = self.loci[0] + self.loci[1]
        return locus.rvs(n)

def experiment(parent_1, parent_2, n=1):
    """Breed parents and observe n offspring."""
    observations = list()
    for i in range(0, n):
        offspring = parent_1 * parent_2
        observations.append(offspring.rvs(1)[0])
    return observations

def generate_population(n=100, gen=20, plants=None):
    """Generate and breed several generations of plants, attempting to select pure white and pure purple plants to mimic Mendel's initial conditions."""
    #initialize population
    if plants is None:
        plants = list()
        for i in range(0,n):
            plants.append(PeaPlant())
    for i in range(0, gen):
        new_pop = list()
        for j in range(0,n):
            plant_1 = random.choice(plants)
            plant_2 = random.choice(plants)
            new_pop.append(plant_1 * plant_2)
        plants = new_pop
    return plants

def determine_color(obs):
    """Given observations from the same plant, declare the plant white or purple based on the mean of the observations."""
    if isinstance(obs, PeaPlant):
        obs = obs.rvs(1)
    mean_obs = float(sum(obs)) / float(len(obs))
    dist_to_zero = abs(mean_obs - 0)
    dist_to_ten = abs(mean_obs - 10)
    if dist_to_zero > dist_to_ten:
        return "purple"
    else:
        return "white"

def punnet_cross_experiments():
    # phenotypes for initial plant constructions
    loci_choices = {"white": Wh, "purple": Pu }
        
    pp = PeaPlant(name="purple", loci=[loci_choices['purple'], loci_choices['purple']])
    ww = PeaPlant(name="white",  loci=[loci_choices['white'], loci_choices['white']])
    pw = PeaPlant(name="hybrid", loci=[loci_choices['purple'], loci_choices['white']])
    plants = [pp,ww,pw]

    # All possible crosses
    num_obs = 5
    for plant_1 in plants:
        for plant_2 in plants:
            print "Experiment: Breeding %s and %s, observing %s offspring" % (plant_1.name, plant_2.name, num_obs)
            obs = experiment(plant_1, plant_2, num_obs)
            print "  observations", obs
            colors = list()
            for ob in obs:
                colors.append(determine_color([ob]))
            offspring_counts = multiset(colors)
            for key in offspring_counts:
                print "  %d %s offspring" % (offspring_counts[key], key)


def main(argv):
    population = generate_population()
    
    # Separate the purple plants and the white plants
    purple_pop = [x for x in population if determine_color(x) == "purple"]
    white_pop = [x for x in population if determine_color(x) == "white"]

    print "Initial population:"
    print "  There are %d white plants and %d purple plants" % (len(white_pop), len(purple_pop))
    
    
    print "Isolating and breeding purple plants"
    # Initialize new population with purple plants
    population = generate_population(plants=purple_pop, gen=1)
        
    # Separate the purple plants and the white plants
    purple_pop = [x for x in population if determine_color(x) == "purple"]
    white_pop = [x for x in population if determine_color(x) == "white"]

    print "New population:"
    print "  There are %d white plants and %d purple plants" % (len(white_pop), len(purple_pop))




if __name__ == '__main__':
    sys.exit(main(sys.argv))
