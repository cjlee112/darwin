# -*- coding: utf-8 -*-

import random
import sys

from scipy import stats
from mendel import *
from entropy import *

Wh = RecessiveAllele("Wh", stats.norm(0,1))
Pu = DominantAllele("Pu", stats.norm(10,1))

def multiset(list_):
    """Returns a multiset (a dictionary) from the input iterable list_."""
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


class RoboMendel(object):
    
    def __init__(self):
        self.population = list()
        self.crossing_model = dict({ tuple(["purple", "purple"]): "purple"})
    
    def initialize_population(self):
        """Initialize a population of 10 purple plants and 10 hybrid plants, all phenotypically purple."""
        print "RoboMendel> Initializing population."
        plants = list()
        for i in range(0, 10):
            plants.append(PeaPlant(loci=[Pu, Pu]))
        for i in range(0, 10):
            plants.append(PeaPlant(loci=[Pu, Wh]))
        self.population = plants

    def determine_color(self, obs):
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

    def population_report(self):
        purple_pop = [x for x in self.population if self.determine_color(x) == "purple"]
        white_pop = [x for x in self.population if self.determine_color(x) == "white"]
        print "RoboMendel> Report> There are %d white plants and %d purple plants" % (len(white_pop), len(purple_pop))
        return dict({"white": len(white_pop), "purple": len(purple_pop)})

    def predict_new_population(self, n=20):
        population_colors = set(self.determine_color(x) for x in self.population)
        predicted_colors = set()
        for color_1 in population_colors:
            for color_2 in population_colors:
                predicted_colors.add(self.crossing_model[(color_1, color_2)] )
        predicted_colors = list(predicted_colors)
        print "RoboMendel> Predict> The next population will consist of individuals of the following colors: %s." % (" ".join(predicted_colors))
        return predicted_colors

    def iterate_population(self, n=20):
        print "RoboMendel> Breeding new population"
        new_pop = list()
        for j in range(0, n):
            plant_1 = random.choice(self.population)
            plant_2 = random.choice(self.population)
            new_pop.append(plant_1 * plant_2)
        self.population = new_pop


def main(argv):
    # Create RoboMendel!
    print "Activating RoboMendel" 
    robomendel = RoboMendel()
    # Obtain initial population of all phenotypically purple plants.
    robomendel.initialize_population()
    # Examine the population, RoboMendel!
    robomendel.population_report()

    prediction_correct = True
    
    # What would happen if we breed a new population?
    predicted_colors = robomendel.predict_new_population()
    # Breed a new population.
    robomendel.iterate_population()
    # Tell us about the new population.
    report = robomendel.population_report()
    # Was your prediction correct?
    colors = set([x for x in report.keys() if report[x] > 0])
    for color in colors:
        if color not in predicted_colors:
            prediction_correct = False
    if prediction_correct:
        print "RoboMendel> Report> No anomolous colors found. Predicted colors were correct."
    else:
        print "RoboMendel> Report> Predicted colors incorrect."


    ##while True:

        ##print "Isolating and breeding purple plants"
        ### Initialize new population with purple plants
        ##population = generate_population(20, plants=purple_pop, gen=1)

        ### Separate the purple plants and the white plants
        ##purple_pop = [x for x in population if determine_color(x) == "purple"]
        ##white_pop = [x for x in population if determine_color(x) == "white"]

        ##print "New population:"
        ##print "  There are %d white plants and %d purple plants" % (len(white_pop), len(purple_pop))

        ##raw_input()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
