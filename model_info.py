# -*- coding: utf-8 -*-

import math
import sys

from scipy import stats
import pylab

from darwin.robomendel import *
from darwin.entropy import *
import darwin.mixture
from darwin.mixture import Mixture

def plot_im_ip_bernoulli():
    n = 100
    simulation_model = stats.bernoulli(0.25)
    outcomes = simulation_model.rvs(n)

    ## log of uninformitive prior
    prior = stats.bernoulli(0.5)
    lup = discrete_sample_Le(outcomes, prior)

    steps = 100
    im_points = []
    ip_points = []
    sum_points = []
    for i in range(1, steps):
        ## likelihood of observations from model
        mean = float(i) / steps
        model = stats.bernoulli(mean)
        l_e = discrete_sample_Le(outcomes, model)
        i_m = l_e - lup
        im_points.append((mean, i_m.mean))

        h_e = box_entropy(outcomes, 7)
        i_p = -l_e - h_e
        ip_points.append((mean, i_p.mean))
        
        print mean, i_m.mean, i_p.mean

        sum_points.append((mean, i_p.mean + i_m.mean))

    pylab.plot([x for (x,y) in im_points], [y for (x,y) in im_points])
    pylab.plot([x for (x,y) in ip_points], [y for (x,y) in ip_points])
    pylab.plot([x for (x,y) in sum_points], [y for (x,y) in sum_points])
    pylab.xlabel('success of bernoulli')
    pylab.ylabel('Im (blue), Ip (green), Im + Ip (red)')
    pylab.grid(True)
    pylab.show()


def plot_im_ip_normal():
    n = 8
    simulation_model = stats.norm(10, 1)
    outcomes = simulation_model.rvs(n)

    ## log of uninformitive prior
    prior = stats.uniform(0, 20)
    lup = sample_Le(outcomes, prior)

    steps = 100
    im_points = []
    ip_points = []
    sum_points = []
    for i in range(100):
        ## likelihood of observations from model
        mean = 8. + 4. * i / steps
        model = stats.norm(mean, 1)
        l_e = sample_Le(outcomes, model)
        i_m = l_e - lup
        im_points.append((mean, i_m.mean))

        h_e = box_entropy(outcomes, 7)
        i_p = -l_e - h_e
        ip_points.append((mean, i_p.mean))

        sum_points.append((mean, i_p.mean + i_m.mean))

    pylab.plot([x for (x,y) in im_points], [y for (x,y) in im_points])
    pylab.plot([x for (x,y) in ip_points], [y for (x,y) in ip_points])
    pylab.plot([x for (x,y) in sum_points], [y for (x,y) in sum_points])
    pylab.xlabel('mean of normal')
    pylab.ylabel('Im, Ip')
    pylab.grid(True)
    pylab.show()

def plot_im_asym_normal():
    im_points = []
    simulation_model = stats.norm(0, 1)

    m = 300
    n = 300

    obs_list = list(simulation_model.rvs(n))
    sample = simulation_model.rvs(m)

    for i in range(3, n):
        obs = numpy.core.array(obs_list[0:i])
        mean = numpy.average(obs)
        var = numpy.average(obs * obs) - mean * mean
        model_obs = stats.norm(mean, math.sqrt(var))
        l_e = box_entropy(obs, min(len(obs)-1, 7), sample=sample)
        log_prior = sample_Le(sample, model_obs)
        i_m = -l_e - log_prior
        im_points.append((i, i_m.mean))

    pylab.plot([x for (x,y) in im_points], [y for (x,y) in im_points])
    pylab.xlabel('sample_size')
    pylab.ylabel('Im')
    pylab.grid(True)
    pylab.show()


def plot_im_asym_bernoulli():
    im_points = []
    simulation_model = stats.bernoulli(0.25)

    m = 300
    n = 100

    obs_list = list(simulation_model.rvs(n))
    sample = simulation_model.rvs(m)

    for i in range(3, n):
        obs = obs_list[0:i]
        count = obs.count(1)
        p = float(count) / i
        model_obs = stats.bernoulli(p)
        
        l_e = box_entropy(obs, min(len(obs)-1, 7), sample=sample)
        log_prior = discrete_sample_Le(sample, model_obs)
        i_m = -l_e - log_prior
        im_points.append((i, i_m.mean))
        print p, i_m.mean


    pylab.plot([x for (x,y) in im_points], [y for (x,y) in im_points])
    pylab.xlabel('sample_size')
    pylab.ylabel('Im')
    pylab.grid(True)
    pylab.show()

def compute_ip_discrete(outcomes, model):
    h_e = He_discrete(outcomes)
    l_e = discrete_sample_Le(outcomes, model)
    i_p = -h_e - l_e
    return i_p

def compute_ip_continuous(outcomes, model):
    h_e = box_entropy(outcomes, 7)
    l_e = sample_Le(outcomes, model)
    i_p = -h_e - l_e
    return i_p

def compute_im_discrete(outcomes, model, prior):
    l_e = discrete_sample_Le(outcomes, model)
    log_prior = discrete_sample_Le(outcomes, prior)
    #log_prior = LogPVector(prior)
    i_m = l_e - log_prior
    return i_m

def compute_im_continuous(outcomes, model, prior):
    l_e = sample_Le(outcomes, model)
    #log_prior = LogPVector(prior)
    log_prior = sample_Le(outcomes, prior)
    i_m = l_e - log_prior
    return i_m

def progeny_model(d, cross=None):
    # Possibly unsure if same species.
    if cross in [('Pu', 'Wh'), ('Wh', 'Pu')]:
        return Multinomial({'y': 1-d, 'n': d})
    # Definitely the same species.
    if cross in [('Wh', 'Wh'), ('Pu', 'Pu')]:
        return Multinomial({'y': 1, 'n': 0})
    return None

def color_model(d, e, w, cross):
    modelPu = stats.norm(10, 1)
    modelWh = stats.norm(0, 1)
    # Same species
    if cross in [('Pu', 'Wh'), ('Wh', 'Pu'), ('Pu', 'Pu')]:
        modelMix = darwin.mixture.Mixture(((e*w, modelWh), (1-e*w, modelPu)))
    if cross in [('Wh', 'Wh')]:
        modelMix = darwin.mixture.Mixture(((d + (1-d)*(e*w), modelWh), ((1-d)*(1-e*w), modelPu)))
    return modelMix

def robomendel_wh_pu_crosses(n, d, e, w, outcomes, cross):
    # Progeny experiment, compute im ip ie
    prior = numpy.array([math.log(0.5)]*n)
    prior = Multinomial({'y': 0.5, 'n': 0.5})
    model = progeny_model(d, cross)

    offspring_obs = []
    for off in outcomes:
        if off is not None:
            offspring_obs.append('y')
        else:
            offspring_obs.append('n')

    i_m = compute_im_discrete(offspring_obs, model, prior)
    i_p = compute_ip_discrete(offspring_obs, model)

    print "  Progeny observation"
    print "  Im: %s, Ip: %s" % (i_m.mean, i_p.mean)

    offspring = [x for x in outcomes if x is not None]
    if len(offspring) != n:
        print "%s offspring are dead" % (n - len(offspring),)

    if not offspring:
        print "  All progeny dead, no color observations"
        return

    color_obs = [p.rvs()[0] for p in offspring]
    # Color experiment, compute im ip ie
    # prior is uniform over color detector range
    model = color_model(d, e, w, cross)
    prior = stats.uniform(-10, 30) # [-10, 20]
    i_m = compute_im_continuous(color_obs, model, prior)
    i_p = compute_ip_continuous(color_obs, model)

    print "  Color observation"
    print "  Im: %s, Ip: %s" % (i_m.mean, i_p.mean)


def main():
    #""" Experimental parameters
    #n == sample size
    #d == probability that Wh is a different species
    #e == probability that white color is an environmental effect
    #w == probability of white flowers from environmental effect """

    white_plant = PeaPlant(genome=PeaPlant.white_genome)
    purple_plant = PeaPlant(genome=PeaPlant.purple_genome)
    lavender_plant = PeaPlant(genome=PeaPlant.lavender_genome)

    ## Is Wh a different species?

    #n = 50
    #(d, e, w) = (0.2, 0.8, 0.1)
    #print "Experiment parameters"
    #print "d: %s, e: %s, w: %s" % (d,e,w)
    #print ""

    #test_outcomes = [ [None]*n, [white_plant]*n, [purple_plant]*n ]
    #offspring = [purple_plant]*(int((1-w)*n))
    #offspring.extend([white_plant]*(int(w*n)))
    #test_outcomes.append(offspring)
    #test_outcomes.append([lavender_plant]*n)

    #offspring = [purple_plant]*(int(n / 2))
    #offspring.extend([white_plant]*(int(n / 2)))
    #test_outcomes.append(offspring)

    #for outcomes in test_outcomes:
        #print "Test Outcomes", multiset(outcomes)
        #print "Wh x Wh"
        #robomendel_wh_pu_crosses(n, d, e, w, outcomes, cross=('Pu', 'Wh'))
        #print "Wh x Pu"
        #robomendel_wh_pu_crosses(n, d, e, w, outcomes, cross=('Wh', 'Wh'))
        #print "Pu x Pu"
        #robomendel_wh_pu_crosses(n, d, e, w, outcomes, cross=('Pu', 'Pu'))
        #print ""

    # Hybrid Cross experiment
    # h == probability that Wh and Pu produce a sterile hybrid
    n = 50
    h = 0.1
    hybrid_plant = white_plant * purple_plant
    test_outcomes = [hybrid_plant * hybrid_plant for i in range(n)]
    model = Multinomial({'y': h, 'n': 1-h})
    prior = numpy.array([math.log(0.5)]*n)

    prior = Multinomial({'y': 0.5, 'n': 0.5})

    offspring_obs = []
    for off in test_outcomes:
        if (off * off) is not None:
            offspring_obs.append('y')
        else:
            offspring_obs.append('n')

    i_m = compute_im_discrete(offspring_obs, model, prior)
    i_p = compute_ip_discrete(offspring_obs, model)

    print "Hybrid progeny experiment"
    print "  probability of fertile hybrid:", h
    print "  -- Hybrid progeny observation --"
    print "  Im: %s, Ip: %s" % (i_m.mean, i_p.mean)

    #offspring = [x for x in outcomes if x is not None]
    #if not offspring:
        #print "  All progeny dead, no color observations"
        #return
    #if len(offspring) != n:
        #print "%s offspring are dead" % (n - len(offspring),)
    
    # Now we've got fertile hybrids, so let's perform all the possible crosses and record the color counts
    n = 200
    plant_map = {'Wh': white_plant, 'Pu': purple_plant, 'Hy': hybrid_plant }
    test_crosses = [('Pu', 'Hy'), ('Wh', 'Hy'), ('Hy', 'Hy')]
    print "  -- Hybrid Cross Colors --"
    for (a, b) in test_crosses:
        plant_1 = plant_map[a]
        plant_2 = plant_map[b]
        offspring = [plant_1 * plant_2 for i in range(n)]
        print '  Cross:', a, ' x ', b, multiset(map(determine_color, offspring))

    # Hypothesize model for cross table color
    modelPu = stats.norm(10, 1)
    modelWh = stats.norm(0, 1)
    
    ## Idealized model
    #table_models = {('Pu', 'Pu'): modelPu, ('Pu', 'Wh'): modelPu, ('Pu', 'Hy'): modelPu, ('Hy', 'Hy'): Mixture(((0.25, modelWh), (0.75, modelPu))), ('Hy', 'Wh'): Mixture(((0.5, modelWh), (0.5, modelPu)))}
    
    ## Trained model
    table_models = dict()
    cross_pairs = [('Pu', 'Hy'), ('Hy', 'Hy'), ('Pu', 'Pu'), ('Hy', 'Wh'), ('Pu', 'Wh'), ('Wh', 'Wh')]
    n = 200
    models_map = {'purple': modelPu, 'white': modelWh}
    print "  -- Training cross color models --"
    for (a, b) in cross_pairs:
        plant_1 = plant_map[a]
        plant_2 = plant_map[b]
        offspring = [plant_1 * plant_2 for i in range(n)]
        mset = multiset(map(determine_color, offspring))
        mixture_tuples = []
        for key in mset.keys():
            mixture_tuples.append((float(mset[key]) / n, models_map[key]))
        print '  Cross:', a, ' x ', b, mset
        table_models[(a,b)] = Mixture(tuple(mixture_tuples))
    
    print "  -- Table Color Crosses --"
    for k,v in table_models.items():
        (a, b) = k
        model = v
        plant_1 = plant_map[a]
        plant_2 = plant_map[b]
        offspring = [plant_1 * plant_2 for i in range(n)]
        print '  Cross:', a, ' x ', b, multiset(map(determine_color, offspring))
        color_obs = [p.rvs()[0] for p in offspring]
        prior = stats.uniform(-10, 30) # [-10, 20]
        i_m = compute_im_continuous(color_obs, model, prior)
        i_p = compute_ip_continuous(color_obs, model)
        print "    Color observation"
        print "    Im: %s, Ip: %s" % (i_m.mean, i_p.mean)

    # Hypothesize model for table, identifying plants as Wh, Pu, or Hy via self-fertilization
    types = ['Pu', 'Hy', 'Wh']
    self_cross_models = dict()
    for t in types:
        self_cross_models[t] = table_models[(t,t)]

    def determine_type(plant, self_cross_models=self_cross_models, n=50):
        """For plant, self-fertilize and determine which model for 'Wh', 'Hy', or 'Pu' has the largest model information."""
        offspring = [plant * plant for i in range(n)]
        color_obs = [p.rvs()[0] for p in offspring]
        types = ['Pu', 'Hy', 'Wh']
        im_list = []
        prior = stats.uniform(-10, 30)
        for t in types:
            model = self_cross_models[t]
            i_m = compute_im_continuous(color_obs, model, prior)
            im_list.append((i_m.mean, t))
        im_list.sort()
        return im_list[-1][1]

    ## Idealized model
    #table_models = {('Pu', 'Pu'): Multinomial( {'Pu': 1, 'Wh': 0, 'Hy': 0}), ('Pu', 'Wh'): Multinomial( {'Pu': 0, 'Wh': 0, 'Hy': 1}), ('Pu', 'Hy'): Multinomial( {'Pu': 0.5, 'Wh': 0, 'Hy': 0.5}), ('Hy', 'Hy'): Multinomial( {'Pu': 0.25, 'Wh': 0.25, 'Hy': 0.5}), ('Hy', 'Wh'): Multinomial( {'Pu': 0, 'Wh': 0.5, 'Hy': 0.5})}

    print "  -- Training cross type models --"
    table_models = dict()
    for (a, b) in cross_pairs:
        plant_1 = plant_map[a]
        plant_2 = plant_map[b]
        offspring = [plant_1 * plant_2 for i in range(n)]
        mset = multiset(map(determine_type, offspring))
        multi_dict = dict()
        for key in mset.keys():
            multi_dict[key] = float(mset[key]) / n
        table_models[(a,b)] = Multinomial(multi_dict)
        print '  Cross:', a, ' x ', b, mset
    
    print "  -- Table Type Crosses --"
    for k,v in table_models.items():
        (a, b) = k
        model = v
        plant_1 = plant_map[a]
        plant_2 = plant_map[b]
        offspring = [plant_1 * plant_2 for i in range(n)]
        offspring_obs = map(determine_type, offspring)
        mset = multiset(offspring_obs)
        prior_dict = dict()
        # Uniformative prior on observed types
        #for key in mset.keys():
            #prior_dict[key] = 1. / len(mset.keys())
        #prior = Multinomial(prior_dict)
        # Uninformative prior on all types, could use training data above
        prior = Multinomial( {'Pu': 0.333, 'Wh': 0.3333, 'Hy': 0.3333})
        i_m = compute_im_discrete(offspring_obs, model, prior)
        i_p = compute_ip_discrete(offspring_obs, model)
        print '  Cross:', a, ' x ', b
        print "    Type observations", mset 
        print "    Im: %s, Ip: %s" % (i_m.mean, i_p.mean)


if __name__ == '__main__':
    sys,exit(main())


