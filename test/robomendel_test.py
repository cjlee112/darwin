# -*- coding: utf-8 -*-
from darwin import mendel
from darwin import model
from darwin import mixture
from darwin import entropy
from darwin import robomendel
from scipy import stats
import numpy
import math
import random
from darwin.robomendel import PeaPlant

purple_plant = PeaPlant(genome=PeaPlant.purple_genome)
white_plant = PeaPlant(genome=PeaPlant.white_genome)
hybrid_plant = purple_plant * white_plant



def get_mix_model(modelWh, modelPu):
    return mixture.Mixture(((0.9, modelPu), (0.1, modelWh)))

def pheno1_setup(modelWh, modelPu):
    pstate = model.LinearState('Pu', modelPu)
    wstate = model.LinearState('Wh', modelWh)
    prior = model.StateGraph({'START':{pstate:0.9, wstate:0.1}})
    stop = model.StopState(useObsLabel=False)
    term = model.StateGraph({pstate:{stop:1.}, wstate:{stop:1.}})
    branches = model.BranchGenerator('chi', prior, iterTag='plantID')
    dg = model.DependencyGraph({'START':{branches:{}}, 'chi':{'STOP':term}})

    obsSet = model.ObsSet('plants')
    for plant in range(2): # two white plants
        obsSet.add_obs(modelWh.rvs(100), plantID=plant)
    for plant in range(2, 20): # 18 purple plants
        obsSet.add_obs(modelPu.rvs(100), plantID=plant)

    m = model.Model(dg, obsSet)
    return m, obsSet

def pheno1_test(modelWh, modelPu):
    m, obsSet = pheno1_setup(modelWh, modelPu)
    logPobs = m.calc_fb()
    print 'logPobs:', logPobs, m.segmentGraph.p_forward(m.logPobsDict)
    llDict = m.posterior_ll()

    mixModel = get_mix_model(modelWh, modelPu)

    for plant in range(20):
        obsLabel = obsSet.get_subset(plantID=plant)
        Le = entropy.SampleEstimator(numpy.array(llDict[obsLabel]))
        LeMix = entropy.sample_Le(obsLabel.get_obs(), mixModel)
        Ie = Le - LeMix
        He = entropy.box_entropy(obsLabel.get_obs(), 7)
        Ip = -Le - He
        print 'plant %d, Ie > %1.3f, mean = %1.3f\tIp > %1.3f, mean = %1.3f' \
              % (plant, Ie.get_bound(), Ie.mean, Ip.get_bound(), Ip.mean)
        
    return llDict


def get_mating_obs(species1, species2, progeny):
    return (species1.rvs(1)[0], species2.rvs(1)[0], progeny)


def mating_test(species, priors=None, **kwargs):
    'generate 2 x 2 test of all possible mating combinations'
    if not priors:
        priors = (1./len(species),) * len(species)
    scm = robomendel.SpeciesCrossModel(species, priors, **kwargs)
    mstate = model.LinearState('mating', scm)
    prior = model.StateGraph({'START':{mstate:1}})
    branches = model.BranchGenerator('chi', prior, iterTag='matingID')
    stop = model.StopState(useObsLabel=False)
    term = model.StateGraph({mstate:{stop:1.}})
    dg = model.DependencyGraph({'START':{branches:{}}, 'chi':{'STOP':term}})

    obsSet = model.ObsSet('mating obs')
    obsSet.add_obs(species[0].rvs(3), matingID=0)
    obsSet.add_obs((species[0].rvs(1)[0], species[1].rvs(1)[0], None),
                   matingID=1)
    obsSet.add_obs((species[0].rvs(1)[0], species[0].rvs(1)[0], None),
                   matingID=2)
    obsSet.add_obs((species[0].rvs(1)[0], species[1].rvs(1)[0],
                    species[0].rvs(1)[0]), matingID=3)
    
    m = model.Model(dg, obsSet)
    logPobs = m.calc_fb()
    llDict = m.posterior_ll()

    for matingID,t in enumerate(((0,0,0), (0,1,None),
                                 (0,0,None), (0,1,0))):
        obsLabel = obsSet.get_subset(matingID=matingID)
        print 'mating %s:\tlogP = %1.3f, %1.3f, %1.3f' % \
              tuple([str(t)] + llDict[obsLabel])
        
def multicond_setup(modelWh, modelPu, pHybrid=0.):
    pstate = model.VarFilterState('Pu', modelPu)
    wstate = model.VarFilterState('Wh', modelWh)
    prior = model.StateGraph({'START':{pstate:0.9, wstate:0.1}})
    stop = model.StopState(useObsLabel=False)
    term = model.StateGraph({pstate:{stop:1.}, wstate:{stop:1.},
                             robomendel.noneState:{stop:1.}})

    sct = robomendel.SpeciesCrossTransition(pHybrid=pHybrid)
    return pstate, wstate, prior, stop, term, sct

def get_family_obs(mom=(0.,), dad=(1.,), child=(0.5,), **tags):
    obsSet = model.ObsSet('mating obs')
    obsSet.add_obs(mom, var='mom', **tags)
    obsSet.add_obs(dad, var='dad', **tags)
    obsSet.add_obs(child, var='child', **tags)
    return obsSet

def multicond_calc(modelWh, modelPu, obsSet):
    '''This test creates nodes representing mom, dad and the child,
    with a multi-cond edge from (mom,dad) --> child'''
    m = model.Model(family_model(modelWh, modelPu), obsSet)
    return m, m.segmentGraph.p_forward(m.logPobsDict)

def multicond_test():
    modelWh = stats.norm(0, 1)
    modelPu = stats.norm(10, 1)
    p = modelWh.pdf((0, 1, 0.5)).prod() * 0.01 * 0.999
    m, logP = multicond_calc(modelWh, modelPu, get_family_obs())
    print math.log(p), logP
    if abs(math.log(p) - logP) > math.log(1.02): # trap > 2% error
        raise ValueError('bad logP value: %1.3f vs %1.3f' %(logP, math.log(p)))

def multicond2_calc(modelWh, modelPu, obsSet):
    '''This test creates nodes representing mom, dad and the child,
    with a multi-cond edge from (mom,dad) --> child
    and tests two different matings simultaneously.'''
    pstate, wstate, prior, stop, term, sct = multicond_setup(modelWh, modelPu)
    moms = model.BranchGenerator('mom', prior, iterTag='matingID')
    dads = model.BranchGenerator('dad', prior, iterTag='matingID')
    dg = model.DependencyGraph({'START':{moms:{}, dads:{}},
                                ('mom', 'dad'):{'child':sct},
                                'child':{'STOP':term}},
                               joinTags=('matingID',))
    m = model.Model(dg, obsSet)
    return m, m.segmentGraph.p_forward(m.logPobsDict)

def multicond2_test():
    modelWh = stats.norm(0, 1)
    modelPu = stats.norm(10, 1)
    obsSet = get_family_obs(matingID=0)
    m1, logP1 = multicond_calc(modelWh, modelPu, obsSet)
    m2, logP2 = multicond2_calc(modelWh, modelPu, obsSet)
    print logP1, logP2
    if abs(logP1 - logP2) > math.log(1.02): # trap > 2% error
        raise ValueError('bad logP value: %1.3f vs %1.3f' %(logP1, logP2))

def multicond3_test():
    modelWh = stats.norm(0, 1)
    modelPu = stats.norm(10, 1)
    obsSet1 = get_family_obs()
    mom2, dad2, child2 = (0.2,), (10.3,), (-0.6,)
    obsSet2 = get_family_obs(mom=mom2, dad=dad2, child=child2)
    obsSetBoth = get_family_obs(matingID=0)
    obsSetBoth.add_obs(mom2, var='mom', matingID=1)
    obsSetBoth.add_obs(dad2, var='dad', matingID=1)
    obsSetBoth.add_obs(child2, var='child', matingID=1)
    m1, logP1 = multicond_calc(modelWh, modelPu, obsSet1)
    m2, logP2 = multicond_calc(modelWh, modelPu, obsSet2)
    mBoth, logPBoth = multicond2_calc(modelWh, modelPu, obsSetBoth)
    print logP1 + logP2, logPBoth
    if abs(logP1 + logP2 - logPBoth) > math.log(1.02): # trap > 2% error
        raise ValueError('bad logP value: %1.3f vs %1.3f' %(logP1 + logP2,
                                                            logPBoth))

def get_2family_obs(modelWh, modelPu):    
    obsSet = model.ObsSet('mating obs')
    obsSet.add_obs(modelWh.rvs(1),var='mom', matingID=0)
    obsSet.add_obs(modelWh.rvs(1),var='dad', matingID=0)
    obsSet.add_obs(modelWh.rvs(1),var='child', matingID=0)
    obsSet.add_obs(modelPu.rvs(1),var='mom', matingID=1)
    obsSet.add_obs(modelPu.rvs(1),var='dad', matingID=1)
    obsSet.add_obs(modelPu.rvs(1),var='child', matingID=1)
    return obsSet

def get_families_obs(n, models):
    'models must be list of tuples [(mom_model, dad_model, child_model),]'
    obsSet = model.ObsSet('mating obs')
    for i, t in enumerate(models):
        for j, var in enumerate(('mom', 'dad', 'child')):
            obsSet.add_obs(t[j].rvs(n), var=var, matingID=i)
    return obsSet


def family_model(modelWh, modelPu):
    'standard mom x dad --> child model'
    pstate, wstate, prior, stop, term, sct = multicond_setup(modelWh, modelPu)
    dg = model.DependencyGraph({'START':{'mom':prior, 'dad':prior},
                                ('mom', 'dad'):{'child':sct},
                                'child':{'STOP':term}})
    return dg

def family_model2(modelWh, modelPu, pHybrid=0.):
    'standard mom x dad --> child model, supports multiple matingID'
    pstate, wstate, prior, stop, term, sct = multicond_setup(modelWh, modelPu, pHybrid)
    moms = model.BranchGenerator('mom', prior, iterTag='matingID')
    dads = model.BranchGenerator('dad', prior, iterTag='matingID')
    dg = model.DependencyGraph({'START':{moms:{}, dads:{}},
                                ('mom', 'dad'):{'child':sct},
                                'child':{'STOP':term}},
                               joinTags=('matingID',))
    return dg

def unrelated_model(modelWh, modelPu):
    'model mom, dad, child as independent'
    pstate = model.VarFilterState('Pu', modelPu)
    wstate = model.VarFilterState('Wh', modelWh)
    prior = model.StateGraph({'START':{pstate:0.9, wstate:0.1}})
    stop = model.StopState(useObsLabel=False)
    term = model.StateGraph({pstate:{stop:1.}, wstate:{stop:1.},
                             robomendel.noneState:{stop:1.}})

    dg = model.DependencyGraph({'START':{'mom':prior, 'dad':prior,
                                         'child':prior},
                                'mom':{'STOP':term},
                                'dad':{'STOP':term},
                                'child':{'STOP':term}})
    return dg

def unrelated_model2(modelWh, modelPu):
    'model mom, dad, child as independent, supports multiple matingID'
    pstate = model.VarFilterState('Pu', modelPu)
    wstate = model.VarFilterState('Wh', modelWh)
    prior = model.StateGraph({'START':{pstate:0.9, wstate:0.1}})
    stop = model.StopState(useObsLabel=False)
    term = model.StateGraph({pstate:{stop:1.}, wstate:{stop:1.},
                             robomendel.noneState:{stop:1.}})

    moms = model.BranchGenerator('mom', prior, iterTag='matingID')
    dads = model.BranchGenerator('dad', prior, iterTag='matingID')
    kids = model.BranchGenerator('child', prior, iterTag='matingID')
    dg = model.DependencyGraph({'START':{moms:{}, dads:{}, kids:{}},
                                'mom':{'STOP':term},
                                'dad':{'STOP':term},
                                'child':{'STOP':term}},
                               joinTags=('matingID',))
    return dg

def environmental_model(modelWh, modelPu):
    'model wh / pu as random extrinsic variable'
    def filter_from_node(fromNode, *args):
        return dict(var=fromNode.var.label)
    pstate = model.VarFilterState('Pu', modelPu, filter_f=filter_from_node)
    wstate = model.VarFilterState('Wh', modelWh, filter_f=filter_from_node)
    peaSpecies = model.SilentState('pea')
    prior = model.StateGraph({'START':{peaSpecies:1.}})
    extSG = model.StateGraph({peaSpecies:{pstate:0.9, wstate:0.1}})
    stop = model.StopState(useObsLabel=False)
    term = model.StateGraph({pstate:{stop:1.}, wstate:{stop:1.}})
    sct = robomendel.SpeciesCrossTransition()

    dg = model.DependencyGraph({'START':{'mom':prior, 'dad':prior},
                                'mom':{'ext':extSG},
                                'dad':{'ext':extSG},
                                ('mom', 'dad'):{'child':sct},
                                'child':{'ext':extSG},
                                'ext':{'STOP':term}})
    return dg

def environmental_model2(modelWh, modelPu):
    'model wh / pu as random extrinsic variable, supports multiple matingID'
    def filter_from_node(fromNode, *args):
        return dict(var=fromNode.var.label)
    pstate = model.VarFilterState('Pu', modelPu, filter_f=filter_from_node)
    wstate = model.VarFilterState('Wh', modelWh, filter_f=filter_from_node)
    peaSpecies = model.SilentState('pea')
    prior = model.StateGraph({'START':{peaSpecies:1.}})
    noSpecies = model.SilentState('no-species')
    extNone = model.VarFilterState('ext-none', model.EmissionDict({None:1.}),
                                   filter_f=filter_from_node)
    extSG = model.StateGraph({peaSpecies:{pstate:0.9, wstate:0.1},
                              noSpecies:{extNone:1.}})
    stop = model.StopState(useObsLabel=False)
    term = model.StateGraph({pstate:{stop:1.}, wstate:{stop:1.}, extNone:{stop:1.}})
    sct = robomendel.SpeciesCrossTransition(noneState=noSpecies)

    moms = model.BranchGenerator('mom', prior, iterTag='matingID')
    dads = model.BranchGenerator('dad', prior, iterTag='matingID')
    dg = model.DependencyGraph({'START':{moms:{}, dads:{}},
                                'mom':{'ext':extSG},
                                'dad':{'ext':extSG},
                                ('mom', 'dad'):{'child':sct},
                                'child':{'ext':extSG},
                                'ext':{'STOP':term}},
                               joinTags=('matingID',))
    return dg



def mixture_model(modelWh, modelPu):
    'model process as single species, with mixture emission'
    mixModel = get_mix_model(modelWh, modelPu)
    peaSpecies = model.VarFilterState('pea', mixModel)
    prior = model.StateGraph({'START':{peaSpecies:1.0}})
    stop = model.StopState(useObsLabel=False)
    term = model.StateGraph({peaSpecies:{stop:1.}, 
                             robomendel.noneState:{stop:1.}})
    sct = robomendel.SpeciesCrossTransition()

    dg = model.DependencyGraph({'START':{'mom':prior, 'dad':prior},
                                ('mom', 'dad'):{'child':sct},
                                'child':{'STOP':term}})
    return dg


def mixture_model2(modelWh, modelPu):
    'model process as single species, with mixture emission, supports multiple matingID'
    mixModel = get_mix_model(modelWh, modelPu)
    peaSpecies = model.VarFilterState('pea', mixModel)
    prior = model.StateGraph({'START':{peaSpecies:1.0}})
    stop = model.StopState(useObsLabel=False)
    term = model.StateGraph({peaSpecies:{stop:1.}, 
                             robomendel.noneState:{stop:1.}})
    sct = robomendel.SpeciesCrossTransition()

    moms = model.BranchGenerator('mom', prior, iterTag='matingID')
    dads = model.BranchGenerator('dad', prior, iterTag='matingID')
    dg = model.DependencyGraph({'START':{moms:{}, dads:{}},
                                ('mom', 'dad'):{'child':sct},
                                'child':{'STOP':term}},
                               joinTags=('matingID',))
    return dg


def basic_pl(segmentGraph):
    f = segmentGraph.fprob[segmentGraph.start].f # forward prob dictionary
    return model.posterior_ll(f)

def print_pl(llDict):
    for obsLabel, ll in llDict.items():
        print '%s\t%s' % (str(obsLabel), ','.join([('%1.2f' % x) for x in ll]))


def pl_for_model(model_f, obsSet):
    modelWh = stats.norm(0, 1)
    modelPu = stats.norm(10, 1)
    dg = model_f(modelWh, modelPu)
    m = model.Model(dg, obsSet)
    logP = m.segmentGraph.p_forward(m.logPobsDict)
    llDict = basic_pl(m.segmentGraph)
    return llDict, logP

def pl1_test(model_f=family_model, obsSet=None):
    if obsSet is None:
        obsSet = get_family_obs()
    llDict, logP = pl_for_model(model_f, obsSet)
    print_pl(llDict)
    return logP
    
    
def merge_forward_dict(f, logP, result=None):
    'merge f into result, adjusted by a fixed prior'
    if result is None:
        result = {}
    for k,v in f.items():
        result[k] = v + logP
    return result


def subgraph_pl_test(modelDict=dict(mix=mixture_model2,
                                    family=family_model2,
                                    unrelated=unrelated_model2,
                                    environmental=environmental_model2),
                     obsSet=None):
    'build multi-model hypergraph, apply to the obs, and print posterior liklihoods'
    p = 1./len(modelDict) # uninformative prior
    modelWh = stats.norm(0, 1)
    modelPu = stats.norm(10, 1)
    if obsSet is None:
        obsSet = get_family_obs(matingID=0)
    stop = model.StopState(useObsLabel=False)
    d = {}
    d2 = {}
    for model_name, model_f in modelDict.items(): # build distinct models
        state = model.SilentState(model_name)
        state.subgraph = model_f(modelWh, modelPu)
        d[state] = p
        d2[state] = {stop:1.}
    prior = model.StateGraph({'START':d})
    term = model.StateGraph(d2)
    dg = model.DependencyGraph({'START':{'model':prior},
                                'model':{'STOP':term}})
    m = model.Model(dg, obsSet)
    m.segmentGraph.p_forward(m.logPobsDict)
    f = m.segmentGraph.fprob[m.start].f
    fmerge = {}
    for node, logP in f.items(): # merge forward calcs from subgraphs
        try:
            subgraph = node.segmentGraph
        except AttributeError:
            pass
        else:
            merge_forward_dict(subgraph.fprob[subgraph.start].f, logP, fmerge)
    llDict = model.posterior_ll(fmerge)
    print_pl(llDict)
    return llDict

   
def robomendel_cross_obs(n1=10, n2=10, n=1):
    'construct n1 Pu x Pu crosses, and n2 Hy x Hy cross; n obs per plant'
    obsSet = model.ObsSet('mating obs')
    parents = [(purple_plant, purple_plant)] * n1 #, (purple_plant, white_plant)]
    parents.extend([(hybrid_plant, hybrid_plant)] * n2)
    for i in range(len(parents)):
        (parent_1, parent_2) = parents[i]
        child = parent_1 * parent_2
        obsSet.add_obs(parent_1.rvs(n), var='mom', matingID=i)
        obsSet.add_obs(parent_2.rvs(n), var='dad', matingID=i)
        obsSet.add_obs(child.rvs(n), var='child', matingID=i)
    #i = len(parents)
    #obsSet.add_obs(purple_plant.rvs(n), var='mom', matingID=i)
    #obsSet.add_obs(purple_plant.rvs(n), var='dad', matingID=i)
    #obsSet.add_obs(white_plant.rvs(n), var='child', matingID=i)
    return obsSet


def pair_test(models=(mixture_model2, family_model2, unrelated_model2,
                      environmental_model2)):
    obsSet = get_family_obs(matingID=0)
    obsSet2 = get_family_obs(matingID=1)
    obsSet += obsSet2
    for m in models:
        ## print 'MODEL', m
        logP = pl1_test(m, obsSet) # combined obs
        logP2 = pl1_test(m, obsSet2) # half the obs
        if abs(logP - 2. * logP2) > 0.02:
            raise AssertionError('bad logP: %1.3f vs %1.3f' %(logP, 2. * logP2))

def hybrid_test(pHybrid=0.01):
    'test Wh x Pu outcome on family_model2 with specified pHybrid value'
    modelWh = stats.norm(0, 1)
    modelPu = stats.norm(10, 1)
    obsSet = get_families_obs(1, ((modelWh, modelPu, modelPu),))
    def my_family_model(modelWh, modelPu):
        return family_model2(modelWh, modelPu, pHybrid)
    logP = pl1_test(my_family_model, obsSet)
    obs = obsSet.get_subset(matingID=0).get_obs()
    assert abs(logP - math.log(.1 * .9 * pHybrid * modelWh.pdf(obs[0])
                               * modelPu.pdf(obs[1]) * modelPu.pdf(obs[2]))) <= 0.02
        
def multi_hybrid_test(hybridVals=(0.1, 0.01, 0.001)):
    'test multiple pHybrid values to ensure PL has correct sensitivity to pHybrid'
    for pHybrid in hybridVals:
        hybrid_test(pHybrid)


def random_cross(n=100, nobs=1):
    obsSet = model.ObsSet('mating obs')
    c = (purple_plant, white_plant)
    for i in range(n):
        mom = random.choice(c)
        dad = random.choice(c)
        child = mom * dad
        obsSet.add_obs(mom.rvs(nobs), var='mom', matingID=i)
        obsSet.add_obs(dad.rvs(nobs), var='dad', matingID=i)
        obsSet.add_obs(child.rvs(nobs), var='child', matingID=i)
    return obsSet


def calc_entropy(data, m):
    vec = entropy.sphere_entropy(data, m)
    return numpy.average(vec.sample)


def mutual_info_test(n=100, nobs=1, m=7):
    obsSet = random_cross(n, nobs)
    all = [obsSet.get_subset(matingID=i).get_obs() for i in range(n)]
    Hall = calc_entropy(all, m)
    moms = [obsSet.get_subset(matingID=i, var='mom').get_obs() for i in range(n)]
    Hmoms = calc_entropy(moms, m)
    dads = [obsSet.get_subset(matingID=i, var='dad').get_obs() for i in range(n)]
    Hdads = calc_entropy(dads, m)
    kids = [obsSet.get_subset(matingID=i, var='child').get_obs() for i in range(n)]
    Hkids = calc_entropy(kids, m)
    parents = [tuple(moms[i])+tuple(dads[i]) for i in range(n)]
    Hparents = calc_entropy(parents, m)
    hcond2 = Hall - Hparents
    print Hmoms, Hdads, Hkids, Hparents, Hall, hcond2 
    print 'Hkids, hcond2:', Hkids, hcond2
    print 'I(child; mom, dad) = %1.3f' % (Hkids - hcond2)
    print 'I(mom; dad) = %1.3f' % (Hmoms + Hdads - Hparents)


