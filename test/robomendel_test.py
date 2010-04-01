from darwin import mendel
from darwin import model
from darwin import mixture
from darwin import entropy
from darwin import robomendel
from scipy import stats
import numpy

def get_mix_model(modelWh, modelPu):
    return mixture.Mixture(((0.9, modelPu), (0.1, modelWh)))

def pheno1_test(modelWh, modelPu):
    pstate = model.LinearState('Pu', modelPu)
    wstate = model.LinearState('Wh', modelWh)
    prior = model.StateGraph({'START':{pstate:0.9, wstate:0.1}})
    stop = model.StopState(useObsLabel=False)
    term = model.StateGraph({pstate:{stop:1.}, wstate:{stop:1.}})
    branches = model.BranchGenerator('chi', prior, iterTag='plantID')
    dg = model.DependencyGraph({'START':branches, 'chi':{'chi':term}})

    obsSet = model.ObsSet('plants')
    for plant in range(2): # two white plants
        obsSet.add_obs(modelWh.rvs(100), plantID=plant)
    for plant in range(2, 20): # 18 purple plants
        obsSet.add_obs(modelPu.rvs(100), plantID=plant)

    m = model.Model(dg, obsSet)
    logPobs = m.calc_fb()
    llDict = m.posterior_ll()

    mixModel = get_mix_model(modelWh, modelPu)

    for plant in range(20):
        obsLabel = obsSet.get_subset(plantID=plant)
        Le = entropy.LogPVector(numpy.array(llDict[obsLabel]))
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
    dg = model.DependencyGraph({'START':branches, 'chi':{'chi':term}})

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
        
