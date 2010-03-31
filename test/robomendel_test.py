from darwin import mendel
from darwin import model
from darwin import mixture
from darwin import entropy
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
    branches = model.BranchGenerator('chi', prior)
    dg = model.DependencyGraph({'START':branches, 'chi':{'chi':term}})

    obsSet = []
    for plant in range(2): # two white plants
        obsSet.append(model.ObsSequenceLabel((modelWh.rvs(100),), label=plant))
    for plant in range(2, 20): # 18 purple plants
        obsSet.append(model.ObsSequenceLabel((modelPu.rvs(100),), label=plant))

    m = model.Model(dg, tuple(obsSet))
    logPobs = m.calc_fb()
    llDict = m.posterior_ll()

    mixModel = get_mix_model(modelWh, modelPu)

    for plant in range(20):
        obsLabel = obsSet[plant].get_next()
        Le = entropy.LogPVector(numpy.array(llDict[obsLabel]))
        LeMix = entropy.sample_Le(obsLabel.seq[0], mixModel)
        Ie = Le - LeMix
        He = entropy.box_entropy(obsLabel.seq[0], 7)
        Ip = -Le - He
        print 'plant %d, Ie > %1.3f, mean = %1.3f\tIp > %1.3f, mean = %1.3f' \
              % (plant, Ie.get_bound(), Ie.mean, Ip.get_bound(), Ip.mean)
        
    return llDict

