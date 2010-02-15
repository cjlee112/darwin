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
    stop = model.StopState()
    term = model.StateGraph({pstate:{stop:1.}, wstate:{stop:1.}})
    dg = model.NodeGraph({'START':{'chi':prior}, 'chi':{'chi':term}})

    d = {}
    for plant in range(2): # two white plants
        d[plant] = modelWh.rvs(100)
    for plant in range(2, 20): # 18 purple plants
        d[plant] = modelPu.rvs(100)
    obsGraph = model.ObsGraph({'START':d})

    m = model.Model(dg, (obsGraph,))
    logPobs = m.calc_fb()
    llDict = m.posterior_ll()

    mixModel = get_mix_model(modelWh, modelPu)

    for plant in range(20):
        obs = d[plant]
        obsLabel = obsGraph.get_label(plant)
        nodeLabel = dg.get_label('chi', (obsLabel,))
        Le = entropy.LogPVector(numpy.array(llDict[nodeLabel]))
        LeMix = entropy.sample_Le(obs, mixModel)
        Ie = Le - LeMix
        He = entropy.box_entropy(obs, 7)
        Ip = -Le - He
        print 'plant %d, Ie > %1.3f, mean = %1.3f\tIp > %1.3f, mean = %1.3f' \
              % (plant, Ie.get_bound(), Ie.mean, Ip.get_bound(), Ip.mean)
        
    return llDict

