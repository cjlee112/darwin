from darwin import mendel
from darwin import model
from scipy import stats

def pheno1_test(modelWh, modelPu):
    pstate = model.LinearState('Pu', modelPu)
    wstate = model.LinearState('Wh', modelWh)
    prior = model.StateGraph({'START':{pstate:0.9, wstate:0.1}})
    stop = model.StopState()
    term = model.StateGraph({pstate:{stop:1.}, wstate:{stop:1.}})
    d = {}
    for plant in range(20):
        d[plant] = prior
    dg = model.DependencyGraph({'START':{0:{0:d}},
                                0:{'STOP':model.TrivialMap({0:term})}})

    obsDict = {}
    for plant in range(2): # two white plants
        obsDict[(0,plant,0)] = modelWh.rvs(100)
    for plant in range(2, 20): # 18 purple plants
        obsDict[(0,plant,0)] = modelPu.rvs(100)

    f, b, fsub, bsub, ll = dg.calc_fb(obsDict)
    logPobs = b[model.START]
    llDict = model.posterior_ll(f)
    return llDict

