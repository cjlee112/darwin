
from darwin.model import *

# test code

def odc_test(p6=.5, n=100):
    'Occasionally Dishonest Casino example'
    p = (1. - p6) / 5.
    L = LinearState('L', EmissionDict({1:p, 2:p, 3:p, 4:p, 5:p, 6:p6}))
    p = 1. / 6.
    F = LinearState('F', EmissionDict({1:p, 2:p, 3:p, 4:p, 5:p, 6:p}))
    stop = LinearStateStop()
    sg = StateGraph({F:{F:0.95, L:0.05, stop:1.},
                     L:{F:0.1, L:0.9, stop:1.}})
    prior = StateGraph({'START':{F:2./3., L:1./3.}})
    hmm = DependencyGraph({'theta':{'theta':sg}, 'START':{'theta':prior}})

    s,obs = hmm.simulate_seq(n)
    obsLabel = ObsSequenceLabel(obs)
    m = Model(hmm, obsLabel)
    logPobs = m.calc_fb()
    llDict = m.posterior_ll()
    for i in range(n): # print posteriors
        obsLabel = ObsSequenceLabel(obs, i, 1)
        nodeLabel = hmm.get_var('theta', obsLabel=obsLabel)
        nodeF = Node(F, nodeLabel)
        nodeL = Node(L, nodeLabel)
        print '%s:%0.3f\t%s:%0.3f\tTRUE:%s,%d,%0.3f' % \
              (nodeF, m.posterior(nodeF),
               nodeL, m.posterior(nodeL),
               s[i], obs[i][0], exp(llDict[obsLabel][0]))
    return m
