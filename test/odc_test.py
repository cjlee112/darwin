
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
    hmm = Model(LabelGraph({0:{0:sg}, 'START':{0:prior}}))

    s,obs = hmm.simulate_seq(n)
    obsGraph = ObsSequence(obs)
    logPobs = hmm.calc_fb((obsGraph,))
    llDict = hmm.posterior_ll()
    for i in range(n): # print posteriors
        obsLabel = obsGraph.get_label(i)
        nodeLabel = hmm.graph.get_label(0, (obsLabel,))
        nodeF = Node(F, nodeLabel)
        nodeL = Node(L, nodeLabel)
        print '%s:%0.3f\t%s:%0.3f\tTRUE:%s,%d,%0.3f' % \
              (nodeF, hmm.posterior(nodeF),
               nodeL, hmm.posterior(nodeL),
               s[i], obs[i], exp(llDict[nodeF.var][0]))
    return hmm
