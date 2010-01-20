
from darwin.model import *

# test code

def odc_test(p6=.5, n=100):
    'Occasionally Dishonest Casino example'
    p = (1. - p6) / 5.
    L = LinearState('L', EmissionDict({1:p, 2:p, 3:p, 4:p, 5:p, 6:p6}))
    p = 1. / 6.
    F = LinearState('F', EmissionDict({1:p, 2:p, 3:p, 4:p, 5:p, 6:p}))
    stop = LinearStateStop()
    sg = StateGraph({F:{F:0.95, L:0.05}, L:{F:0.1, L:0.9}})
    prior = StateGraph({'START':{F:2./3., L:1./3.}})
    term = StateGraph({F:{stop:1.}, L:{stop:1.}})
    dg = BasicHMM(sg, prior, term)

    s,obs = dg.simulate_seq(n)
    obsDict = obs_sequence(0, obs)
    f, b, fsub, bsub, ll = dg.calc_fb(obsDict)
    logPobs = b[START]
    llDict = posterior_ll(f)
    for i in range(n): # print posteriors
        nodeF = Node(F, 0, (i,), obsDict)
        nodeL = Node(L, 0, (i,), obsDict)
        print '%s:%0.3f\t%s:%0.3f\tTRUE:%s,%d,%0.3f' % \
              (nodeF, exp(fsub[nodeF] + b[nodeF] - logPobs),
               nodeL, exp(fsub[nodeL] + b[nodeL] - logPobs),
               s[i], obs[i], exp(llDict[nodeF.get_obs_label(i)][0]))
    return dg
