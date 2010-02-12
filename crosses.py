# -*- coding: utf-8 -*-

from darwin.robomendel import *
from darwin.entropy import *

import math
import pylab

if __name__ == '__main__':
    no_progeny = 0.0
    white_plant = PeaPlant(genome=PeaPlant.white_genome)
    purple_plant = PeaPlant(genome=PeaPlant.purple_genome)
    steps = 100
    im_points = []
    ip_points = []

    w = 0.1
    e = 0.1
    for i in range(1, steps):
        d = float(i) / float(steps)
        n = 50

        ## Wh x Wh
        model = Multinomial({'white': d + (1-d)*e*w, 'purple': (1-d)*(1-e) + (1-d)*e*(1-w), 'dead': 0})
        ## Wh x Pu
        #model = Multinomial({'white': (1-d)*e*w, 'purple': (1-d)*(1-e) + (1-d)*e*(1-w), 'dead': d})

        n = 50
        ## No progeny
        #outcomes = ['dead']*n
        # White progeny
        outcomes = ['white']*n
        # Purple progeny
        #outcomes = ['purple']*n
        ##Mixed progeny
        #outcomes = ['purple']*(int((1-w)*n))
        #outcomes.extend(['white']*(int(w*n)))

        ## likelihood of observations from model
        l_e = discrete_sample_Le(outcomes, model)
        ## log of uninformitive prior
        k = 3
        lup = LogPVector(numpy.array([math.log(1./k)]*n))
        i_m = l_e - lup
        im_points.append((d, i_m.mean))

        h_e = He_discrete(outcomes)
        l_e = discrete_sample_Le(outcomes, model)
        i_p = -h_e - l_e
        ip_points.append((d, i_p.mean))
        
    pylab.plot([x for (x, y) in ip_points], [y for (x, y) in ip_points], 'bo')
    pylab.title('e=' + str(e) + ' w=' + str(w))
    pylab.xlabel('delta')

    pylab.ylabel('Ip, Im')

    pylab.plot([x for (x, y) in im_points], [y for (x, y) in im_points], 'r+')
    #pylab.title('e=' + str(e) + ' w=' + str(w))
    #pylab.xlabel('delta')
    pylab.show()

