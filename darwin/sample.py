# -*- coding: utf-8 -*-

import random
import sys

from scipy import stats
from mendel import *

Wh = RecessiveLocus("Wh", stats.norm(10,1))
Pu = DominantLocus("Pu", stats.norm(1,1))


class PeaPlant(object):
    def __init__(self, loci=None):
        if loci is None:
            genes = {"white": Wh, "purple": Pu }
            self.loci = list()
            for i in range(0,2):
                key = random.choice(genes.keys())
                self.loci.append(genes[key])
        else:
            self.loci = loci

    def __mul__(self, mate):
        """Reproduce!"""
        return PeaPlant(loci=[random.choice(self.loci), random.choice(mate.loci)] )
        
    def obs(self, n=1):
        """Observe n times."""
        locus = self.loci[0]+self.loci[1]
        #print locus
        #print locus.phenotype
        return locus(n)
        #return (locus.phenotype.rvs(n))
        

def main(argv):
    pp_1 = PeaPlant()
    pp_2 = PeaPlant()
    print "parent 1 ", pp_1.obs(3)
    print "parent 2 ", pp_2.obs(3)
    offspring = pp_1 * pp_2
    print "offsrping", offspring.obs(3)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
