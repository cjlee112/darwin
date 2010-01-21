# -*- coding: utf-8 -*-

from scipy import stats
from mendel import *
from entropy import *

Wh = RecessiveAllele("Wh", stats.norm(0,1))
Pu = DominantAllele("Pu", stats.norm(10,1))

chrWh = Chromosome([(0.5,Wh)])
chrPu = Chromosome([(0.5,Pu)])

plantWh = DiploidGenome({1:(chrWh,chrWh)})
plantPu = DiploidGenome({1:(chrPu,chrPu)})