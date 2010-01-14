from scipy import stats
import mendel

Wh = mendel.RecessiveAllele("Wh", stats.norm(0,1))
Pu = mendel.DominantAllele("Pu", stats.norm(10,1))
chrWh = mendel.Chromosome([(0.5,Wh)])
chrPu = mendel.Chromosome([(0.5,Pu)])
plantWh = mendel.DiploidGenome({1:(chrWh,chrWh)})
plantPu = mendel.DiploidGenome({1:(chrPu,chrPu)})
plantHy = plantPu * plantWh

