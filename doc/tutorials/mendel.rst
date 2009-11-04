=================================
Basic Mendelian Genetics Modeling
=================================

Modeling Mendel's Purple vs. White Alleles
------------------------------------------

We simply define our alleles by associating with each a probability
distribution for our observable, in this case "flower color"::

   >>> import mendel
   >>> from scipy import stats
   >>> Wh = mendel.RecessiveAllele("Wh", stats.norm(0,1))
   >>> Pu = mendel.DominantAllele("Pu", stats.norm(10,1))

We can collect a sample of observations from each allele, 
using the scipy.stats standard method ``rvs.(n)``, which 
returns a "random value sample" of *n* values e.g.::

   >>> Wh.rvs(10)
   array([-1.02362863, -0.09901247,  0.9929487 , -1.65607318, -0.35166926,
           1.28685628,  1.72235647,  0.17670913, -1.31171361, -0.69117854])

We can construct a chromosome containing just this locus, in a 
"white" flavor and "purple" flavor (the decimal number is
the map position of the locus on the chromosome, in Morgan units;
this is used for modeling homologous recombination, which is not
relevant to this tutorial)::

   >>> chrWh = mendel.Chromosome([(0.5,Wh)])
   >>> chrPu = mendel.Chromosome([(0.5,Pu)])

We can then construct two plants, one *Wh,Wh* and another *Pu,Pu*::

   >>> plantWh = mendel.DiploidGenome({1:(chrWh,chrWh)})
   >>> plantPu = mendel.DiploidGenome({1:(chrPu,chrPu)})

The ``get_phenotypes()`` method returns a list of all phenotypes
of the individual (i.e. all loci).  We only have one locus, so
we can get a sample of observations of its phenotype via::

   >>> plantPu.get_phenotypes()[0].rvs(10)
   array([ 10.20513478,  11.03047278,  11.26576122,   7.99768398,
           10.50421657,  10.31153061,   9.19066293,  12.51388695,
           10.12299845,  11.10217414])

``repr()`` of each plant shows its genotype::
   >>> plantPu
   {1: [(Pu,Pu)]}
   >>> plantWh
   {1: [(Wh,Wh)]}

We can construct a hybrid by crossing the two plants::

   >>> plantHy = plantPu * plantWh
   >>> plantHy
   {1: [(Pu,Wh)]}
   >>> plantHy.get_phenotypes()[0].rvs(10)
   array([ 11.61391677,   9.82060816,  10.0791646 ,   9.03163224,
           11.58503748,   9.37505245,   8.51441069,   9.19029652,
            7.81026714,  10.94369656])

Let's generate 10 children from the *Hy x Hy* cross::
   >>> [plantHy * plantHy for i in range(10)]
   [{1: [(Pu,Pu)]}, {1: [(Pu,Pu)]}, {1: [(Wh,Wh)]}, {1: [(Pu,Wh)]}, 
    {1: [(Wh,Wh)]}, {1: [(Wh,Pu)]}, {1: [(Pu,Wh)]}, {1: [(Wh,Pu)]}, 
    {1: [(Wh,Pu)]}, {1: [(Pu,Wh)]}]
