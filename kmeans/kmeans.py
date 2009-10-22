
import random
from scipy import stats
from math import *
from Pycluster import *


class TestData(list):
    def __init__(self,n,probs,models):
        self.probs=probs
        self.models=models
        list.__init__(self)
        counts=stats.multinom(n,probs,1)[0]
        for i in range(len(models)):
            self+=[x for x in models[i].rvs(counts[i])]
        self.counts=counts
        random.shuffle(self)
        #self.priors=len(models)*[1./len(models)]


class NormalModel(object):
    _modelClass=stats.norm
    def __init__(self,parent,data=None,prior=1.):
        self.parent=parent
        if data is None:
            data=parent.data
        self.update(data,prior)

    def update(self,data,prior):
        mu,variance=self.parent.moments(data)
##         print 'mu,variance,n = %3.2f, %3.2f, %d' %(mu,variance,len(data))
##         print data[:10]
        self._model=self._modelClass(mu,sqrt(variance))
        self.prior=prior
    def __getattr__(self,attr):
        return getattr(self._model,attr)


class KModels(object):
    def __init__(self,k,data):
        self.k=k
        self.data=data
        #var,mean,l=self.bestmodel(data,len(data),1)
        #model0=stats.norm(mean,sqrt(var))
        self.models=[NormalModel(self)]
        self.priors=[1.]
        self.refine()

    def refine(self):
        nmodel=len(self.models)
        l=[]
        for model in self.models:
            l.append([p*model.prior for p in model.pdf(self.data)])
        self.likelihoods=l
        a=[[] for model in self.models]
        j=0
        logP=0.
        logConf=0.
        for x in self.data:
            l=[(self.likelihoods[i][j],i) for i in range(nmodel)]
            p=0.
            for pmax,i in l: p+=pmax
            logP+=log(p)
            pmax,i=max(l) # FIND THE BEST MODEL (MAX POSTERIOR)
            logConf+=log(pmax/p)
            a[i].append(x) # ADD TO LIST FOR MODEL i
            j+=1
        n=float(len(self.data))
        for i in range(nmodel):
            self.models[i].update(a[i],len(a[i])/n)
        print 'log p(obs)=%3.2f\tlog conf=%3.2f' % (logP,logConf)

        
    def moments(self,data):
        'compute data mean, variance'
        x=0.
        x2=0.
        for y in data:
            x+=y
            x2+=y*y
        n=len(data)
        x/=n
        x2/=n
        return x,x2-x*x

    def bestmodel(self,data,n,trials):
        'get (var,mean,sample) with lowest variance'
        l=[]
        for i in range(trials):
            sample=random.sample(data,n)
            mean,var=self.moments(sample)
            l.append((var,mean,sample))
        l.sort()
        return l[0]


            

##     def computeP(self):
##         l=[]
##         for i in range(len(self.data)):
##             p=0.
##             for j in range(len(self.models)):
##                 p+=self.priors[j]*self.likelihoods[j][i]
##             l.append(p)
##             pObs+=log(p)

