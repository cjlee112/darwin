
import random
from scipy import stats
from math import *
from Pycluster import kcluster


def moments(data):
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


def generateModels(n,modelKlass,mu,sigma):
    return [modelKlass(random.random()*mu,random.random()*sigma) for i in range(n)]


class TestData(list):
    def __init__(self,n,models,probs=None):
        '''TestData(n,probs,models) takes three arguments:
          n: number of observations to generate
          probs: list of probabilities, should sum to 1.0
          models: list of model objects'''
        if probs is None: # GENERATE RANDOM PROBABILITIES
            l=[random.random() for i in range(len(models))]
            s=sum(l)
            probs=map(lambda x:x/s,l) # NORMALIZE PROBABILITIES
        self.probs=probs
        self.models=models
        list.__init__(self)
        counts=stats.multinom(n,probs[0:-1],1)[0]
        for i in range(len(models)):
            self+=[x for x in models[i].rvs(counts[i])]
        self.counts=counts
        self.sort()
        #random.shuffle(self)
        #self.priors=len(models)*[1./len(models)]


class ObsDensity(list):
    def __init__(self,data,maxshift=1.5):
        'data must be sorted!!'
        list.__init__(self)  # CALL DEFAULT CONSTRUCTOR
        left=None
        last=data[0]
        count=0.
        p=1./len(data)
        for x in data[1:]:
            right=float(x+last)/2.
            count+=p
            if left is None: # 1ST VALUE
                left=last
            elif last==x: # JUST COUNT IDENTICAL VALUES
                continue
            elif (last-left)/float(right-last)>maxshift:
                self.append([left,last-(right-last),(left+last-(right-last))/2.,0.])
                left=last-(right-last)
            elif float(right-last)/(last-left)>maxshift:
                right=last+last-left
            self.append([left,right,last,count]) # SAVE INTERVAL AROUND last
            left=right
            last=x
            count=0.
        self.append([left,last,last,p]) # SAVE LAST VALUE

    def smoothDensity(self,w):
        start=0
        l=[0. for x in self] # INITIALIZE DENSITY TO ZEROES
        for left,right,x,p in self:
            while self[start][1]<=x-w: # SKIP PAST NON-OVERLAPPING INTERVALS
                start+=1
            if self[start][0]>x-w: # INCOMPLETE COVERAGE
                dw=x-w-self[start][0] # NOT ABLE TO COVER THIS AMOUNT
            else:
                dw=0.
            i=start
            while i<len(self) and self[i][0]<x+w: # FIND LAST OVERLAPPING INTERVALS
                i+=1
            if i>0 and self[i-1][1]<x+w: # INCOMPLETE COVERAGE
                dw+=x+w-self[i-1][1] # NOT ABLE TO COVER THIS AMOUNT
            i=start
            p/=2*w-dw # TRANSFORM INTO DENSITY, TO SPREAD OVER INTERVAL [x-w,x+w]
            while i<len(self) and self[i][0]<x+w: # FIND ALL OVERLAPPING INTERVALS
                if x-w>self[i][0]: # FIND OVERLAP START AND STOP
                    left=x-w
                else:
                    left=self[i][0]
                if x+w<self[i][1]:
                    right=x+w
                else:
                    right=self[i][1]
                l[i]+=p*(right-left)
                i+=1
        for i in range(len(self)): # SAVE THE FINAL SMOOTHED DENSITY
            self[i][3]=l[i]

    def relativeEntropy(self,model):
        'compute relative entropy contribution of each density slice, as list'
        d=0.
        l=[]
        for left,right,x,p in self:
            if p>0.:
                dd=p*log(p/((right-left)*model.pdf(x)))
            else:
                dd=0.
            d+=dd
            l.append(dd)
        return d,l

class Model(object):
    'wrapper for a distribution, adds update() method'
    def __init__(self,parent,data=None,prior=1.,delta=0.01,modelClass=stats.norm):
        self.parent=parent
        self._modelClass=modelClass
        if data is None:
            data=parent.data
        self.update(data,prior,delta)

    def update(self,data,prior,delta):
        'recompute model using data list and specified prior'
        mu,variance=moments(data)
        sd=sqrt(variance)
        if sd<delta/sqrt(2*pi): # ENFORCE MINIMUM RESOLUTION
            sd=delta/sqrt(2*pi)
##         print 'mu,variance,n = %3.2f, %3.2f, %d' %(mu,variance,len(data))
##         print data[:10]
        self._model=self._modelClass(mu,sd)
        self.prior=prior
    def __getattr__(self,attr):
        'just wrap all attributes and methods of self._model'
        return getattr(self._model,attr)


class RootModel(object):
    def __init__(self,data,delta):
        self.w=max(data)-min(data)
        self.delta=delta

    def computeP(self,data):
        return len(data)*log(self.delta/self.w)

    def pdf(self,x):
        return 1./self.w

class ModelLayer(list):
    '''Takes data list as obs, and acts as list of models of these obs'''
    def __init__(self,k,data,parent,delta,logConfidence=log(100.)):
        '''k: initial number of models to try
        data: list of observations
        parent: next model layer
        delta: minimum resolution width
        logConfidence: ln(odds) ratio required for model acceptance'''
        list.__init__(self) # CALL DEFAULT CONSTRUCTOR
        self.k=k
        self.data=data
        data.parent=self
        self.parent=parent
        parent.data=self
        #var,mean,l=self.bestmodel(data,len(data),1)
        #model0=stats.norm(mean,sqrt(var))
        self.models=[Model(self)]
        self.priors=[1.]
        self.logConfidence=logConfidence
        self.delta=delta
        self.lastP=self.parent.computeP(self.data) # COMPUTE LOG-P IN PARENT

    def pdf(self,x,models=None):
        'return Pr(x) according to total PDF for this model layer'
        if models is None:
            models=self.models
        p=0.
        for model in models:
            p+=model.prior*model.pdf(x)
        return p
        

    def computeP(self,data,models=None):
        '''Return log-P for all items in data list'''
        if models is None:
            models=self.models
        l=[]
        for model in models: # COMPUTE PRIOR-WEIGHTED PROB.
            prior=self.delta*model.prior
            l.append([p*prior for p in model.pdf(self.data)])
        for i in range(1,len(models)):
            l[0]=map(lambda x,y:x+y,l[0],l[i]) # SUM THE PROB VECTORS
        logP=0.
        for p in l[0]: # NOW TAKE PRODUCT IN LOG-SPACE
            logP+=log(p)
        return logP

    def extend(self,k):
        '''try to improve likelihood:
        likelihood of obs in our models times likelihood of our models
        in parent,
        vs. likelihood of obs in parent'''
        clusterid,err,nfound=kcluster([(p,) for p in self.data],k,npass=5)
        l=[[] for i in range(k)]
        j=0
        for i in clusterid: # CONSTRUCT MODEL LISTS OF DATA
            l[i].append(self.data[j])
            j+=1
        newmodel=[]
        means=[]
        for j in range(k):
            model=Model(self,l[j],float(len(l[j]))/len(self.data),self.delta)
            newmodel.append(model)
            means.append(model.args[0])
        logP=self.computeP(self.data,newmodel) # COMPUTE NEW LOG-P
        logP+=self.parent.computeP(means) # COMPUTE LOG-P FOR newmodels
        print 'logP:',logP
        if logP>self.lastP+self.logConfidence: # ACCEPT THE NEW MODEL
            self.models=newmodel
            self.lastP=logP
            self.k=k
            return True
        return False

##     def refine(self):
##         nmodel=len(self.models)
##         l=[]
##         for model in self.models:
##             l.append([p*model.prior for p in model.pdf(self.data)])
##         self.likelihoods=l
##         a=[[] for model in self.models]
##         j=0
##         logP=0.
##         logConf=0.
##         for x in self.data:
##             l=[(self.likelihoods[i][j],i) for i in range(nmodel)]
##             p=0.
##             for pmax,i in l: p+=pmax
##             logP+=log(p)
##             pmax,i=max(l) # FIND THE BEST MODEL (MAX POSTERIOR)
##             logConf+=log(pmax/p)
##             a[i].append(x) # ADD TO LIST FOR MODEL i
##             j+=1
##         n=float(len(self.data))
##         for i in range(nmodel):
##             self.models[i].update(a[i],len(a[i])/n)
##         print 'log p(obs)=%3.2f\tlog conf=%3.2f' % (logP,logConf)

        
##     def bestmodel(self,data,n,trials):
##         'get (var,mean,sample) with lowest variance'
##         l=[]
##         for i in range(trials):
##             sample=random.sample(data,n)
##             mean,var=moments(sample)
##             l.append((var,mean,sample))
##         return min(l)


            

##     def computeP(self):
##         l=[]
##         for i in range(len(self.data)):
##             p=0.
##             for j in range(len(self.models)):
##                 p+=self.priors[j]*self.likelihoods[j][i]
##             l.append(p)
##             pObs+=log(p)

