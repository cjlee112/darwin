
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

def weightedMoments(data,w,minSigma=.01):
    'compute data [(mu, sigma)],[n] for each model in w'
    x=[[0.,0.] for l in w]
    n=[sum(l) for l in w]
    i=0
    for y in data: # SUM 1ST AND 2ND MOMENTS
        for j in range(len(w)):
            x[j][0]+=y*w[j][i]
            x[j][1]+=y*y*w[j][i]
        i+=1
    for j in range(len(w)): # COMPUTE MU, SIGMA, AND TRUNCATE SIGMA
        x[j][0]/=n[j]
        x[j][1]=x[j][1]/n[j] -x[j][0]*x[j][0]
        if x[j][1]>minSigma*minSigma:
            x[j][1]=sqrt(x[j][1])
        else:
            x[j][1]=minSigma
    return x,n


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
        return len(data)*log(self.delta/self.w),(1./self.w,)*len(data),(1.,)*len(data)

    def pdf(self,x):
        try: # HANDLE x AS LIST OF VALUES
            return (1./self.w,)*len(x)
        except TypeError: # HANDLE x AS AN INDIVIDUAL VALUE
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
        #self.models=[Model(self)]
        self.models=[]  # INITIALLY, MODEL IS EMPTY!
        self.w=[]
        self.logConfidence=logConfidence
        self.delta=delta
        self.lastP=self.computeP(self.data)[0] # COMPUTE LOG-P IN PARENT

    def pdf(self,x,models=None):
        'return Pr(x) according to total PDF for this model layer'
        if models is None:
            models=self.models
        try: # HANDLE x AS LIST OF VALUES
            return [self.pdf(y,models) for y in x]
        except TypeError: pass # HANDLE x AS AN INDIVIDUAL VALUE
        p=0.
        prior=1.
        for model in models:
            p+=model.prior*model.pdf(x)
            prior-=model.prior
        if prior>0.: # ADD COMPONENT DERIVED FROM PARENT DISTRIBUTION, IF ANY
            p+=prior*self.parent.pdf(x)
        return p
        

    def computeP(self,data,models=None):
        '''Return logP,li[],post[] for all obs in data list'''
        if models is None:
            models=self.models
        l=[]
        parentPrior=1.
        for model in models: # COMPUTE PRIOR-WEIGHTED PROB.
            parentPrior-=model.prior
            prior=self.delta*model.prior
            l.append([p*prior for p in model.pdf(self.data)])
        if parentPrior>0.: # GET CONTRIBUTION FROM PARENT
            lsum=[parentPrior*self.delta*p for p in self.parent.pdf(self.data)]
        else: # NO PROBABILITY FROM PARENT
            lsum=[0.]*len(self.data)
        for i in range(len(models)): # SUM LIKELIHOOD OF ALL MODELS FOR EACH OBS
            lsum=map(lambda x,y:x+y,lsum,l[i]) # SUM THE PROB VECTORS
        l=[map(lambda x,y:x/y,li,lsum) for li in l] # GET POSTERIOR OBS ASSIGNMENTS
        logP=0.
        for p in lsum: # NOW TAKE PRODUCT IN LOG-SPACE
            logP+=log(p)
        # TREAT models AS EMITTED FROM parent
        logPmodel=self.parent.computeP([model.args[0] for model in models])[0]
        return logP+logPmodel,lsum,l

    def weightedModels(self,data,w,modelKlass=stats.norm):
        moments,n=weightedMoments(data,w) # COMPUTE WEIGHTED MOMENTS
        models=[modelKlass(m[0],m[1]) for m in moments] # CREATE MODELS
        for i in range(len(moments)): # SAVE PRIOR FOR EACH MODEL
            models[i].prior=float(n[i])/len(data)
        return models

    def convergeModel(self,w,minDiff=log(1.1),nMax=5):
        'return converged models,obs-likelihoods,obs-posteriors'
        pLast=-1e20
        logP=-1e19
        i=0
        while logP-pLast>minDiff:
            pLast=logP
            models=self.weightedModels(self.data,w) # GENERATE ADJUSTED MODELS...
            logP,li,w=self.computeP(self.data,models) # COMPUTE PROBABILITIES
            print 'logP',logP
            i+=1
            if i>=nMax: # ONLY PERFORM nMax ITERATIONS
                break
        return models,li,w,logP

    def addREModel(self,obsDensity,minD=0.02,monitor=None,**kwargs):
        d,re=obsDensity.relativeEntropy(self) # COMPUTE REL ENT VS. CURRENT MODEL
        if monitor is not None:
            monitor(re,obsDensity,self)
        #reDensity=[]
        reSum=0.
        reMax=0.
        xmax=None
        start=None
        for i in range(len(obsDensity)): # FIND BIGGEST PEAK MASS
            y=re[i]/(obsDensity[i][1]-obsDensity[i][0])
            if y>minD: # ABOVE THRESHOLD
                if start is None: # START OF A NEW PEAK
                    start=obsDensity[i][0]
                reSum+=re[i]
                if reSum>reMax: # RECORD BIGGEST MASS PEAK
                    reMax=reSum
                    xmax=obsDensity[i][1]
                    xmin=start
            else: # BELOW THRESHOLD
                start=None
                reSum=0.
##         reDensity=[(re[i]/(obsDensity[i][1]-obsDensity[i][0]),i) for i in range(len(obsDensity))]
##         peak=max(reDensity) # FIND THE PEAK
##         if peak[0]<minD:
        if xmax is None:
            return None
##         i=peak[1] # FIND [MIN,MAX] INTERVAL ABOVE minD THRESHOLD
##         while i>0 and reDensity[i-1][0]>minD:
##             i-=1
##         xmin=obsDensity[i][0]
##         i=peak[1]
##         while i+1<len(obsDensity) and reDensity[i+1][0]>minD:
##             i+=1
##         xmax=obsDensity[i][1]
        w=[[x for x in l] for l in self.w] # DEEP COPY OF OBS WEIGHT MATRIX
        wNew=[0.]*len(self.data) # ADD NEW MODEL INITIALLY WITH NO OBS
        i=0
        while i<len(self.data) and self.data[i]<xmin: # SKIP PAST OBS TO THE LEFT OF PEAK
            i+=1
        while i<len(self.data) and self.data[i]<xmax: # MARK OBS IN THIS PEAK
            for l in w: # REMOVE THIS OBS FROM OLD MODELS
                l[i]=0.
            wNew[i]=1. # ASSIGN IT TO NEW MODEL
            i+=1
        w.append(wNew) # ADD NEW MODEL TO THE TEMPORARY WEIGHT MATRIX
        return self.convergeModel(w,**kwargs) # GENERATE A CONVERGED MODEL FROM THIS START

    def buildModels(self,obsDensity=None,wSmooth=0.5,minAccept=log(100.),**kwargs):
        if obsDensity is None:
            obsDensity=ObsDensity(self.data)
            obsDensity.smoothDensity(wSmooth)
        pLast=-1e20
        logP=-1e19
        i=0
        while logP-pLast>minAccept:
            try: # SAVE NEWLY ACCEPTED MODEL
                self.models=models
                self.w=w
                print 'added model',i
                i+=1
            except NameError: # NO MODEL TO SAVE YET
                pass
            pLast=logP
            result=self.addREModel(obsDensity,**kwargs)
            if result is None:
                break
            else:
                models,li,w,logP=result
        return pLast

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
        #means=[]
        for j in range(k):
            model=Model(self,l[j],float(len(l[j]))/len(self.data),self.delta)
            newmodel.append(model)
            #means.append(model.args[0])
        logP=self.computeP(self.data,newmodel)[0] # COMPUTE NEW LOG-P
        #logP+=self.parent.computeP(means) # COMPUTE LOG-P FOR newmodels
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

