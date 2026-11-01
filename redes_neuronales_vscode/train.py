import numpy as np, pickle
rng=np.random.default_rng(3)
def spirals(n,noise=0.35):
    t=np.sqrt(rng.uniform(0,1,n))*3.2*np.pi
    a=np.c_[t*np.cos(t), t*np.sin(t)]/(3.2*np.pi)*5
    X=np.r_[a,-a]+rng.normal(0,noise,(2*n,2)); y=np.r_[np.zeros(n),np.ones(n)]
    return X,y
Xtr,ytr=spirals(260); Xva,yva=spirals(120)
sizes=[2,64,64,1]
P=[(rng.normal(0,np.sqrt(2/a),(a,b)),np.zeros(b)) for a,b in zip(sizes,sizes[1:])]
def fwd(P,X):
    hs=[X]; h=X
    for i,(W,b) in enumerate(P):
        z=h@W+b; h=np.maximum(z,0) if i<len(P)-1 else 1/(1+np.exp(-z)); hs.append(h)
    return hs
def bce(p,y): p=np.clip(p[:,0],1e-7,1-1e-7); return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
E=3000; lr=0.003; m=[(np.zeros_like(W),np.zeros_like(b)) for W,b in P]; v=[(np.zeros_like(W),np.zeros_like(b)) for W,b in P]
snaps={}; L=[]
for ep in range(1,E+1):
    hs=fwd(P,Xtr); d=(hs[-1][:,0]-ytr)[:,None]/len(ytr)
    grads=[]
    for i in reversed(range(len(P))):
        W,b=P[i]; gW=hs[i].T@d+3e-4*W; gb=d.sum(0); grads.insert(0,(gW,gb))
        if i: d=(d@W.T)*(hs[i]>0)
    for i,((W,b),(gW,gb)) in enumerate(zip(P,grads)):
        mW,mb=m[i]; vW,vb=v[i]
        mW=0.9*mW+0.1*gW; mb=0.9*mb+0.1*gb; vW=0.999*vW+0.001*gW**2; vb=0.999*vb+0.001*gb**2
        m[i]=(mW,mb); v[i]=(vW,vb); c1=1-0.9**ep; c2=1-0.999**ep
        P[i]=(W-lr*(mW/c1)/(np.sqrt(vW/c2)+1e-8), b-lr*(mb/c1)/(np.sqrt(vb/c2)+1e-8))
    ptr=fwd(P,Xtr)[-1]; pva=fwd(P,Xva)[-1]
    L.append((bce(ptr,ytr),bce(pva,yva),((ptr[:,0]>.5)==ytr).mean(),((pva[:,0]>.5)==yva).mean()))
    snaps[ep]=[(W.copy(),b.copy()) for W,b in P]
print(L[0],L[499],L[1999],L[-1])
pickle.dump(dict(Xtr=Xtr,ytr=ytr,Xva=Xva,yva=yva,L=np.array(L),snaps={k:snaps[k] for k in sorted(set([1]+list(np.unique(np.round(E*(np.arange(1,391)/390)**3.2).astype(int).clip(1,E)))))}),open('work/run.pkl','wb'))
