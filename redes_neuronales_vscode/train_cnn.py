import numpy as np, pickle
from sklearn.datasets import load_digits
rng=np.random.default_rng(1)
d=load_digits(); X=d.images/16.0; y=d.target
idx=rng.permutation(len(X)); tr,va=idx[:1400],idx[1400:]
Xtr,ytr,Xva,yva=X[tr],y[tr],X[va],y[va]
def im2col(X):
    Xp=np.pad(X,((0,0),(1,1),(1,1)))
    return np.stack([Xp[:,i:i+8,j:j+8] for i in range(3) for j in range(3)],-1).reshape(len(X),64,9)
Ctr,Cva=im2col(Xtr),im2col(Xva)
K=8
P=dict(Wc=rng.normal(0,0.3,(9,K)),bc=np.zeros(K),Wd=rng.normal(0,np.sqrt(2/128),(128,10)),bd=np.zeros(10))
def fwd(P,C):
    Z=C@P['Wc']+P['bc']; A=np.maximum(Z,0).reshape(-1,8,8,K)
    A6=A.reshape(-1,4,2,4,2,K); M=A6.max((2,4)); F=M.reshape(-1,128)
    lg=F@P['Wd']+P['bd']; lg-=lg.max(1,keepdims=True); p=np.exp(lg); p/=p.sum(1,keepdims=True)
    return Z,A6,M,F,p
def ce(p,y): return -np.mean(np.log(p[np.arange(len(y)),y]+1e-9))
E=400; lr=0.0015; m={k:0 for k in P}; v={k:0 for k in P}; snaps={}; L=[]
keys=set(np.unique(np.round(E*(np.arange(1,391)/390)**3).astype(int).clip(1,E)))|{1}
for ep in range(1,E+1):
    Z,A6,M,F,p=fwd(P,Ctr); n=len(ytr)
    dl=p.copy(); dl[np.arange(n),ytr]-=1; dl/=n
    g={'Wd':F.T@dl+1e-4*P['Wd'],'bd':dl.sum(0)}
    dM=(dl@P['Wd'].T).reshape(-1,4,4,K)
    mask=(A6==M[:,:,None,:,None,:])
    dA=(mask*dM[:,:,None,:,None,:]).reshape(-1,64,K)*(Z>0)
    g['Wc']=Ctr.reshape(-1,9).T@dA.reshape(-1,K); g['bc']=dA.sum((0,1))
    for k in P:
        m[k]=0.9*m[k]+0.1*g[k]; v[k]=0.999*v[k]+0.001*g[k]**2
        P[k]=P[k]-lr*(m[k]/(1-0.9**ep))/(np.sqrt(v[k]/(1-0.999**ep))+1e-8)
    pv=fwd(P,Cva)[-1]
    L.append((ce(p,ytr),ce(pv,yva),(p.argmax(1)==ytr).mean(),(pv.argmax(1)==yva).mean()))
    if ep in keys: snaps[ep]={k:P[k].copy() for k in P}
print(L[0],L[50],L[200],L[-1])
pickle.dump(dict(Xva=Xva,yva=yva,Cva=Cva,L=np.array(L),snaps=snaps),open('work/run_cnn.pkl','wb'))
