import numpy as np, pickle
from sklearn.datasets import load_digits
rng = np.random.default_rng(0)
X = (load_digits().images.reshape(-1, 64) / 16.0 > 0.4).astype(float)
sig = lambda z: 1 / (1 + np.exp(-z))
def train_rbm(data, nh, epochs, lr=0.02, tag=""):
    nv = data.shape[1]; W = rng.normal(0, 0.01, (nv, nh)); b = np.log(data.mean(0) + 1e-3) - np.log(1 - data.mean(0) + 1e-3); c = np.zeros(nh)
    vW = 0; chains = rng.random((200, nv)) < 0.5
    snaps, err = [], []; step = 0
    for ep in range(epochs):
        mom = 0.5 if ep < 5 else 0.9
        for bi in np.array_split(rng.permutation(len(data)), len(data) // 100):
            v0 = data[bi]; ph0 = sig(v0 @ W + c)
            h = rng.random(chains.shape[:1] + (nh,)) < sig(chains @ W + c)       # PCD
            for _ in range(1):
                pv = sig(h @ W.T + b); chains = rng.random(pv.shape) < pv
                ph = sig(chains @ W + c); h = rng.random(ph.shape) < ph
            gW = (v0.T @ ph0) / len(v0) - (chains.T @ ph) / len(chains)
            vW = mom * vW + lr * (gW - 2e-4 * W); W += vW
            b += lr * (v0.mean(0) - chains.mean(0)); c += lr * (ph0.mean(0) - ph.mean(0))
            rec = sig((rng.random(ph0.shape) < ph0) @ W.T + b); err.append(np.mean((rec - v0) ** 2)); step += 1
            if step % 3 == 0 or step < 30: snaps.append((step, W.copy(), b.copy(), c.copy()))
        print(tag, ep, err[-1])
    return W, b, c, snaps, np.array(err)
W1, b1, c1, S1, E1 = train_rbm(X, 150, 150, tag="rbm1")
H1 = sig(X @ W1 + c1)
W2, b2, c2, S2, E2 = train_rbm(H1, 60, 30, tag="rbm2")
pickle.dump(dict(X=X, W1=W1, b1=b1, c1=c1, S1=S1, E1=E1, S2=S2, E2=E2), open('work/run_dbn.pkl', 'wb'))
