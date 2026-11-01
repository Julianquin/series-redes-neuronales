import numpy as np, pickle
from sklearn.datasets import load_digits
rng = np.random.default_rng(0)
d = load_digits(); X = d.images.reshape(-1, 64) / 16.0; y = d.target
idx = rng.permutation(len(X)); tr, va = idx[:1400], idx[1400:]
sizes = [64, 128, 2, 128, 64]
P = [(rng.normal(0, np.sqrt(2 / a), (a, b)), np.zeros(b)) for a, b in zip(sizes, sizes[1:])]
sig = lambda z: 1 / (1 + np.exp(-z))
def fwd(P, X):
    hs = [X]; h = X
    for i, (W, b) in enumerate(P):
        z = h @ W + b
        h = sig(z) if i == 3 else (z if i == 1 else np.maximum(z, 0)); hs.append(h)
    return hs
ITERS, lr = 2000, 0.002
m = [[0, 0] for _ in P]; v = [[0, 0] for _ in P]; L = []; snaps = {}
keys = set(np.unique(np.round(ITERS * (np.arange(1, 391) / 390) ** 2.2).astype(int).clip(1, ITERS))) | {1}
for it in range(1, ITERS + 1):
    hs = fwd(P, X[tr]); out = hs[-1]
    dz = (out - X[tr]) * 2 / out.size * out * (1 - out) * 64  # MSE through sigmoid
    for i in reversed(range(len(P))):
        W, b = P[i]; gW = hs[i].T @ dz; gb = dz.sum(0)
        if i: dz = dz @ W.T * (1 if i == 2 else (hs[i] > 0))
        for j, gr in enumerate((gW, gb)):
            m[i][j] = 0.9 * m[i][j] + 0.1 * gr; v[i][j] = 0.999 * v[i][j] + 0.001 * gr ** 2
        P[i] = (W - lr * (m[i][0] / (1 - 0.9 ** it)) / (np.sqrt(v[i][0] / (1 - 0.999 ** it)) + 1e-8),
                b - lr * (m[i][1] / (1 - 0.9 ** it)) / (np.sqrt(v[i][1] / (1 - 0.999 ** it)) + 1e-8))
    L.append((np.mean((out - X[tr]) ** 2), np.mean((fwd(P, X[va])[-1] - X[va]) ** 2)))
    if it in keys: snaps[it] = [(W.copy(), b.copy()) for W, b in P]
    if it % 400 == 0: print(it, L[-1])
pickle.dump(dict(X=X, y=y, va=va, L=np.array(L), snaps=snaps), open('work/run_ae.pkl', 'wb'))
