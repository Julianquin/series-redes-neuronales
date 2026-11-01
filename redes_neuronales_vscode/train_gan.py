import numpy as np, pickle
rng = np.random.default_rng(3)
def real(n):
    k = rng.integers(0, 8, n); ang = k * np.pi / 4
    return np.c_[np.cos(ang), np.sin(ang)] * 2.0 + rng.normal(0, 0.12, (n, 2))
def init(sizes): return [[rng.normal(0, np.sqrt(2 / a), (a, b)), np.zeros(b)] for a, b in zip(sizes, sizes[1:])]
def fwd(P, x):
    hs = [x]
    for i, (W, b) in enumerate(P):
        z = hs[-1] @ W + b
        hs.append(z if i == len(P) - 1 else np.where(z > 0, z, 0.2 * z))
    return hs
def bwd(P, hs, d):
    g = [None] * len(P)
    for i in reversed(range(len(P))):
        W, b = P[i]; g[i] = [hs[i].T @ d, d.sum(0)]; d = d @ W.T
        if i: d = d * np.where(hs[i] > 0, 1, 0.2)
    return g, d
class Adam:
    def __init__(s, P, lr): s.lr = lr; s.m = [[0, 0] for _ in P]; s.v = [[0, 0] for _ in P]; s.t = 0
    def step(s, P, g):
        s.t += 1
        for i in range(len(P)):
            for j in range(2):
                s.m[i][j] = 0.5 * s.m[i][j] + 0.5 * g[i][j]; s.v[i][j] = 0.999 * s.v[i][j] + 0.001 * g[i][j] ** 2
                P[i][j] = P[i][j] - s.lr * (s.m[i][j] / (1 - 0.5 ** s.t)) / (np.sqrt(s.v[i][j] / (1 - 0.999 ** s.t)) + 1e-8)
if __name__ == "__main__":
    G = init([2, 128, 128, 2]); D = init([2, 128, 128, 1]); oG = Adam(G, 1e-3); oD = Adam(D, 1e-3)
    sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    ITERS, B = 4000, 256; L = []; snaps = {}
    keys = set(np.unique(np.round(ITERS * (np.arange(1, 391) / 390) ** 1.6).astype(int).clip(1, ITERS))) | {1}
    Zfix = rng.normal(0, 1, (600, 2)); Rfix = real(600)
    for it in range(1, ITERS + 1):
        xr = real(B); xf = fwd(G, rng.normal(0, 1, (B, 2)))[-1]
        x = np.r_[xr, xf]; yb = np.r_[np.ones(B), np.zeros(B)]
        hs = fwd(D, x); p = sig(hs[-1][:, 0])
        lD = -np.mean(yb * np.log(p + 1e-9) + (1 - yb) * np.log(1 - p + 1e-9))
        gD, _ = bwd(D, hs, ((p - yb) / len(yb))[:, None]); oD.step(D, gD)
        hg = fwd(G, rng.normal(0, 1, (B, 2))); hd = fwd(D, hg[-1]); q = sig(hd[-1][:, 0])
        lG = -np.mean(np.log(q + 1e-9))
        _, dx = bwd(D, hd, ((q - 1) / B)[:, None]); gG, _ = bwd(G, hg, dx); oG.step(G, gG)
        L.append((lD, lG, (p[:B] > .5).mean() * .5 + (p[B:] < .5).mean() * .5))
        if it in keys: snaps[it] = ([[w.copy(), b.copy()] for w, b in G], [[w.copy(), b.copy()] for w, b in D])
        if it % 500 == 0:
            f = fwd(G, Zfix)[-1]; ang = np.round(np.arctan2(f[:, 1], f[:, 0]) / (np.pi / 4)) % 8
            print(it, L[-1], np.bincount(ang.astype(int), minlength=8))
    pickle.dump(dict(L=np.array(L), snaps=snaps, Zfix=Zfix, Rfix=Rfix), open('work/run_gan.pkl', 'wb'))
