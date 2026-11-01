import numpy as np, pickle
rng = np.random.default_rng(0)
V, T, d = 26, 8, 32
P = dict(E=rng.normal(0, 0.5, (V, d)), Pe=rng.normal(0, 0.5, (T, d)), Wq=rng.normal(0, d ** -0.5, (d, d)),
         Wk=rng.normal(0, d ** -0.5, (d, d)), Wv=rng.normal(0, d ** -0.5, (d, d)), Wo=rng.normal(0, d ** -0.5, (d, V)), bo=np.zeros(V))
s = d ** -0.5
def fwd(P, tok):
    x = P['E'][tok] + P['Pe']; Q, K, Vv = x @ P['Wq'], x @ P['Wk'], x @ P['Wv']
    S = Q @ K.transpose(0, 2, 1) * s; S -= S.max(-1, keepdims=True); A = np.exp(S); A /= A.sum(-1, keepdims=True)
    O = A @ Vv; lg = O @ P['Wo'] + P['bo']; lg -= lg.max(-1, keepdims=True); p = np.exp(lg); p /= p.sum(-1, keepdims=True)
    return x, Q, K, Vv, A, O, p
ITERS, B, lr = 300, 64, 0.001
m = {k: 0 for k in P}; v = {k: 0 for k in P}; L = []; snaps = {}
keys = set(np.unique(np.round(ITERS * (np.arange(1, 391) / 390) ** 1.3).astype(int).clip(1, ITERS))) | {1}
vt = rng.integers(0, V, (500, T))
for it in range(1, ITERS + 1):
    tok = rng.integers(0, V, (B, T)); y = tok[:, ::-1]
    x, Q, K, Vv, A, O, p = fwd(P, tok)
    bi, ti = np.meshgrid(np.arange(B), np.arange(T), indexing='ij')
    dl = p.copy(); dl[bi, ti, y] -= 1; dl /= B * T
    g = {'Wo': np.einsum('btd,btv->dv', O, dl), 'bo': dl.sum((0, 1))}
    dO = dl @ P['Wo'].T; dA = dO @ Vv.transpose(0, 2, 1); dVv = A.transpose(0, 2, 1) @ dO
    dS = A * (dA - (dA * A).sum(-1, keepdims=True))
    dQ = dS @ K * s; dK = dS.transpose(0, 2, 1) @ Q * s
    g['Wq'] = np.einsum('btd,bte->de', x, dQ); g['Wk'] = np.einsum('btd,bte->de', x, dK); g['Wv'] = np.einsum('btd,bte->de', x, dVv)
    dx = dQ @ P['Wq'].T + dK @ P['Wk'].T + dVv @ P['Wv'].T
    g['Pe'] = dx.sum(0); gE = np.zeros_like(P['E']); np.add.at(gE, tok, dx); g['E'] = gE
    for k in P:
        m[k] = 0.9 * m[k] + 0.1 * g[k]; v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
        P[k] = P[k] - lr * (m[k] / (1 - 0.9 ** it)) / (np.sqrt(v[k] / (1 - 0.999 ** it)) + 1e-8)
    pv = fwd(P, vt)[-1]
    L.append((-np.log(p[bi, ti, y] + 1e-9).mean(), (pv.argmax(-1) == vt[:, ::-1]).mean(), (pv.argmax(-1) == vt[:, ::-1]).all(1).mean()))
    if it in keys: snaps[it] = {k: P[k].copy() for k in P}
    if it % 100 == 0: print(it, L[-1])
pickle.dump(dict(L=np.array(L), snaps=snaps, V=V, T=T, d=d), open('work/run_attn.pkl', 'wb'))
