"""GAN que aprende a colocar pueblos falsos en el mapa de España (península + Baleares). Reanudable: python3 train_gan_es.py SEGUNDOS"""
import numpy as np, pickle, json, os, sys, time
from scipy.spatial import cKDTree
STATE = 'work/gan_es_state.pkl'; OUT = 'work/run_gan_es.pkl'
d = json.load(open('data/municipios_geo.json')); pen = [x for x in d if x['c'] not in ('Canarias', 'Ceuta', 'Melilla')]
LON0, LAT0, CS = -3.7, 40.2, np.cos(np.radians(40.2))
def to_xy(lon, lat): return np.c_[(np.asarray(lon) - LON0) * CS, np.asarray(lat) - LAT0] / 3.0
def to_lonlat(xy): return xy[:, 0] * 3.0 / CS + LON0, xy[:, 1] * 3.0 + LAT0
REAL = to_xy([x['lon'] for x in pen], [x['lat'] for x in pen]).astype(np.float64)
TREE = cKDTree(REAL); KM = 1 / (111.0 * 3.0)                     # 1 km en unidades normalizadas (aprox.)
def land_pct(F): return float((TREE.query(F)[0] < 10 * KM).mean())          # % de falsos a <10 km de un pueblo real
def coverage(F): return float((cKDTree(F).query(REAL)[0] < 15 * KM).mean())   # % de pueblos reales con un falso a <15 km
ZD, HID, ITERS, B = 8, 256, 8000, 512
sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))
def init(rng, sizes): return [[rng.normal(0, np.sqrt(2 / a), (a, b)), np.zeros(b)] for a, b in zip(sizes, sizes[1:])]
def fwd(P, x):
    hs = [x]
    for i, (W, b) in enumerate(P):
        z = hs[-1] @ W + b; hs.append(z if i == len(P) - 1 else np.where(z > 0, z, 0.2 * z))
    return hs
def bwd(P, hs, dd):
    g = [None] * len(P)
    for i in reversed(range(len(P))):
        W, b = P[i]; g[i] = [hs[i].T @ dd, dd.sum(0)]; dd = dd @ W.T
        if i: dd = dd * np.where(hs[i] > 0, 1, 0.2)
    return g, dd
def adam(P, st, g, lr):
    st['t'] += 1; t = st['t']
    for i in range(len(P)):
        for j in range(2):
            st['m'][i][j] = 0.5 * st['m'][i][j] + 0.5 * g[i][j]; st['v'][i][j] = 0.999 * st['v'][i][j] + 0.001 * g[i][j] ** 2
            P[i][j] -= lr * (st['m'][i][j] / (1 - 0.5 ** t)) / (np.sqrt(st['v'][i][j] / (1 - 0.999 ** t)) + 1e-8)
if __name__ == '__main__':
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 80; t0 = time.time()
    if os.path.exists(STATE): S = pickle.load(open(STATE, 'rb'))
    else:
        rng = np.random.default_rng(7)
        S = dict(rng=rng, G=init(rng, [ZD, HID, HID, 2]), D=init(rng, [2, HID, HID, 1]), it=0, L=[], snaps={}, metrics={},
                 Z=rng.normal(0, 1, (2500, ZD)))
        S['oG'] = dict(t=0, m=[[0, 0] for _ in S['G']], v=[[0, 0] for _ in S['G']]); S['oD'] = dict(t=0, m=[[0, 0] for _ in S['D']], v=[[0, 0] for _ in S['D']])
    keys = set(np.unique(np.round(ITERS * (np.arange(1, 131) / 130) ** 1.7).astype(int).clip(1, ITERS))) | {1}
    rng, G, D = S['rng'], S['G'], S['D']
    while S['it'] < ITERS and time.time() - t0 < budget:
        S['it'] += 1; it = S['it']
        xr = REAL[rng.integers(0, len(REAL), B)]; xf = fwd(G, rng.normal(0, 1, (B, ZD)))[-1]
        x = np.r_[xr, xf]; yb = np.r_[np.ones(B), np.zeros(B)]; hs = fwd(D, x); p = sig(hs[-1][:, 0])
        lD = -np.mean(yb * np.log(p + 1e-9) + (1 - yb) * np.log(1 - p + 1e-9))
        gD, _ = bwd(D, hs, ((p - yb) / len(yb))[:, None]); adam(D, S['oD'], gD, 2e-4)
        hg = fwd(G, rng.normal(0, 1, (B, ZD))); hd = fwd(D, hg[-1]); q = sig(hd[-1][:, 0]); lG = -np.mean(np.log(q + 1e-9))
        _, dx = bwd(D, hd, ((q - 1) / B)[:, None]); gG, _ = bwd(G, hg, dx); adam(G, S['oG'], gG, 2e-4)
        S['L'].append((lD, lG))
        if it in keys:
            S['snaps'][it] = ([[w.copy(), b.copy()] for w, b in G], [[w.copy(), b.copy()] for w, b in D])
            F = fwd(G, S['Z'])[-1]; S['metrics'][it] = (land_pct(F), coverage(F))
        if it % 500 == 0: print(it, round(lD, 3), round(lG, 3), S['metrics'].get(max(k for k in S['metrics'])), f'{time.time() - t0:.0f}s', flush=True)
    pickle.dump(S, open(STATE, 'wb'))
    if S['it'] >= ITERS:
        pickle.dump(dict(L=np.array(S['L']), snaps=S['snaps'], metrics=S['metrics'], Z=S['Z'], REAL=REAL, ITERS=ITERS), open(OUT, 'wb')); print('hecho')
    else: print('pausa en', S['it'])
