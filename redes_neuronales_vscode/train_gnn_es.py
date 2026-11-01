"""GNN vs MLP: ¿de qué comunidad es este pueblo? (nombre / nombre + vecinos). python3 train_gnn_es.py"""
import numpy as np, json, pickle, zlib, unicodedata
import scipy.sparse as sp
from scipy.spatial import cKDTree
rng = np.random.default_rng(0)
d = [x for x in json.load(open('data/municipios_geo.json')) if x['c'] not in ('Ceuta', 'Melilla')]
N = len(d); COMM = sorted(set(x['c'] for x in d)); y = np.array([COMM.index(x['c']) for x in d]); C = len(COMM)
lat = np.array([x['lat'] for x in d]); lon = np.array([x['lon'] for x in d])
XY = np.c_[lon * np.cos(np.radians(lat)), lat]                       # solo para construir el grafo de vecinos
K = 8; _, nb = cKDTree(XY).query(XY, K + 1); rows = np.repeat(np.arange(N), K); cols = nb[:, 1:].ravel()
A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N)).tocsr(); A = ((A + A.T) > 0).astype(np.float64) + sp.eye(N)
dg = np.asarray(A.sum(1)).ravel(); Dm = sp.diags(1 / np.sqrt(dg)); AH = (Dm @ A @ Dm).tocsr()
EDGES = np.array(sp.triu(A, 1).nonzero()).T
DIM = 1024
def feats(name):
    s = '^' + unicodedata.normalize('NFC', name.lower()) + '$'; v = np.zeros(DIM)
    for n in (2, 3):
        for i in range(len(s) - n + 1): v[zlib.crc32(s[i:i + n].encode()) % DIM] += 1
    return v / (np.linalg.norm(v) + 1e-9)
X = np.stack([feats(x['n']) for x in d])
idx = rng.permutation(N); TR = np.sort(idx[:N // 10]); TE = np.sort(idx[N // 10:])
AX2 = AH @ (AH @ X)                                                   # nombres de vecinos y vecinos de vecinos
def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def init(din, h=64): return dict(W1=rng.normal(0, np.sqrt(2 / din), (din, h)), b1=np.zeros(h), W2=rng.normal(0, np.sqrt(2 / h), (h, C)), b2=np.zeros(C))
def fwd(P, kind):
    Xin = X if kind == 'mlp' else AX2; Z1 = Xin @ P['W1'] + P['b1']; Hh = np.maximum(Z1, 0)
    Z2 = Hh @ P['W2'] + P['b2']
    if kind == 'gnn': Z2 = AH @ Z2
    return Xin, Z1, Hh, softmax(Z2)
def step(P, st, kind, lr=5e-3):
    Xin, Z1, Hh, p = fwd(P, kind); dz = np.zeros_like(p); dz[TR] = p[TR]; dz[TR, y[TR]] -= 1; dz /= len(TR)
    if kind == 'gnn': dz = AH.T @ dz
    g = dict(W2=Hh.T @ dz + 1e-4 * P['W2'], b2=dz.sum(0)); dh = dz @ P['W2'].T * (Z1 > 0)
    g['W1'] = Xin.T @ dh + 1e-4 * P['W1']; g['b1'] = dh.sum(0); st['t'] += 1; t = st['t']
    for k in P:
        st['m'][k] = 0.9 * st['m'].get(k, 0) + 0.1 * g[k]; st['v'][k] = 0.999 * st['v'].get(k, 0) + 0.001 * g[k] ** 2
        P[k] = P[k] - lr * (st['m'][k] / (1 - 0.9 ** t)) / (np.sqrt(st['v'][k] / (1 - 0.999 ** t)) + 1e-8)
    return -np.log(p[TR, y[TR]] + 1e-9).mean(), -np.log(p[TE, y[TE]] + 1e-9).mean(), (p[TE].argmax(1) == y[TE]).mean()
if __name__ == '__main__':
    E = 300; keys = sorted(set(np.unique(np.round(E * (np.arange(1, 101) / 100) ** 1.6).astype(int).clip(1, E))))
    out = dict(N=N, COMM=COMM, y=y, lat=lat, lon=lon, names=[x['n'] for x in d], prov=[x['p'] for x in d], TR=TR, TE=TE, EDGES=EDGES, keys=keys)
    for kind in ('mlp', 'gnn'):
        P = init(DIM); st = dict(t=0, m={}, v={}); L = []; preds = {}
        for e in range(1, E + 1):
            L.append(step(P, st, kind))
            if e in keys: preds[e] = fwd(P, kind)[3].argmax(1).astype(np.int16)
        out[kind] = dict(L=np.array(L), preds=preds); print(kind, 'acierto examen:', round(L[-1][2], 3), 'pérdida', np.round(L[-1][:2], 3))
    pickle.dump(out, open('work/run_gnn_es.pkl', 'wb'))
