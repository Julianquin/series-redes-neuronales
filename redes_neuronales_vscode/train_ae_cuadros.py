import numpy as np, pickle, sys, time
from PIL import Image
f32 = np.float32; rng = np.random.default_rng(0)
A = 'cuadros_pd/'
PAINT = [('Van Gogh', A + 'jcjohnson_starry_night.jpg'), ('Hokusai', A + 'style_wave.jpg'), ('Turner', A + 'inputs_shipwreck.jpg'),
         ('Munch', A + 'inputs_the_scream.jpg'), ('Kandinsky', A + 'jcjohnson_composition_vii.jpg')]
HT, PS, HOLD = 128, 16, 0.24
imgs = []
for _, p in PAINT:
    im = Image.open(p).convert('RGB'); imgs.append(np.asarray(im.resize((int(im.width * HT / im.height), HT), Image.LANCZOS), dtype=f32) / 255)
def sample(ci, n, test, r):
    im = imgs[ci]; w = im.shape[1]; cut = int(w * (1 - HOLD))
    lo, hi = (cut + 2, w - PS) if test else (0, cut - 2 - PS)
    xs = r.integers(lo, hi + 1, n); ys = r.integers(0, HT - PS + 1, n)
    return np.stack([im[y:y + PS, x:x + PS].ravel() for x, y in zip(xs, ys)])
rt = np.random.default_rng(1)
XTR = np.concatenate([sample(c, 2400, False, rt) for c in range(5)]); YTR = np.repeat(np.arange(5), 2400)
rv = np.random.default_rng(2)
XVA = np.concatenate([sample(c, 200, True, rv) for c in range(5)]); YVA = np.repeat(np.arange(5), 200)
D = PS * PS * 3
def init(k):
    sizes = [D, 256, 64, k, 64, 256, D]
    return [[rng.normal(0, np.sqrt(2 / a), (a, b)).astype(f32), np.zeros(b, f32)] for a, b in zip(sizes, sizes[1:])]
LAT = 2   # índice de la capa latente (salida lineal)
def forward(P, X):
    hs = [X]
    for i, (W, b) in enumerate(P):
        z = hs[-1] @ W + b
        hs.append(1 / (1 + np.exp(-z)) if i == len(P) - 1 else (z if i == LAT else np.maximum(z, 0)))
    return hs
def encode(P, X):
    h = X
    for i in range(LAT + 1):
        z = h @ P[i][0] + P[i][1]; h = z if i == LAT else np.maximum(z, 0)
    return h
def decode(P, Z):
    h = Z
    for i in range(LAT + 1, len(P)):
        z = h @ P[i][0] + P[i][1]; h = 1 / (1 + np.exp(-z)) if i == len(P) - 1 else np.maximum(z, 0)
    return h
def step(P, opt, X):
    hs = forward(P, X); out = hs[-1]; n = len(X)
    d = (out - X) * 2 / out.size * out * (1 - out); g = [None] * len(P)
    for i in reversed(range(len(P))):
        W, b = P[i]; g[i] = [hs[i].T @ d, d.sum(0)]
        if i: d = d @ W.T * (1 if i == LAT + 1 else (hs[i] > 0))
    opt['t'] += 1; t = opt['t']
    for i in range(len(P)):
        for j in range(2):
            opt['m'][i][j] = 0.9 * opt['m'][i][j] + 0.1 * g[i][j]; opt['v'][i][j] = 0.999 * opt['v'][i][j] + 0.001 * g[i][j] ** 2
            P[i][j] -= 1e-3 * (opt['m'][i][j] / (1 - 0.9 ** t)) / (np.sqrt(opt['v'][i][j] / (1 - 0.999 ** t)) + 1e-8)
    return float(np.mean((out - X) ** 2))
if __name__ == '__main__':
    ITERS = int(sys.argv[1]); BS = 256
    nets = {k: init(k) for k in (2, 32)}
    opts = {k: dict(t=0, m=[[0, 0] for _ in nets[k]], v=[[0, 0] for _ in nets[k]]) for k in nets}
    keys = sorted(set(np.unique(np.round(ITERS * (np.arange(1, 131) / 130) ** 2.0).astype(int).clip(1, ITERS))) | {1})
    snaps = {}; L = {2: [], 32: []}; t0 = time.time()
    for it in range(1, ITERS + 1):
        idx = rng.integers(0, len(XTR), BS)
        for k in nets: step(nets[k], opts[k], XTR[idx])
        if it in keys:
            snaps[it] = {k: [[w.copy(), b.copy()] for w, b in nets[k]] for k in nets}
            for k in nets: L[k].append((it, float(np.mean((forward(nets[k], XVA)[-1] - XVA) ** 2))))
        if it % 250 == 0: print(it, {k: round(L[k][-1][1], 4) for k in L}, f'{time.time() - t0:.0f}s', flush=True)
    pickle.dump(dict(snaps=snaps, L=L, ITERS=ITERS), open('work/run_ae_cuadros.pkl', 'wb')); print('hecho')
