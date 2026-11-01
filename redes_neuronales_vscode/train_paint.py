import numpy as np, pickle, time, sys
from PIL import Image
f32 = np.float32
TH, TW = 144, 230
SRC = 'cuadros_pd/starry_night_neural_style.jpg'
img = np.asarray(Image.open(SRC).convert('RGB').resize((TW, TH), Image.LANCZOS), dtype=f32) / 255
ys, xs = np.mgrid[0:TH, 0:TW]
COORD = np.c_[(xs.ravel() + .5) / TW, (ys.ravel() + .5) / TW].astype(f32)      # y escalada por el ancho -> píxeles cuadrados
TARGET = img.reshape(-1, 3)
rng = np.random.default_rng(0)
M, SIGMA = 96, 9.0
B = rng.normal(0, SIGMA, (2, M)).astype(f32)

def featurize(kind, c):
    if kind == 'plain': return c
    z = 2 * np.pi * c @ B
    return np.concatenate([np.sin(z), np.cos(z)], 1).astype(f32)

def init(din, hid=192, depth=3):
    sizes = [din] + [hid] * depth + [3]
    return [[(rng.normal(0, np.sqrt(2 / a), (a, b))).astype(f32), np.zeros(b, f32)] for a, b in zip(sizes, sizes[1:])]

def forward(P, x):
    hs = [x]
    for i, (W, b) in enumerate(P):
        z = hs[-1] @ W + b
        hs.append(1 / (1 + np.exp(-z)) if i == len(P) - 1 else np.maximum(z, 0))
    return hs

class Adam:
    def __init__(s, P, lr): s.lr = lr; s.m = [[np.zeros_like(w), np.zeros_like(b)] for w, b in P]; s.v = [[np.zeros_like(w), np.zeros_like(b)] for w, b in P]; s.t = 0
    def step(s, P, g):
        s.t += 1; c1 = 1 - 0.9 ** s.t; c2 = 1 - 0.999 ** s.t
        for i in range(len(P)):
            for j in range(2):
                s.m[i][j] = 0.9 * s.m[i][j] + 0.1 * g[i][j]; s.v[i][j] = 0.999 * s.v[i][j] + 0.001 * g[i][j] ** 2
                P[i][j] -= s.lr * (s.m[i][j] / c1) / (np.sqrt(s.v[i][j] / c2) + 1e-8)

def train_step(P, opt, x, y):
    hs = forward(P, x); out = hs[-1]; n = len(x)
    d = (out - y) * 2 / (n * 3) * out * (1 - out); g = [None] * len(P)
    for i in reversed(range(len(P))):
        W, b = P[i]; g[i] = [hs[i].T @ d, d.sum(0)]
        if i: d = (d @ W.T) * (hs[i] > 0)
    opt.step(P, g)

def psnr(P, kind):
    out = forward(P, featurize(kind, COORD))[-1]; return float(-10 * np.log10(np.mean((out - TARGET) ** 2)))

if __name__ == '__main__':
    ITERS = int(sys.argv[1]); BS = int(sys.argv[2]); lr = 2e-3
    nets = {k: init(2 if k == 'plain' else 2 * M) for k in ('plain', 'fourier')}
    opts = {k: Adam(nets[k], lr) for k in nets}
    keys = sorted(set(np.unique(np.round(ITERS * (np.arange(1, 131) / 130) ** 2.0).astype(int).clip(1, ITERS))) | {1})
    snaps = {}; ps = {'plain': [], 'fourier': []}; t0 = time.time()
    for it in range(1, ITERS + 1):
        idx = rng.integers(0, len(COORD), BS)
        for k in nets: train_step(nets[k], opts[k], featurize(k, COORD[idx]), TARGET[idx])
        if it in keys:
            snaps[it] = {k: [[w.copy(), b.copy()] for w, b in nets[k]] for k in nets}
            for k in nets: ps[k].append((it, psnr(nets[k], k)))
        if it % 100 == 0 or it == 20: print(it, {k: round(ps[k][-1][1], 2) for k in ps}, f'{time.time() - t0:.0f}s', flush=True)
    pickle.dump(dict(snaps=snaps, ps=ps, B=B, TH=TH, TW=TW, M=M, SIGMA=SIGMA, ITERS=ITERS, BS=BS), open('work/run_paint.pkl', 'wb'))
    print('hecho')
