import numpy as np, pickle, sys, time
from PIL import Image
f32 = np.float32; rng = np.random.default_rng(0)
A = 'cuadros_pd/'
PAINT = [('Van Gogh', 'La noche estrellada', 1889, A + 'jcjohnson_starry_night.jpg'),
         ('Hokusai', 'La gran ola', 1831, A + 'style_wave.jpg'),
         ('Turner', 'El naufragio', 1805, A + 'inputs_shipwreck.jpg'),
         ('Munch', 'El grito', 1893, A + 'inputs_the_scream.jpg'),
         ('Kandinsky', 'Composición VII', 1913, A + 'jcjohnson_composition_vii.jpg')]
HT, PS, HOLD = 256, 32, 0.24
GRAY = len(sys.argv) > 2 and sys.argv[2] == 'gray'
imgs = []
for _, _, _, p in PAINT:
    im = Image.open(p).convert('RGB'); im = im.resize((int(im.width * HT / im.height), HT), Image.LANCZOS)
    imgs.append(np.asarray(im, dtype=f32) / 255)
def to_in(x): return (x.mean(-1, keepdims=True) if GRAY else x) - 0.5
def split_x(w): return int(w * (1 - HOLD))
def sample(ci, n, test, r):
    im = imgs[ci]; w = im.shape[1]; cut = split_x(w)
    lo, hi = (cut + 4, w - PS) if test else (0, cut - 4 - PS)
    xs = r.integers(lo, hi + 1, n); ys = r.integers(0, HT - PS + 1, n)
    return np.stack([im[y:y + PS, x:x + PS] for x, y in zip(xs, ys)]), np.c_[xs, ys]
rv = np.random.default_rng(99)
VAL = [sample(c, 80, True, rv) for c in range(5)]
Xva = np.concatenate([v[0] for v in VAL]); yva = np.repeat(np.arange(5), 80)
CIN = 1 if GRAY else 3
def he(*s, fan): return (rng.normal(0, np.sqrt(2 / fan), s)).astype(f32)
P = dict(W1=he(9 * CIN, 16, fan=9 * CIN), b1=np.zeros(16, f32), W2=he(9 * 16, 32, fan=144), b2=np.zeros(32, f32),
         Wd=he(8 * 8 * 32, 5, fan=2048), bd=np.zeros(5, f32))
def im2col(X):
    B, H_, W_, C = X.shape; Xp = np.pad(X, ((0, 0), (1, 1), (1, 1), (0, 0)))
    return np.concatenate([Xp[:, i:i + H_, j:j + W_, :] for i in range(3) for j in range(3)], -1)
def col2im(dc, shape):
    B, H_, W_, C = shape; dXp = np.zeros((B, H_ + 2, W_ + 2, C), f32); dc = dc.reshape(B, H_, W_, 9, C); k = 0
    for i in range(3):
        for j in range(3): dXp[:, i:i + H_, j:j + W_, :] += dc[..., k, :]; k += 1
    return dXp[:, 1:-1, 1:-1, :]
def pool(A_):
    B, H_, W_, F = A_.shape; A6 = A_.reshape(B, H_ // 2, 2, W_ // 2, 2, F); M = A6.max((2, 4)); return M, (A6 == M[:, :, None, :, None, :])
def forward(P, X, keep=False):
    c1 = im2col(X); z1 = c1 @ P['W1'] + P['b1']; a1 = np.maximum(z1, 0); p1, m1 = pool(a1)
    c2 = im2col(p1); z2 = c2 @ P['W2'] + P['b2']; a2 = np.maximum(z2, 0); p2, m2 = pool(a2)
    F = p2.reshape(len(X), -1); lg = F @ P['Wd'] + P['bd']; lg = lg - lg.max(1, keepdims=True); pr = np.exp(lg); pr /= pr.sum(1, keepdims=True)
    return (pr, dict(c1=c1, z1=z1, a1=a1, m1=m1, p1=p1, c2=c2, z2=z2, m2=m2, F=F, X=X)) if keep else pr
def grads(P, X, y):
    pr, c = forward(P, X, True); B = len(X); dl = pr.copy(); dl[np.arange(B), y] -= 1; dl /= B
    g = dict(Wd=c['F'].T @ dl + 1e-4 * P['Wd'], bd=dl.sum(0))
    dp2 = (dl @ P['Wd'].T).reshape(B, 8, 8, 32)
    da2 = (c['m2'] * dp2[:, :, None, :, None, :]).reshape(B, 16, 16, 32) * (c['z2'] > 0)
    g['W2'] = c['c2'].reshape(-1, 144).T @ da2.reshape(-1, 32); g['b2'] = da2.sum((0, 1, 2))
    dp1 = col2im(da2.reshape(-1, 32) @ P['W2'].T, (B, 16, 16, 16))
    da1 = (c['m1'] * dp1[:, :, None, :, None, :]).reshape(B, 32, 32, 16) * (c['z1'] > 0)
    g['W1'] = c['c1'].reshape(-1, 9 * CIN).T @ da1.reshape(-1, 16); g['b1'] = da1.sum((0, 1, 2))
    return g, -np.log(pr[np.arange(B), y] + 1e-9).mean()
if __name__ == '__main__':
    ITERS = int(sys.argv[1]); lr = 1.5e-3; B = 60
    m = {k: 0 for k in P}; v = {k: 0 for k in P}; L = []; snaps = {}; t0 = time.time()
    keys = sorted(set(np.unique(np.round(ITERS * (np.arange(1, 131) / 130) ** 2.0).astype(int).clip(1, ITERS))) | {1})
    for it in range(1, ITERS + 1):
        xs, ys = [], []
        for c in range(5):
            x, _ = sample(c, B // 5, False, rng); xs.append(x); ys += [c] * (B // 5)
        X = np.concatenate(xs); flip = rng.random(len(X)) < .5; X[flip] = X[flip][:, :, ::-1]
        g, loss = grads(P, to_in(X), np.array(ys))
        for k in P:
            m[k] = 0.9 * m[k] + 0.1 * g[k]; v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
            P[k] = P[k] - lr * (m[k] / (1 - 0.9 ** it)) / (np.sqrt(v[k] / (1 - 0.999 ** it)) + 1e-8)
        if it in keys or it % 100 == 0:
            pv = forward(P, to_in(Xva)); vl = -np.log(pv[np.arange(len(yva)), yva] + 1e-9).mean(); acc = (pv.argmax(1) == yva).mean()
            L.append((it, loss, vl, acc))
            if it in keys: snaps[it] = {k: P[k].copy() for k in P}
            if it % 100 == 0: print(it, round(loss, 3), round(vl, 3), round(acc, 3), f'{time.time() - t0:.0f}s', flush=True)
    pickle.dump(dict(snaps=snaps, L=L, gray=GRAY, ITERS=ITERS), open(f'work/run_cuadros_{"gray" if GRAY else "rgb"}.pkl', 'wb'))
    print('hecho')
