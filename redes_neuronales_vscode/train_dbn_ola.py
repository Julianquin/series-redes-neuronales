import numpy as np, pickle, sys
from PIL import Image
rng = np.random.default_rng(0); f32 = np.float32
SRC = 'cuadros_pd/style_wave.jpg'
HT, PS, HOLD = 256, 20, 0.24
im = Image.open(SRC).convert('L'); im = im.resize((int(im.width * HT / im.height), HT), Image.LANCZOS)
GRAY = np.asarray(im, f32) / 255; INK = (GRAY < 0.55).astype(f32)          # 1 = tinta
WID = INK.shape[1]; CUT = int(WID * (1 - HOLD))
def patches(n, test, r, lo_ink=0.12, hi_ink=0.85):
    out = []
    lo, hi = (CUT + 2, WID - PS) if test else (0, CUT - 2 - PS)
    while len(out) < n:
        x = r.integers(lo, hi + 1); y = r.integers(0, HT - PS + 1); p = INK[y:y + PS, x:x + PS]
        if lo_ink <= p.mean() <= hi_ink: out.append(p.ravel())
    return np.array(out, f32)
X = patches(10000, False, np.random.default_rng(1)); XTE = patches(400, True, np.random.default_rng(2))
sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))
def train_rbm(data, nh, epochs, lr=0.01, tag='', every=30):
    nv = data.shape[1]; W = rng.normal(0, 0.01, (nv, nh)).astype(f32); m_ = data.mean(0)
    b = (np.log(m_ + 1e-3) - np.log(1 - m_ + 1e-3)).astype(f32); c = np.zeros(nh, f32); vW = 0
    chains = (rng.random((200, nv)) < 0.5).astype(f32); snaps, err = [], []; step = 0
    for ep in range(epochs):
        mom = 0.5
        for bi in np.array_split(rng.permutation(len(data)), len(data) // 100):
            v0 = data[bi]; ph0 = sig(v0 @ W + c)
            h = (rng.random((len(chains), nh)) < sig(chains @ W + c)).astype(f32)
            pv = sig(h @ W.T + b); chains = (rng.random(pv.shape) < pv).astype(f32); ph = sig(chains @ W + c)
            gW = (v0.T @ ph0) / len(v0) - (chains.T @ ph) / len(chains)
            vW = mom * vW + lr * (gW - 1e-3 * W); W = W + vW
            b = b + lr * (v0.mean(0) - chains.mean(0)); c = c + lr * (ph0.mean(0) - ph.mean(0))
            rec = sig((rng.random(ph0.shape) < ph0).astype(f32) @ W.T + b); err.append(float(np.mean((rec - v0) ** 2))); step += 1
            if step % every == 0 or step <= 30: snaps.append((step, W.copy(), b.copy(), c.copy()))
        if ep % 25 == 24: print(tag, ep + 1, round(err[-1], 4), flush=True)
    return W, b, c, snaps, np.array(err)
if __name__ == '__main__':
    W1, b1, c1, S1, E1 = train_rbm(X, 300, 150, tag='rbm1', every=50)
    H1 = sig(X @ W1 + c1)
    W2, b2, c2, S2, E2 = train_rbm(H1, 120, 30, tag='rbm2', every=10)
    pickle.dump(dict(W1=W1, b1=b1, c1=c1, S1=S1, E1=E1, S2=S2, E2=E2), open('work/run_dbn_ola_v2.pkl', 'wb')); print('hecho')
