import pickle, numpy as np
from PIL import Image
import train_dbn_ola as T
P = T.PS; sig = T.sig
def load(path='work/run_dbn_ola_v2.pkl'):
    r = pickle.load(open(path, 'rb')); return r['W1'], r['b1'], r['c1']
def region():
    y0, y1 = 20, 236; x0 = T.CUT + 2; return T.INK[y0:y1, x0:T.WID], (y0, x0)
def damage(R, frac=0.4, seed=7):
    rg = np.random.default_rng(seed); M = rg.random(R.shape) < frac
    return np.where(M, (rg.random(R.shape) < 0.5).astype(np.float32), R), M
def repair(Wt, b, c, D, M, iters=6, stride=4, hist=False):
    est = D.copy().astype(np.float32); Hh, Ww = est.shape; out = []
    ys = list(range(0, Hh - P + 1, stride)); xs = list(range(0, Ww - P + 1, stride))
    if ys[-1] != Hh - P: ys.append(Hh - P)
    if xs[-1] != Ww - P: xs.append(Ww - P)
    for k in range(iters):
        pats = np.stack([est[y:y + P, x:x + P].ravel() for y in ys for x in xs])
        rec = sig(sig(pats @ Wt + c) @ Wt.T + b)
        acc = np.zeros_like(est); cnt = np.zeros_like(est); i = 0
        for y in ys:
            for x in xs:
                acc[y:y + P, x:x + P] += rec[i].reshape(P, P); cnt[y:y + P, x:x + P] += 1; i += 1
        est = np.where(M, acc / np.maximum(cnt, 1), D); out.append(est.copy())
    return out if hist else est
if __name__ == '__main__':
    W, b, c = load(); R, _ = region(); D, M = damage(R)
    hist = repair(W, b, c, D, M, hist=True)
    for k, e in enumerate(hist): print(k + 1, 'aciertos en píxeles dañados:', round(float(((e > .5) == (R > .5))[M].mean()), 3))
    print('antes:', round(float((D == R)[M].mean()), 3))
    to = lambda a: Image.fromarray(((1 - (a > .5)) * 255).astype(np.uint8))
    s = Image.new('L', (R.shape[1] * 3 + 20, R.shape[0]), 128)
    for i, a in enumerate([R, D, hist[-1]]): s.paste(to(a), (i * (R.shape[1] + 10), 0))
    s = s.resize((s.width * 2, s.height * 2), Image.NEAREST); s.save('work/ola_repair.png'); print(R.shape)
