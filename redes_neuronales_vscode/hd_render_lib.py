import pickle, numpy as np, time
from PIL import Image
import train_paint as T
r = pickle.load(open('work/run_paint.pkl', 'rb')); ks = sorted(r['snaps']); NK = len(ks)
DW, DH = 880, 550
def render(P, kind, w, h, box=(0, 0, 1, 1)):
    x0, y0, x1, y1 = box
    xs_ = x0 + (np.arange(w) + .5) / w * (x1 - x0); ys_ = y0 + (np.arange(h) + .5) / h * (y1 - y0)
    gx, gy = np.meshgrid(xs_, ys_); c = np.c_[gx.ravel(), gy.ravel() * T.TH / T.TW].astype(np.float32)
    out = np.empty((w * h, 3), np.float32)
    for i in range(0, len(c), 60000): out[i:i + 60000] = T.forward(P, T.featurize(kind, c[i:i + 60000]))[-1]
    return (np.clip(out.reshape(h, w, 3), 0, 1) * 255).astype(np.uint8)
