import pickle, numpy as np
from PIL import Image
import train_paint as T
r = pickle.load(open('work/run_paint.pkl', 'rb')); ks = sorted(r['snaps'])
DW, DH = 440, 275
def render(P, kind, w, h, box=(0, 0, 1, 1)):
    x0, y0, x1, y1 = box
    xs_ = x0 + (np.arange(w) + .5) / w * (x1 - x0); ys_ = y0 + (np.arange(h) + .5) / h * (y1 - y0)
    gx, gy = np.meshgrid(xs_, ys_); c = np.c_[gx.ravel(), gy.ravel() * T.TH / T.TW].astype(np.float32)
    return (np.clip(T.forward(P, T.featurize(kind, c))[-1].reshape(h, w, 3), 0, 1) * 255).astype(np.uint8)
out = {k: np.stack([render(r['snaps'][s][k], k, DW, DH) for s in ks]) for k in ('plain', 'fourier')}
np.save('work/paint_plain.npy', out['plain']); np.save('work/paint_fourier.npy', out['fourier'])
BOX = (0.36, 0.14, 0.60, 0.38); fin = r['snaps'][ks[-1]]
np.save('work/paint_zoom_net.npy', render(fin['fourier'], 'fourier', DW, DH, BOX))
np.save('work/paint_zoom_plain.npy', render(fin['plain'], 'plain', DW, DH, BOX))
full = Image.fromarray((T.img * 255).astype(np.uint8))
cw, ch = int(T.TW * (BOX[2] - BOX[0])), int(T.TH * (BOX[3] - BOX[1]))
crop = full.crop((int(T.TW * BOX[0]), int(T.TH * BOX[1]), int(T.TW * BOX[0]) + cw, int(T.TH * BOX[1]) + ch))
np.save('work/paint_zoom_pix.npy', np.asarray(crop.resize((DW, DH), Image.NEAREST)))
np.save('work/paint_orig.npy', np.asarray(Image.open(T.SRC).convert('RGB').resize((DW, DH), Image.LANCZOS)))
n_f = sum(w.size + b.size for w, b in fin['fourier']); n_p = sum(w.size + b.size for w, b in fin['plain'])
pickle.dump(dict(keys=ks, ps=r['ps'], BOX=BOX, crop_px=(cw, ch), n_fourier=int(n_f), n_plain=int(n_p), ITERS=r['ITERS'], M=r['M'], npix=T.TH * T.TW, TH=T.TH, TW=T.TW), open('work/paint_meta.pkl', 'wb'))
print('listo', len(ks), cw, ch, n_f, n_p)
