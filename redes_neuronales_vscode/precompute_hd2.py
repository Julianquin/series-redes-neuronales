"""Ep. 1b (Reel): pre-renderiza en alta resolución las dos redes (normal y Fourier) + imagen grande y recortes del zoom.
Reanudable: python3 precompute_hd2.py [SEGUNDOS]   (sin argumento = hasta terminar). Requiere work/run_paint.pkl."""
import pickle, numpy as np, os, sys, time
from PIL import Image
from hd_render_lib import render, r, ks, NK
import train_paint as T
os.makedirs('work/hd', exist_ok=True)
IDX = np.unique(np.round(np.linspace(0, 1, 90) ** 0.85 * (NK - 1)).astype(int)); BOX = (0.36, 0.14, 0.60, 0.38)
if not os.path.exists('work/hd_meta.pkl'):
    pickle.dump(dict(IDX=IDX.tolist(), keys=ks, ps=r['ps'], BOX=BOX, ITERS=r['ITERS'], M=r['M']), open('work/hd_meta.pkl', 'wb'))
fin = r['snaps'][ks[-1]]
if not os.path.exists('work/hd_hero.npy'): np.save('work/hd_hero.npy', render(fin['fourier'], 'fourier', 1080, 675))
if not os.path.exists('work/hd_zoom_net.npy'): np.save('work/hd_zoom_net.npy', render(fin['fourier'], 'fourier', 880, 550, BOX))
if not os.path.exists('work/hd_zoom_pix.npy'):
    full = Image.fromarray((T.img * 255).astype(np.uint8)); cw, ch = int(T.TW * (BOX[2] - BOX[0])), int(T.TH * (BOX[3] - BOX[1]))
    crop = full.crop((int(T.TW * BOX[0]), int(T.TH * BOX[1]), int(T.TW * BOX[0]) + cw, int(T.TH * BOX[1]) + ch))
    np.save('work/hd_zoom_pix.npy', np.asarray(crop.resize((880, 550), Image.NEAREST)))
t0 = time.time(); budget = float(sys.argv[1]) if len(sys.argv) > 1 else 1e12
for j, i in enumerate(IDX):
    for kind, (w, h) in (('fourier', (800, 500)), ('plain', (400, 250))):
        p = f'work/hd/{kind}_{j:03d}.npy'
        if os.path.exists(p): continue
        if time.time() - t0 > budget: print('pausa', j, kind); sys.exit(0)
        np.save(p, render(r['snaps'][ks[i]][kind], kind, w, h))
print('completo', len(IDX))
