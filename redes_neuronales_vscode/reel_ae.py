"""Ep. 5b · Autoencoder · «¿Cabe un cuadro en 2 números?»  Uso: python3 reel_ae.py prev f… | chunk 85 | encode"""
import numpy as np, pickle
from PIL import Image, ImageDraw
from reel_common import *
import train_ae_cuadros as T

CCOL = [(110, 168, 255), (80, 214, 224), (236, 150, 70), (244, 116, 190), (190, 124, 255)]
NAMES = [p[0] for p in T.PAINT]
RUN = pickle.load(open("work/run_ae_cuadros.pkl", "rb")); KEYS = sorted(RUN["snaps"]); NK = len(KEYS); ITERS = RUN["ITERS"]
LD = {k: np.array(RUN["L"][k]) for k in (2, 32)}
PF = RUN["snaps"][KEYS[-1]]
SHOW = np.random.default_rng(3).choice(len(T.XTR), 1500, replace=False); XS, YS = T.XTR[SHOW], T.YTR[SHOW]
def patch(x, s, nearest=True): return Image.fromarray((np.clip(x, 0, 1).reshape(16, 16, 3) * 255).astype(np.uint8)).resize((int(s), int(s)), Image.NEAREST if nearest else Image.BICUBIC)

# vista suavizada del mapa latente durante el entrenamiento
TRAIN_N = 165
def kidx(rf): return int(round((min(rf, TRAIN_N - 25) / (TRAIN_N - 25)) ** 0.9 * (NK - 1)))
BOUNDS, cur = [], None
for rf in range(TRAIN_N):
    Z = T.encode(RUN["snaps"][KEYS[kidx(rf)]][2], XS); lo, hi = np.percentile(Z, 1, 0), np.percentile(Z, 99, 0)
    b = np.r_[(lo + hi) / 2, (hi - lo).max() / 2 * 1.12]; cur = b if cur is None else 0.85 * cur + 0.15 * b; BOUNDS.append(cur.copy())
ZF = T.encode(PF[2], XS); CEN = np.median(ZF, 0); MED = []
for c in range(5):
    zc = ZF[YS == c]; dvec = np.median(zc, 0) - CEN; dvec /= np.linalg.norm(dvec) + 1e-9
    pr = (zc - CEN) @ dvec; MED.append(np.median(zc[pr >= np.percentile(pr, 80)], 0))
ROUTE = [1, 0, 2, 3]
HOOK_X = T.XVA[3]; HOOK_Z = T.encode(PF[2], HOOK_X[None])[0]; HOOK_R = T.forward(PF[2], HOOK_X[None])[-1][0]

def scatter(img, box, Z, cx, cy, r, labels=False, alpha=170, rad=4):
    x0, y0, x1, y1 = box; d = ImageDraw.Draw(img, "RGBA"); d.rounded_rectangle(box, radius=22, fill=(14, 20, 30, 255), outline=(44, 58, 74, 255), width=2)
    s = (x1 - x0); to = lambda z: (x0 + (z[0] - cx + r) / (2 * r) * s, y0 + (cy + r - z[1]) / (2 * r) * (y1 - y0))
    lay = Image.new("RGBA", img.size, (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
    for z, c in zip(Z, YS):
        px, py = to(z)
        if x0 + 6 < px < x1 - 6 and y0 + 6 < py < y1 - 6: ld.ellipse((px - rad, py - rad, px + rad, py + rad), fill=CCOL[c] + (alpha,))
    img.alpha_composite(lay); d = ImageDraw.Draw(img, "RGBA")
    if labels:
        pos = sorted([(to(np.median(Z[YS == c], 0)), c) for c in range(5)], key=lambda t: t[0][1]); placed = []
        for (px, py), c in pos:
            for (qx, qy) in placed:
                if abs(px - qx) < 150 and abs(py - qy) < 38: py = qy + 38
            px = min(max(px, x0 + 80), x1 - 80); py = min(max(py, y0 + 24), y1 - 24); placed.append((px, py))
            d.text((px, py), NAMES[c], font=font(FS["monob"], 30), fill=CCOL[c], anchor="mm", stroke_width=5, stroke_fill=(8, 12, 18))
    return to

def s_hook(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["¿Cabe un cuadro", "en 2 números?"], 320, size=84)
    a1, a2, a3 = ease(rf / 8), ease((rf - 14) / 8), ease((rf - 28) / 8); s = 300
    paste_img(img, patch(HOOK_X, s), (W - s) / 2, 475, s, s, radius=16, alpha=a1); d = ImageDraw.Draw(img, "RGBA")
    small(d, "un trocito de Van Gogh · 768 números", 810, col=SOFT, size=28)
    if a2 > 0:
        d.text((W / 2, 865), "↓", font=font(FS["head"], 50), fill=GOLD + (int(255 * a2),), anchor="mm")
        d.text((W / 2, 940), f"({dec(HOOK_Z[0], 2)} ; {dec(HOOK_Z[1], 2)})", font=font(FS["head"], 72), fill=GOLD + (int(255 * a2),), anchor="mm")
        d.text((W / 2, 1015), "↓", font=font(FS["head"], 50), fill=GOLD + (int(255 * a2),), anchor="mm")
    if a3 > 0:
        paste_img(img, patch(HOOK_R, s), (W - s) / 2, 1050, s, s, radius=16, alpha=a3); d = ImageDraw.Draw(img, "RGBA")
        small(d, "lo que reconstruye con solo esos 2", 1385, col=SOFT, size=28)
    return img

BARS = [(768, "entra un trozo · 768 números"), (256, ""), (64, ""), (2, "2 números"), (64, ""), (256, ""), (768, "sale su reconstrucción · 768")]
def s_how(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["La red comprime", "y reconstruye"], 320, size=84)
    for i, (k, lab) in enumerate(BARS):
        a = ease((rf - i * 5) / 8)
        if a <= 0: continue
        w = 80 + 700 * (np.log(k) / np.log(768)); y = 560 + i * 105; col = GOLD if k == 2 else BLUE
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
        ld.rounded_rectangle((W / 2 - w / 2, y - 26, W / 2 + w / 2, y + 26), radius=26, fill=col + (230,))
        if lab: ld.text((W / 2, y + (0 if k == 2 else 0)), lab if k == 2 else "", font=font(FS["monob"], 28), fill=(8, 12, 18), anchor="mm")
        if lab and k != 2: ld.text((W / 2, y - 52 if i == 0 else y + 52), lab, font=font(FS["mono"], 28), fill=SOFT, anchor="mm")
        img.alpha_composite(faded(lay, a))
    d = ImageDraw.Draw(img, "RGBA")
    if rf > 45: body(d, ["Nadie le dice de qué cuadro", "es cada trozo"], 1340, col=GOLD, size=44)
    return img

def s_train(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["Los cuadros se", "ordenan solos"], 300, size=82)
    k = KEYS[kidx(rf)]; P = RUN["snaps"][k]; cx, cy, r = BOUNDS[rf]
    scatter(img, (210, 430, 870, 960), T.encode(P[2], XS), cx, cy, r, labels=rf > 60)
    d = ImageDraw.Draw(img, "RGBA"); small(d, f"cada punto, un trozo · posición = sus 2 números · iteración {fmt(k)}", 995, col=SOFT, size=26)
    chart(img, (110, 1040, 970, 1440), [dict(xs=LD[2][:, 0], ys=LD[2][:, 1], upto=k, col=BLUE, name="2 núm.", val_fmt=lambda v: dec(v, 4)),
                                        dict(xs=LD[32][:, 0], ys=LD[32][:, 1], upto=k, col=GOLD, name="32 núm.", val_fmt=lambda v: dec(v, 4))],
          "error al reconstruir", xmax=ITERS, ylim=(0.008, 0.12), yticks=[0.01, 0.03, 0.1], yfmt=lambda v: dec(v, 2))
    return img

SEL5 = [c * 200 + 3 for c in range(5)]
def s_compare(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["¿Qué guarda", "con 2 números?"], 320, size=84)
    X = T.XVA[SEL5]; rows = [("trozos que nunca vio", X, WHITE), ("con 2 números", T.forward(PF[2], X)[-1], BLUE), ("con 32 números", T.forward(PF[32], X)[-1], GOLD)]
    ts, step = 160, 176; x0 = (W - (5 * step - 16)) // 2
    for i, (lab, arr, col) in enumerate(rows):
        a = ease((rf - i * 12) / 10); y = 505 + i * 245
        if a <= 0: continue
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
        ld.text((x0, y), T_(lab), font=font(FS["monob"], 30), fill=col, anchor="lm")
        for j in range(5): paste_img(lay, patch(arr[j], ts), x0 + j * step, y + 30, ts, ts, radius=14)
        img.alpha_composite(faded(lay, a))
    d = ImageDraw.Draw(img, "RGBA")
    if rf > 40: body(d, ["Con 2 números guarda el color y la luz.", "Con 32, también algo de forma."], 1300, size=42)
    return img

SEG = 30
def s_walk(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["Paseo por", "el mapa"], 300, size=86)
    s = min(len(ROUTE) - 1.0001, max(0, (rf - 6) / SEG)); a, t = int(s), ease(s - int(s))
    z = MED[ROUTE[a]] + (MED[ROUTE[a + 1]] - MED[ROUTE[a]]) * t
    paste_img(img, patch(T.decode(PF[2], z[None])[0], 480), 300, 462, 480, 480, radius=18, outline=GOLD, owidth=5); d = ImageDraw.Draw(img, "RGBA")
    chip(d, W / 2, 965, f"de {NAMES[ROUTE[a]]} a {NAMES[ROUTE[a + 1]]}", col=WHITE, size=32, anchor="mm", bg=(30, 40, 54, 255))
    cx, cy, r = BOUNDS[-1]; to = scatter(img, (310, 1010, 770, 1430), ZF, cx, cy, r, alpha=120, rad=3); d = ImageDraw.Draw(img, "RGBA")
    d.line([to(MED[c]) for c in ROUTE], fill=WHITE + (150,), width=4, joint="curve")
    px, py = to(z); d.ellipse((px - 15, py - 15, px + 15, py + 15), outline=GOLD, width=5)
    return img

def s_end(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["¿Qué cuadros", "mezclo?"], 330)
    cx, cy, r = BOUNDS[-1]; nx, ny, t = 8, 5, 100; x0 = (W - nx * (t + 6)) / 2
    gr = np.array([[cx + a * r, cy - b * r] for b in np.linspace(-0.8, 0.8, ny) for a in np.linspace(-0.8, 0.8, nx)])
    for i, x in enumerate(T.decode(PF[2], gr)):
        rr_, cc = divmod(i, nx); paste_img(img, patch(x, t, False), x0 + cc * (t + 6), 480 + rr_ * (t + 6), t, t, radius=8)
    d = ImageDraw.Draw(img, "RGBA")
    body(d, ["Comenta dos obras de dominio público"], 1080)
    d.text((W / 2, 1170), T_("Redes Neuronales · Ep. 5"), font=font(FS["monob"], 32), fill=GOLD, anchor="mm")
    d.text((W / 2, 1260), HANDLE, font=font(FS["head"], 56), fill=WHITE, anchor="mm")
    return img

TL = Timeline([("hook", 75), ("how", 90), ("train", TRAIN_N), ("compare", 90), ("walk", 120), ("end", 75)],
              dict(hook=s_hook, how=s_how, train=s_train, compare=s_compare, walk=s_walk, end=s_end))

def make_audio():
    A = Audio(TL.N); A.pad()
    for nm in ("how", "train", "compare", "walk", "end"): A.at(TL.start(nm) - 6, A.whoosh(), 0.2)
    A.at(0, A.tone(55, 0.6, 9), 0.5)
    for k, f0 in enumerate((440, 554.4, 659.3)): A.at(k * 14, A.tone(f0, 0.5, 7), 0.12)
    for i in range(7): A.at(TL.start("how") + i * 5, A.tone(330 * 2 ** ((3 - abs(3 - i)) / 6), 0.3, 12), 0.09)
    w0 = TL.start("walk"); seg = np.arange(int(120 / FPS * A.sr)) / A.sr
    fr = 330 * 2 ** (np.clip((seg * FPS - 6) / SEG, 0, 3) / 3); A.at(w0, 0.06 * np.sin(2 * np.pi * np.cumsum(fr) / A.sr), 1.0)
    A.chime(TL.start("end")); return A

if __name__ == "__main__":
    cli(TL, "ep5b_reel_autoencoder", make_audio)
