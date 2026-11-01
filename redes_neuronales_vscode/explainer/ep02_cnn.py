"""Episodio 02 · Redes convolucionales.  Uso: python3 ep02_cnn.py info | prev f… | chunk 85 | encode"""
import os, sys, pickle, math
import numpy as np
from PIL import Image, ImageDraw
from explainer import *
from reel_common import T_
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_cuadros as TC

HERE = os.path.dirname(os.path.abspath(__file__))
tl = Timeline(os.path.join(HERE, "guion_ep02_cnn.yaml"))
CX = (PANEL[0] + PANEL[2]) / 2
CCOL = [(110, 168, 255), (80, 214, 224), (236, 150, 70), (244, 116, 190), (190, 124, 255)]
NAMES = [p[0] for p in TC.PAINT]

# ---------- datos reales (Ep. 2b) ----------
RGBR = pickle.load(open("work/run_cuadros_rgb.pkl", "rb")); GRYR = pickle.load(open("work/run_cuadros_gray.pkl", "rb"))
KEYS = sorted(RGBR["snaps"]); LOG = {it: (l, vl, a) for it, l, vl, a in RGBR["L"]}
LC = np.array([(it, l, vl) for it, l, vl, a in RGBR["L"]]); LC[:, 1] = np.convolve(np.pad(LC[:, 1], 2, mode="edge"), np.ones(5) / 5, "valid")
LG = np.array([(it, l, vl) for it, l, vl, a in GRYR["L"]])
ACC_C = LOG[KEYS[-1]][2]; ACC_G = [a for it, l, vl, a in GRYR["L"] if it == GRYR["ITERS"]][0]
to_rgb = lambda x: x - 0.5; to_gray = lambda x: x.mean(-1, keepdims=True) - 0.5
def arr_img(a, gray=False):
    a = a.mean(-1, keepdims=True).repeat(3, -1) if gray else a
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
PAINT_IMG = [arr_img(im) for im in TC.imgs]
HX, HC = TC.VAL[1]; _pr = TC.forward(RGBR["snaps"][KEYS[-1]], to_rgb(HX)).argmax(1)
HOOK_P = HX[max([i for i in range(len(HX)) if _pr[i] == 1], key=lambda i: HX[i].std())]
rm = np.random.default_rng(4); TILE = np.concatenate([c * 80 + rm.choice(80, 4, replace=False) for c in range(5)])[:18]; rm.shuffle(TILE)
TX, TY = TC.Xva[TILE], TC.yva[TILE]
PRED_C = {k: TC.forward(RGBR["snaps"][k], to_rgb(TX)).argmax(1) for k in KEYS}
PRED_G = TC.forward(GRYR["snaps"][GRYR["ITERS"]], to_gray(TX)).argmax(1)

# recorte en grises para explicar el filtro: la zona con más bordes de La gran ola
GW = TC.imgs[1].mean(-1); best, bxy = -1, (0, 0)
for y in range(0, 200, 6):
    for x in range(0, 300, 6):
        c = GW[y:y + 12, x:x + 12]; g = np.abs(np.diff(c, axis=1)).mean()
        if g > best: best, bxy = g, (x, y)
CROP = GW[bxy[1]:bxy[1] + 12, bxy[0]:bxy[0] + 12]
KER = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)       # filtro de ejemplo: bordes verticales
OUT = np.array([[np.sum(CROP[i:i + 3, j:j + 3] * KER) for j in range(10)] for i in range(10)])

def label(d, x, y, txt, col=SOFT, size=26, anchor="mm", bold=False):
    d.text((x, y), txt, font=font(FS["monob" if bold else "mono"], size), fill=col, anchor=anchor)
def hl(s, t, name, base=WHITE, win=1.2):
    return (GOLD if 0 <= t - s.cue(name) < win else base, s.a(t, name, 0.35))
def gray_cell(v): v = int(np.clip(v, 0, 1) * 255); return (v, v, v)

# ---------- escenas ----------
def hook(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); sz = 360
    paste_img(lay, arr_img(HOOK_P), CX - sz / 2, 530, sz, sz, radius=16, outline=GOLD, owidth=6, resample=Image.NEAREST); d = ImageDraw.Draw(lay, "RGBA")
    label(d, CX, 910, "32 × 32 píxeles · ~1 % del cuadro", SOFT, 24)
    ims = [p.resize((int(p.width * 86 / p.height), 86), Image.LANCZOS) for p in PAINT_IMG]; tot = sum(i.width for i in ims) + 14 * 4; x = CX - tot / 2
    for c, im in enumerate(ims):
        paste_img(lay, im, x, 950, im.width, 86, radius=8); d = ImageDraw.Draw(lay, "RGBA")
        label(d, x + im.width / 2, 1056, NAMES[c], CCOL[c], 20); x += im.width + 14

SMALL = np.asarray(PAINT_IMG[1].resize((24, 14), Image.LANCZOS)); PERM = np.random.default_rng(3).permutation(24 * 14)
def problem(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); a_v = s.a(t, "vecinos", 0.4); a_p = s.a(t, "pesos", 0.4)
    if a_v < 1:
        w, h = 400, 233; y0 = 600; k = 1 - a_v
        paste_img(lay, Image.fromarray(SMALL).resize((w, h), Image.NEAREST), 100, y0, w, h, radius=12, alpha=k)
        u = ease((t - s.cue("desordenas")) / 1.2); flat = SMALL.reshape(-1, 3); mix = flat.copy(); n_ = int(u * len(flat)); mix[:n_] = flat[PERM[:n_]]
        paste_img(lay, Image.fromarray(mix.reshape(14, 24, 3)).resize((w, h), Image.NEAREST), 580, y0, w, h, radius=12, alpha=k * s.a(t, "separado"))
        d = ImageDraw.Draw(lay, "RGBA"); label(d, 300, y0 + h + 30, "la imagen", SOFT + (int(255 * k),), 24); label(d, 780, y0 + h + 30, "desordenada: al MLP le da igual", SOFT + (int(255 * k),), 24)
    if a_v > 0 and a_p < 1:
        k = a_v * (1 - a_p); cs = 40; x0, y0 = CX - 6 * cs, 560
        lay2 = Image.new("RGBA", lay.size, (0, 0, 0, 0)); d2 = ImageDraw.Draw(lay2)
        for i in range(12):
            for j in range(12): d2.rectangle((x0 + j * cs, y0 + i * cs, x0 + (j + 1) * cs - 2, y0 + (i + 1) * cs - 2), fill=gray_cell(CROP[i, j]))
        ph = (t - s.cue("vecinos")) * 1.4; gi, gj = int(4 + 3 * math.sin(ph)), int(4 + 3 * math.cos(ph * 0.8))
        d2.rectangle((x0 + gj * cs - 3, y0 + gi * cs - 3, x0 + (gj + 3) * cs + 1, y0 + (gi + 3) * cs + 1), outline=GOLD, width=6)
        lay.alpha_composite(faded(lay2, k)); d = ImageDraw.Draw(lay, "RGBA")
        if s.a(t, "borde") > 0: label(d, CX, 1065, "un borde = cómo cambian los píxeles vecinos", GOLD + (int(255 * k * s.a(t, "borde")),), 26, bold=True)
    if a_p > 0:
        u = ease((t - s.cue("pesos")) / 1.0); n = int(3_072_000 * u)
        d.text((CX, 640), "UNA CAPA DENSA SOBRE 32×32×3", font=font(FS["monob"], 28), fill=SOFT + (int(255 * a_p),), anchor="mm")
        d.text((CX, 720), "3.072 entradas × 1.000 neuronas", font=font(FS["body"], 40), fill=WHITE + (int(255 * a_p),), anchor="mm")
        d.text((CX, 840), f"{fmt(n)}", font=font(FS["head"], 110), fill=RED + (int(255 * a_p),), anchor="mm")
        d.text((CX, 935), "pesos solo en la primera capa", font=font(FS["body"], 36), fill=SOFT + (int(255 * a_p),), anchor="mm")

F_CONV = Formula([("o", r"$(I * K)(i,j)$"), ("eq", "$=$"), ("s", r"$\sum_{m}\sum_{n}$"), ("i", r"$I(i+m,\,j+n)$"), ("k", r"$\cdot\,K(m,n)$")], px=58)
def filter_(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); cs = 30; x0, y0 = 100, 600; ox, oy = 620, 630
    for i in range(12):
        for j in range(12): d.rectangle((x0 + j * cs, y0 + i * cs, x0 + (j + 1) * cs - 2, y0 + (i + 1) * cs - 2), fill=gray_cell(CROP[i, j]))
    label(d, x0 + 6 * cs, y0 - 30, "imagen (grises)", SOFT, 22)
    a_f = s.a(t, "filtro")
    kx, ky = 470, 560
    if a_f > 0:
        for i in range(3):
            for j in range(3):
                v = KER[i, j]; col = GOLD if v > 0 else (BLUE if v < 0 else (60, 70, 82))
                d.rectangle((kx + j * 34, ky + i * 34, kx + (j + 1) * 34 - 3, ky + (i + 1) * 34 - 3), fill=col + (int(255 * a_f),))
                label(d, kx + j * 34 + 15, ky + i * 34 + 15, f"{int(v):+d}" if v else "0", (8, 12, 18, int(255 * a_f)), 18, bold=True)
        label(d, kx + 51, ky - 18, "filtro 3×3", GOLD + (int(255 * a_f),), 22, bold=True)
    pos = int(np.clip((t - s.cue("recorre")) * 7, 0, 99)) if s.a(t, "recorre") > 0 else -1
    if pos >= 0:
        pi, pj = divmod(pos, 10)
        d.rectangle((x0 + pj * cs - 3, y0 + pi * cs - 3, x0 + (pj + 3) * cs + 1, y0 + (pi + 3) * cs + 1), outline=GOLD, width=5)
        mx = np.abs(OUT).max() + 1e-9
        for q in range(pos + 1):
            qi, qj = divmod(q, 10); v = OUT[qi, qj] / mx
            col = tuple(int(18 + (c - 18) * min(1, abs(v)) ) for c in (GOLD if v > 0 else BLUE))
            d.rectangle((ox + qj * 34, oy + qi * 34, ox + (qj + 1) * 34 - 3, oy + (qi + 1) * 34 - 3), fill=col)
        d.rectangle((ox + pj * 34 - 2, oy + pi * 34 - 2, ox + (pj + 1) * 34, oy + (pi + 1) * 34), outline=WHITE, width=3)
        label(d, ox + 170, oy - 30, "mapa de activación", GOLD if s.a(t, "mapa") > 0 else SOFT, 22, bold=s.a(t, "mapa") > 0)
        if s.a(t, "suma") > 0: chip(d, ox + 170, oy + 365, f"Σ (píxeles × filtro) = {dec(OUT[pi, pj] if abs(OUT[pi, pj]) >= 0.005 else 0.0, 2)}", col=WHITE, size=24, anchor="mm", bg=(30, 40, 54, 255))
    st = {"o": hl(s, t, "mapa"), "eq": (WHITE, s.a(t, "multiplica")), "s": hl(s, t, "suma"), "i": hl(s, t, "multiplica"), "k": hl(s, t, "multiplica")}
    F_CONV.draw(lay, CX, 1060, st, max_w=900)
    label(d, CX, 1135, "filtro de ejemplo hecho a mano (bordes verticales); los de la red se aprenden", MUTED, 20)

def sharing(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); w, h = 560, 312; x0, y0 = CX - w / 2, 560
    paste_img(lay, PAINT_IMG[1].resize((w, h), Image.LANCZOS), x0, y0, w, h, radius=12); d = ImageDraw.Draw(lay, "RGBA")
    spots = [(0.12, 0.2), (0.45, 0.15), (0.75, 0.55), (0.3, 0.7), (0.6, 0.35)]
    for k, (u, v) in enumerate(spots):
        a = ease((t - s.cue("mismo") - k * 0.25) / 0.3)
        if a <= 0: continue
        bx, by = x0 + u * w, y0 + v * h; d.rectangle((bx, by, bx + 48, by + 48), outline=GOLD + (int(255 * a),), width=4)
        d.line((bx + 24, by + 48, CX, 920), fill=GOLD + (int(120 * a),), width=2)
    a_m = s.a(t, "mismo")
    d.rounded_rectangle((CX - 60, 900, CX + 60, 960), radius=10, fill=(40, 32, 14, int(255 * a_m)), outline=GOLD + (int(255 * a_m),), width=3)
    label(d, CX, 930, "1 filtro", GOLD + (int(255 * a_m),), 24, bold=True)
    if s.a(t, "millones") > 0:
        a = s.a(t, "millones"); d.text((290, 1050), "3.072.000", font=font(FS["head"], 60), fill=RED + (int(255 * a),), anchor="mm")
        label(d, 290, 1110, "capa densa", SOFT + (int(255 * a),), 24)
    if s.a(t, "nueve") > 0:
        a = s.a(t, "nueve"); d.text((790, 1050), "9", font=font(FS["head"], 80), fill=GREEN + (int(255 * a),), anchor="mm")
        label(d, 790, 1110, "un filtro 3×3 (27 en color)", SOFT + (int(255 * a),), 24)

def block(d, x, y, w, h, n, col, a, label_=None):
    for k in range(n - 1, -1, -1):
        o = k * 7; d.rectangle((x + o, y - o, x + w + o, y + h - o), fill=(20, 28, 38, int(255 * a)), outline=col + (int(255 * a),), width=2)
F_POOL = Formula([("y", r"$y$"), ("eq", "$=$"), ("m", r"$\max(x_{1},\,x_{2},\,x_{3},\,x_{4})$")], px=50)
def layers(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); a_a = s.a(t, "apila"); a_p = s.a(t, "pooling")
    stages = [(110, 160, 1, BLUE, "imagen\n32×32×3", 1.0), (290, 160, 4, GOLD, "conv + ReLU\n32×32×16", a_a), (480, 80, 4, GOLD, "pooling\n16×16×16", a_p),
              (620, 80, 6, GOLD, "conv + ReLU\n16×16×32", a_a), (770, 40, 6, GOLD, "pooling\n8×8×32", a_p), (900, 0, 0, GREEN, "densa\n5 cuadros", s.a(t, "decide"))]
    for (x, sz, n, col, lab, a) in stages:
        if a <= 0: continue
        cy = 720
        if sz: block(d, x - sz / 2, cy - sz / 2, sz, sz, n, col, a)
        else:
            for k in range(5): d.ellipse((x - 14, cy - 90 + k * 42, x + 14, cy - 62 + k * 42), fill=CCOL[k] + (int(255 * a),))
        for li, part in enumerate(lab.split("\n")): label(d, x + (n * 3 if sz else 0), 840 + li * 28, part, (WHITE if li == 0 else MUTED) + (int(255 * a),), 21 if li == 0 else 19, bold=li == 0)
    if s.a(t, "bordes") > 0: chip(d, 300, 580, "bordes", col=GOLD + (int(255 * s.a(t, "bordes")),), size=24, anchor="mm", bg=(40, 32, 14, int(255 * s.a(t, "bordes"))))
    if s.a(t, "combinaciones") > 0: chip(d, 640, 580, "combinaciones de bordes", col=GOLD + (int(255 * s.a(t, "combinaciones")),), size=24, anchor="mm", bg=(40, 32, 14, int(255 * s.a(t, "combinaciones"))))
    a_m = s.a(t, "maximo")
    if a_m > 0:
        vals = [[3, 1], [0, 7]]; x0, y0 = 300, 960
        for i in range(2):
            for j in range(2):
                hot = vals[i][j] == 7; d.rectangle((x0 + j * 58, y0 + i * 58, x0 + (j + 1) * 58 - 4, y0 + (i + 1) * 58 - 4), fill=((40, 32, 14) if hot else (26, 34, 44)) + (int(255 * a_m),), outline=(GOLD if hot else (70, 86, 104)) + (int(255 * a_m),), width=3)
                label(d, x0 + j * 58 + 27, y0 + i * 58 + 27, str(vals[i][j]), (GOLD if hot else WHITE) + (int(255 * a_m),), 30, bold=True)
        d.text((x0 + 150, y0 + 56), "→", font=font(FS["head"], 46), fill=GOLD + (int(255 * a_m),), anchor="mm")
        d.rectangle((x0 + 200, y0 + 27, x0 + 258, y0 + 85), fill=(40, 32, 14, int(255 * a_m)), outline=GOLD + (int(255 * a_m),), width=3); label(d, x0 + 229, y0 + 56, "7", GOLD + (int(255 * a_m),), 32, bold=True)
        F_POOL.draw(lay, 800, 1015, {"y": (WHITE, a_m), "eq": (WHITE, a_m), "m": hl(s, t, "maximo")})

CODE = """def conv2d(img, K):              # img: H×W, K: 3×3
    H, W = img.shape
    out = np.zeros((H - 2, W - 2))
    for i in range(H - 2):
        for j in range(W - 2):
            ventana = img[i:i+3, j:j+3]
            out[i, j] = np.sum(ventana * K)
    return out

def max_pool(x):                  # bloques 2×2
    H, W = x.shape[0] // 2, x.shape[1] // 2
    x = x[:2*H, :2*W].reshape(H, 2, W, 2)
    return x.max(axis=(1, 3))

# una capa: convolución + ReLU + pooling
mapa = max_pool(np.maximum(0, conv2d(img, K) + b))"""
CB = CodeBlock(CODE, size=26, lh=1.38)
GROUPS = [("bucles", range(0, 8)), ("pooling", range(9, 13)), ("capa", range(14, 16))]
def code(lay, s, t):
    act, foc = None, 0.0
    for name, rng in GROUPS:
        if t >= s.cue(name) - 0.1: act, foc = set(rng), ease((t - s.cue(GROUPS[0][0])) / 0.4)
    if act is not None and t >= s.cue("multiplicamos") - 0.1 and t < s.cue("pooling") - 0.1: act = {5, 6}
    CB.draw(lay, 104, 540, act, foc)
    d = ImageDraw.Draw(lay, "RGBA"); label(d, CX, 1150, "versión didáctica; en el entrenamiento real es la misma operación vectorizada", MUTED, 20)

def filt_img(Wc, f, size):
    w = Wc[:, f].reshape(3, 3, 3); w = (w - w.min()) / (w.max() - w.min() + 1e-9)
    return Image.fromarray((w * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
def learn(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    u = np.clip((t - s.cue("azar")) / max(1.0, s.cue("dieciseis") - s.cue("azar")), 0, 1) if s.a(t, "azar") > 0 else 0
    k = KEYS[int(round(u ** 0.7 * (len(KEYS) - 1)))]; Wc = RGBR["snaps"][k]["W1"]
    label(d, 330, 550, "los 16 filtros (3×3, en color)", SOFT, 22)
    for f in range(16):
        r_, c_ = divmod(f, 4); paste_img(lay, filt_img(Wc, f, 100), 110 + c_ * 112, 575 + r_ * 112, 100, 100, radius=8)
    d = ImageDraw.Draw(lay, "RGBA"); label(d, 775, 550, "lo que ve el trocito tras cada filtro", SOFT, 22)
    X = to_rgb(HOOK_P[None]); P = RGBR["snaps"][k]; Z = TC.im2col(X) @ P["W1"] + P["b1"]; A = np.maximum(Z, 0).reshape(32, 32, 16)
    for f in range(16):
        r_, c_ = divmod(f, 4); m = A[..., f]; m = m / (m.max() + 1e-9)
        im = Image.fromarray((np.stack([m * 244, m * 190, m * 78], -1)).astype(np.uint8)).resize((88, 88), Image.NEAREST)
        paste_img(lay, im, 600 + c_ * 98, 575 + r_ * 112, 88, 88, radius=6)
    d = ImageDraw.Draw(lay, "RGBA"); chip(d, CX, 1050, f"iteración {fmt(k)}", col=WHITE, size=26, anchor="mm", bg=(30, 40, 54, 255))
    if s.a(t, "retro") > 0: label(d, CX, 1115, "empiezan al azar · la retropropagación los ajusta", GOLD + (int(255 * s.a(t, "retro")),), 24, bold=True)

TS, TSTEP = 92, 102
def mosaic(lay, preds, gray=0.0, show=None):
    x0 = CX - (6 * TSTEP - 10) / 2; y0 = 540
    for j in range(18):
        r_, c_ = divmod(j, 6); x, y = x0 + c_ * TSTEP, y0 + r_ * TSTEP
        im = arr_img(TX[j], gray >= 0.5).resize((TS, TS), Image.NEAREST); paste_img(lay, im, x, y, TS, TS, radius=10)
        sh = 1.0 if show is None else show[j]
        if preds is not None and sh > 0:
            ok = preds[j] == TY[j]; ImageDraw.Draw(lay).rounded_rectangle((x - 3, y - 3, x + TS + 2, y + TS + 2), radius=12, outline=(GREEN if ok else RED), width=5)
def kidx_demo(s, t):
    u = np.clip((t - s.cue("ciegas")) / max(1.0, s.cue("acierta") + 1.5 - s.cue("ciegas")), 0, 1) if t >= s.cue("ciegas") else 0
    return KEYS[int(round(u ** 0.9 * (len(KEYS) - 1)))]
def demo(lay, s, t):
    k = kidx_demo(s, t); mosaic(lay, PRED_C[k] if s.a(t, "nunca") > 0 else None); d = ImageDraw.Draw(lay, "RGBA"); acc = LOG[k][2]
    d.text((CX, 880), T_(f"acierta el {dec(acc * 100)} %"), font=font(FS["head"], 56), fill=GOLD if acc > .9 else WHITE, anchor="mm")
def color(lay, s, t):
    gm = s.a(t, "quitamos", 0.4); sh = np.clip((t - s.cue("quitamos") - 0.6 - np.arange(18) * 0.08) / 0.3, 0, 1) if t > s.cue("quitamos") else np.zeros(18)
    mosaic(lay, PRED_G if gm >= 1 else PRED_C[KEYS[-1]], gray=gm, show=sh if gm >= 1 else None); d = ImageDraw.Draw(lay, "RGBA")
    p = ease((t - s.cue("baja")) / 1.0) if t > s.cue("baja") else 0; acc = ACC_C + (ACC_G - ACC_C) * p
    d.text((CX, 880), T_(f"acierta el {dec(acc * 100)} %"), font=font(FS["head"], 56), fill=RED if p >= 1 else WHITE, anchor="mm")

def when(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    rows = [("dominaron", "✓", GREEN, "la visión por ordenador, de 2012 a hoy"), ("eficiente", "✓", GREEN, "eficiente: pocos pesos, mucha imagen"),
            ("limite", "✗", RED, "mira vecindarios en el espacio…")]
    for i, (cue, sym, col, txt) in enumerate(rows):
        a = s.a(t, cue); y = 580 + i * 80
        if a <= 0: continue
        d.ellipse((96, y - 24, 144, y + 24), fill=col + (int(60 * a),), outline=col + (int(255 * a),), width=3)
        d.text((120, y), sym, font=font(FS["monob"], 30), fill=col + (int(255 * a),), anchor="mm"); d.text((166, y), txt, font=font(FS["body"], 34), fill=WHITE + (int(255 * a),), anchor="lm")
    a = s.a(t, "tiempo")
    if a > 0:
        words = ["el", "gato", "se", "sentó", "en", "la", "alfombra"]; f = font(FS["monob"], 30); x = 110
        for k, w in enumerate(words):
            aa = ease((t - s.cue("tiempo") - k * 0.12) / 0.3); tw = f.getlength(w) + 30
            d.rounded_rectangle((x, 900, x + tw, 956), radius=12, fill=(26, 34, 44, int(255 * aa)), outline=(70, 86, 104, int(255 * aa)), width=2)
            d.text((x + tw / 2, 928), w, font=f, fill=WHITE + (int(255 * aa),), anchor="mm"); x += tw + 12
        label(d, CX, 1010, "…pero no recuerda el orden en el tiempo", RED + (int(255 * a),), 28, bold=True)

def next_(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    d.text((CX, 600), "PRÓXIMO EPISODIO", font=font(FS["monob"], 30), fill=SOFT, anchor="mm")
    d.text((CX, 680), T_("Redes recurrentes"), font=font(FS["head"], 66), fill=GOLD, anchor="mm")
    words = ["el", "gato", "se", "sentó"]; f = font(FS["monob"], 34)
    for k, w in enumerate(words):
        x = 170 + k * 200; d.rounded_rectangle((x - 70, 900, x + 70, 960), radius=12, fill=(26, 34, 44), outline=(70, 86, 104), width=2)
        d.text((x, 930), w, font=f, fill=WHITE, anchor="mm")
        d.ellipse((x - 34, 770, x + 34, 838), outline=GOLD, width=4); label(d, x, 804, "h", GOLD, 28, bold=True); d.line((x, 840, x, 898), fill=MUTED, width=3)
        if k < 3:
            ph = ((t * 1.2) % 1); d.line((x + 36, 804, x + 164, 804), fill=GOLD, width=4); px = x + 36 + 128 * ph; d.ellipse((px - 7, 797, px + 7, 811), fill=GOLD)
    label(d, CX, 1040, "una memoria que pasa de palabra en palabra", SOFT, 28)
    a = s.a(t, "recurrentes"); d.line((262, 360, 336, 360), fill=GOLD + (int(255 * a),), width=4); d.polygon([(336, 352), (336, 368), (350, 360)], fill=GOLD + (int(255 * a),))

def end(tl_, t):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    d.text((W / 2, 720), "REDES NEURONALES", font=font(FS["head"], 84), fill=GOLD, anchor="mm")
    d.text((W / 2, 810), "Ep. 2 · Redes convolucionales", font=font(FS["body"], 46), fill=WHITE, anchor="mm")
    d.rounded_rectangle((200, 900, 880, 1140), radius=24, fill=(16, 22, 32, 255), outline=(44, 58, 74, 255), width=2)
    d.text((W / 2, 960), "PRÓXIMO: REDES RECURRENTES", font=font(FS["monob"], 28), fill=GOLD, anchor="mm")
    d.text((W / 2, 1020), "código completo en la descripción", font=font(FS["body"], 30), fill=SOFT, anchor="mm")
    d.text((W / 2, 1090), HANDLE, font=font(FS["head"], 52), fill=WHITE, anchor="mm")
    return img

def _lim(*ys):
    v = np.concatenate([np.asarray(y, float) for y in ys]); return max(1e-3, v.min() * 0.7), v.max() * 1.35

def curve_perdida(img, box, s, t):
    """El entrenamiento en color, revelándose entre «a ciegas» y «acierta»."""
    t0, t1 = s.cue("ciegas"), s.cue("acierta")
    u = np.clip((t - t0) / max(0.5, t1 - t0), 0, 1) ** 1.2
    up = LC[0, 0] + u * (LC[-1, 0] - LC[0, 0])
    lo, hi = _lim(LC[:, 1], LC[:, 2])
    chart(img, box, [dict(xs=LC[:, 0], ys=LC[:, 1], upto=up, col=BLUE, name="aprende", val_fmt=lambda v: dec(v, 3)),
                     dict(xs=LC[:, 0], ys=LC[:, 2], upto=up, col=GOLD, name="examen", val_fmt=lambda v: dec(v, 3))],
          "error durante el entrenamiento", xmax=LC[-1, 0], ylim=(lo, hi),
          yticks=[x for x in (0.01, 0.1, 1) if lo <= x <= hi] or [lo, hi], yfmt=lambda v: dec(v, 2))

def curve_color(img, box, s, t):
    """Color contra blanco y negro: la de gris entra en «baja»."""
    gris = 1e9 if t >= s.cue("baja") else 0
    lo, hi = _lim(LC[:, 2], LG[:, 2])
    chart(img, box, [dict(xs=LC[:, 0], ys=LC[:, 2], upto=1e9, col=GOLD, name="color", val_fmt=lambda v: dec(v, 3)),
                     dict(xs=LG[:, 0], ys=LG[:, 2], upto=gris, col=WHITE, name="blanco y negro", val_fmt=lambda v: dec(v, 3))],
          "error en el examen", xmax=max(LC[-1, 0], LG[-1, 0]), ylim=(lo, hi),
          yticks=[x for x in (0.01, 0.1, 1) if lo <= x <= hi] or [lo, hi], yfmt=lambda v: dec(v, 2))

SCENES = dict(_extras=dict(code=CODE, code_title="cnn.py", formulas={"conv": F_CONV, "pool": F_POOL},
                           curves={"perdida": curve_perdida, "color": curve_color}),
              hook=hook, problem=problem, filter=filter_, sharing=sharing, layers=layers, code=code, learn=learn, demo=demo, color=color, when=when, next=next_, _end=end)
if __name__ == "__main__":
    cli(tl, SCENES, "ep02_cnn_piloto")
