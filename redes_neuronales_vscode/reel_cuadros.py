"""Ep. 2b · CNN · «¿De qué cuadro es este trozo?»  Uso: python3 reel_cuadros.py prev f… | chunk 85 | encode"""
import numpy as np, pickle
from PIL import Image, ImageDraw
from reel_common import *
import reel_common as RC
import train_cuadros as T

CCOL = [(110, 168, 255), (80, 214, 224), (236, 150, 70), (244, 116, 190), (190, 124, 255)]
NAMES = [p[0] for p in T.PAINT]
RGBR = pickle.load(open("work/run_cuadros_rgb.pkl", "rb")); GRYR = pickle.load(open("work/run_cuadros_gray.pkl", "rb"))
KEYS = sorted(RGBR["snaps"]); NK = len(KEYS); LOG = {it: (l, vl, a) for it, l, vl, a in RGBR["L"]}
ACC_C = LOG[KEYS[-1]][2]; ACC_G = [a for it, l, vl, a in GRYR["L"] if it == GRYR["ITERS"]][0]
LC = np.array([(it, l, vl) for it, l, vl, a in RGBR["L"]]); LC[:, 1] = np.convolve(np.pad(LC[:, 1], 2, mode="edge"), np.ones(5) / 5, "valid"); LG = np.array([(it, l, vl) for it, l, vl, a in GRYR["L"]])
CH_BOX = (110, 1150, 970, 1470)
to_rgb = lambda x: x - 0.5
to_gray = lambda x: x.mean(-1, keepdims=True) - 0.5

# --- trozo del gancho: Hokusai, bien clasificado y con textura ---
HX, HC = T.VAL[1]; pr = T.forward(RGBR["snaps"][KEYS[-1]], to_rgb(HX)).argmax(1)
cand = [i for i in range(len(HX)) if pr[i] == 1]; HOOK_I = max(cand, key=lambda i: HX[i].std())
HOOK_P, (HOOK_X, HOOK_Y) = HX[HOOK_I], HC[HOOK_I]
# --- mosaico: 4 trozos de examen por cuadro ---
rm = np.random.default_rng(4); TILE = np.concatenate([c * 80 + rm.choice(80, 4, replace=False) for c in range(5)]); rm.shuffle(TILE)
TX, TY = T.Xva[TILE], T.yva[TILE]
PRED_C = np.stack([T.forward(RGBR["snaps"][k], to_rgb(TX)).argmax(1) for k in KEYS])
PRED_G = T.forward(GRYR["snaps"][GRYR["ITERS"]], to_gray(TX)).argmax(1)
def arr_img(a, gray=False):
    a = a.mean(-1, keepdims=True).repeat(3, -1) if gray else a
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
PAINT_IMG = [arr_img(im) for im in T.imgs]
TS, TSTEP = 124, 136; GX0 = (W - (5 * TS + 4 * 12)) // 2; GY0 = 432
TILE_C = [arr_img(x).resize((TS, TS), Image.NEAREST) for x in TX]; TILE_G = [arr_img(x, True).resize((TS, TS), Image.NEAREST) for x in TX]

# lienzo principal (La gran ola) para revelación y explicación
PW = 960; PH = int(PW * T.imgs[1].shape[0] / T.imgs[1].shape[1]); PX, PY = (W - PW) // 2, 520; SCL = PW / T.imgs[1].shape[1]
HOOK_BOX = (280, 520, 520)       # x, y, lado del trozo grande en el gancho

def thumbs_row(img, d, y, h, gray=False, split=False, names=True, highlight=None):
    ims = [p.resize((int(p.width * h / p.height), h), Image.LANCZOS) for p in PAINT_IMG]
    gap = 18; tot = sum(i.width for i in ims) + gap * 4; s = min(1.0, 900 / tot)
    ims = [i.resize((int(i.width * s), int(i.height * s)), Image.LANCZOS) for i in ims]; tot = sum(i.width for i in ims) + gap * 4
    x = (W - tot) / 2
    for c, im in enumerate(ims):
        paste_img(img, im.convert("L").convert("RGB") if gray else im, x, y, im.width, im.height, radius=10,
                  outline=(GOLD if highlight == c else None), owidth=5); d = ImageDraw.Draw(img, "RGBA")
        if split:
            cut = x + im.width * (1 - T.HOLD); d = shade(img, (cut, y, x + im.width, y + im.height), (8, 12, 18, 110))
            for yy_ in range(int(y), int(y + im.height), 10): d.line((cut, yy_, cut, yy_ + 5), fill=GOLD, width=3)
        if names: d.text((x + im.width / 2, y + im.height + 26), NAMES[c], font=font(FS["mono"], 24), fill=CCOL[c] if highlight is None else (GOLD if highlight == c else MUTED), anchor="mm")
        x += im.width + gap
    return ImageDraw.Draw(img, "RGBA")

def s_hook(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["¿De qué cuadro", "es este trozo?"], 330)
    x, y, s = HOOK_BOX; paste_img(img, arr_img(HOOK_P), x, y, s, s, radius=18, outline=GOLD, owidth=6, resample=Image.NEAREST); d = ImageDraw.Draw(img, "RGBA")
    small(d, "32×32 píxeles · ~1 % del cuadro", y + s + 34, col=SOFT, size=28)
    d = thumbs_row(img, d, 1130, 120)
    body(d, ["¿Lo adivinas antes que la red?"], 1360, col=GOLD)
    return img

def s_reveal(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    a = ease(rf / 12); paint = PAINT_IMG[1].resize((PW, PH), Image.LANCZOS)
    paste_img(img, paint, PX, PY, PW, PH, radius=16, alpha=a); d = ImageDraw.Draw(img, "RGBA")
    t = ease(rf / 24); x0, y0, s0 = HOOK_BOX; x1, y1, s1 = PX + HOOK_X * SCL, PY + HOOK_Y * SCL, T.PS * SCL
    x, y, s = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, s0 + (s1 - s0) * t
    if t < 1: paste_img(img, arr_img(HOOK_P), x, y, s, s, radius=int(18 * (1 - t)) + 2, resample=Image.NEAREST); d = ImageDraw.Draw(img, "RGBA")
    pulse = 6 + 3 * np.sin(rf / 3); d.rounded_rectangle((x - pulse, y - pulse, x + s + pulse, y + s + pulse), radius=6, outline=GOLD, width=5)
    headline(d, ["Es La gran ola", "de Hokusai"], 330)
    if rf > 26:
        a2 = ease((rf - 26) / 10); lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
        body(ld, ["¿Puede una red neuronal aprender", "a acertarlo con un trocito así?"], 1170)
        small(ld, "red convolucional (CNN) · 5 cuadros posibles", 1300, col=MUTED, size=28)
        img.alpha_composite(faded(lay, a2))
    return img

def s_how(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["Aprende con la izquierda,", "se examina con la derecha"], 330, size=74)
    paste_img(img, PAINT_IMG[1].resize((PW, PH), Image.LANCZOS), PX, PY, PW, PH, radius=16); d = ImageDraw.Draw(img, "RGBA")
    cut = PX + PW * (1 - T.HOLD)
    d = shade(img, (cut, PY, PX + PW, PY + PH), (8, 12, 18, 120))
    for yy_ in range(PY, PY + PH, 14): d.line((cut, yy_, cut, yy_ + 7), fill=GOLD, width=4)
    r = np.random.default_rng(rf // 3); s = T.PS * SCL
    for _ in range(7):
        x = PX + r.uniform(0, (cut - PX) - s); y = PY + r.uniform(0, PH - s); d.rectangle((x, y, x + s, y + s), outline=WHITE + (220,), width=3)
    chip(d, PX + 20, PY + 20, "APRENDE", col=WHITE, size=30); chip(d, PX + PW - 20, PY + 20, "EXAMEN", col=GOLD, size=30, anchor="rt")
    if rf > 30:
        a2 = ease((rf - 30) / 10); lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
        ld = thumbs_row(lay, ld, 1120, 105, split=True)
        body(ld, ["Igual con los 5 cuadros: el examen", "son trozos que nunca ha visto"], 1340, size=40)
        img.alpha_composite(faded(lay, a2))
    return img

def grid(img, d, preds, gray_mix=0.0, show=None):
    for j in range(20):
        r_, c_ = divmod(j, 5); x = GX0 + c_ * TSTEP; y = GY0 + r_ * TSTEP
        t = TILE_C[j] if gray_mix <= 0 else (TILE_G[j] if gray_mix >= 1 else Image.blend(TILE_C[j], TILE_G[j], gray_mix))
        paste_img(img, t, x, y, TS, TS, radius=14); d = ImageDraw.Draw(img, "RGBA")
        sh = 1.0 if show is None else show[j]
        if preds is not None and sh > 0:
            ok = preds[j] == TY[j]; col = (GREEN if ok else RED) + (int(255 * sh),)
            d.rounded_rectangle((x - 3, y - 3, x + TS + 2, y + TS + 2), radius=16, outline=col, width=6)
    return d

def legend(d, y):
    f = font(FS["mono"], 26); items = [(c, NAMES[c]) for c in range(5)]; tot = sum(f.getlength(nm) + 40 for _, nm in items) + 20 * 4; x = (W - tot) / 2
    for c, nm in items:
        d.ellipse((x, y - 11, x + 22, y + 11), fill=CCOL[c]); d.text((x + 32, y), nm, font=f, fill=SOFT, anchor="lm"); x += f.getlength(nm) + 40 + 20
    small(d, "borde verde = acierta · rojo = falla · punto = cuadro que cree", y + 44, size=24)

def kidx(rf, n): return int(round(min(1.0, rf / (n - 30)) ** 0.9 * (NK - 1)))
def s_train(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["20 trozos que", "nunca ha visto"], 290, size=74)
    k = kidx(rf, n); d = grid(img, d, PRED_C[k])
    acc = LOG[KEYS[k]][2]
    d.text((W / 2, 1035), T_(f"acierta {dec(acc * 100)} %"), font=font(FS["head"], 84), fill=GOLD if acc > .9 else WHITE, anchor="mm")
    small(d, f"de 400 trozos nunca vistos · iteración {fmt(KEYS[k])}", 1102, col=SOFT, size=30)
    chart(img, CH_BOX, [dict(xs=LC[:, 0], ys=LC[:, 1], upto=KEYS[k], col=BLUE, name="aprende"),
                        dict(xs=LC[:, 0], ys=LC[:, 2], upto=KEYS[k], col=GOLD, name="examen")],
          "error de la red", xmax=RGBR["ITERS"], ylim=(0.005, 3), yticks=[0.01, 0.1, 1])
    return img

def s_twist(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["¿Y si le quitamos", "el color?"], 290, size=78)
    gm = ease(rf / 14); show = np.clip((rf - 26 - np.arange(20) * 1.6) / 5, 0, 1) if rf >= 26 else np.zeros(20)
    d = grid(img, d, PRED_G if rf >= 26 else None, gm, show)
    p = ease((rf - 30) / 40); acc = ACC_C + (ACC_G - ACC_C) * p
    col = WHITE if p < 1 else RED
    d.text((W / 2, 1035), T_(f"acierta {dec(acc * 100)} %"), font=font(FS["head"], 84), fill=col, anchor="mm")
    if rf > 70:
        a = ease((rf - 70) / 10); lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
        body(ld, ["El color le chivaba parte de la respuesta"], 1102, col=GOLD, size=36)
        img.alpha_composite(faded(lay, a))
    up = 10 ** (np.log10(GRYR["ITERS"]) * ease((rf - 20) / 60)) if rf > 20 else 0
    chart(img, CH_BOX, [dict(xs=LC[:, 0], ys=LC[:, 2], upto=1e9, col=GOLD, name="color"),
                        dict(xs=LG[:, 0], ys=LG[:, 2], upto=up, col=WHITE, name="B/N")],
          "error de examen", xmax=GRYR["ITERS"], ylim=(0.2, 3), yticks=[0.3, 1, 2], xlabel="iteración · B/N = red gemela sin color")
    return img

def s_end(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["¿Qué cuadro", "le pongo?"], 330)
    rows = [[0, 1], [2, 3, 4]]; hs = [250, 215]; y = 520
    for row, h in zip(rows, hs):
        ims = [PAINT_IMG[c].resize((int(PAINT_IMG[c].width * h / PAINT_IMG[c].height), h), Image.LANCZOS) for c in row]
        tot = sum(i.width for i in ims) + 20 * (len(ims) - 1); x = (W - tot) / 2
        for im in ims: paste_img(img, im, x, y, im.width, im.height, radius=14); x += im.width + 20
        y += h + 20
    d = ImageDraw.Draw(img, "RGBA")
    body(d, ["Comenta una obra de dominio público"], 1110)
    d.text((W / 2, 1200), T_("Redes Neuronales · Ep. 2"), font=font(FS["monob"], 32), fill=GOLD, anchor="mm")
    d.text((W / 2, 1290), HANDLE, font=font(FS["head"], 56), fill=WHITE, anchor="mm")
    return img

TL = Timeline([("hook", 75), ("reveal", 90), ("how", 90), ("train", 165), ("twist", 120), ("end", 75)],
              dict(hook=s_hook, reveal=s_reveal, how=s_how, train=s_train, twist=s_twist, end=s_end))

def make_audio():
    A = Audio(TL.N); A.pad()
    for nm in ("reveal", "how", "train", "twist", "end"): A.at(TL.start(nm) - 6, A.whoosh(), 0.22)
    A.at(0, A.tone(55, 0.6, 9), 0.5)
    for k in range(3): A.at(15 + k * 18, A.tone(880, 0.18, 30), 0.12)                 # tic-tac del reto
    A.at(TL.start("reveal") + 24, A.tone(1046.5, 0.8, 5) + 0.6 * A.tone(1568, 0.8, 5), 0.14)   # aterriza el trozo
    s0 = TL.start("train"); prev = None
    for rf in range(0, 165, 2):
        ok = int((PRED_C[kidx(rf, 165)] == TY).sum())
        if prev is not None and ok > prev: A.at(s0 + rf, A.tone(330 + 22 * ok, 0.25, 16), 0.07)
        prev = ok
    t0 = TL.start("twist"); tt = np.arange(int(1.2 * A.sr)) / A.sr
    A.at(t0, np.exp(-tt * 2) * np.sin(2 * np.pi * (440 - 160 * tt) * tt), 0.18)
    for j in range(20):
        if PRED_G[j] != TY[j]: A.at(t0 + 26 + j * 1.6, A.tone(196, 0.25, 18), 0.09)
    A.chime(TL.start("end")); return A

if __name__ == "__main__":
    cli(TL, "ep2b_reel_cuadros", make_audio)
