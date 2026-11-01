import numpy as np, pickle, subprocess, wave
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import render_mlp as R
import train_cuadros as T

W, H, FPS = 1080, 1920, 30
S1, S2 = 330, 120                  # entrenamiento en color / giro sin color
HOLD_F, END_F = 20, 66
N = S1 + S2 + HOLD_F + END_F
GOLD, BLUE, WHITE, GREY, EDGE, PANEL = R.GOLD, R.BLUE, R.WHITE, R.GREY, R.EDGE, R.PANEL
GREEN, REDC = (96, 206, 140), (236, 96, 96)
CCOL = [(92, 156, 255), (80, 210, 220), (230, 150, 70), (240, 110, 190), (190, 120, 255)]
HANDLE = R.HANDLE
RGB = pickle.load(open("work/run_cuadros_rgb.pkl", "rb")); GRY = pickle.load(open("work/run_cuadros_gray.pkl", "rb"))
KEYS = sorted(RGB["snaps"]); NK = len(KEYS); ITERS = RGB["ITERS"]
LOG = {it: (l, vl, a) for it, l, vl, a in RGB["L"]}
gray_acc = [a for it, l, vl, a in GRY["L"] if it == GRY["ITERS"]][0]
fmt = lambda n: f"{n:,}".replace(",", ".")
dec = lambda v, k=1: f"{v:.{k}f}".replace(".", ",")
ease = lambda t: 0 if t <= 0 else 1 if t >= 1 else t * t * (3 - 2 * t)
_fc = {}
def sg(size, w=700):
    if (size, w) not in _fc: _fc[(size, w)] = R.font(R.SG, size, w)
    return _fc[(size, w)]
fS = R.font(R.JB, 22, 400)

# ---- mosaico: 8 trozos de examen por cuadro ----
rm = np.random.default_rng(4)
TILE_IDX = np.concatenate([c * 80 + rm.choice(80, 8, replace=False) for c in range(5)]); rm.shuffle(TILE_IDX)
TX = T.Xva[TILE_IDX]; TY = T.yva[TILE_IDX]
to_rgb = lambda x: x - 0.5
to_gray = lambda x: x.mean(-1, keepdims=True) - 0.5
PRED_RGB = np.stack([T.forward(RGB["snaps"][k], to_rgb(TX)).argmax(1) for k in KEYS])
PRED_GRAY = T.forward(GRY["snaps"][GRY["ITERS"]], to_gray(TX)).argmax(1)
TS = 98
def tile_img(x, gray):
    a = (x.mean(-1, keepdims=True).repeat(3, -1) if gray else x)
    return Image.fromarray((a * 255).astype(np.uint8)).resize((TS, TS), Image.BICUBIC)
TILES_C = [tile_img(x, False) for x in TX]; TILES_G = [tile_img(x, True) for x in TX]
PATCH_PCT = np.mean([32 * 32 / (im.shape[0] * im.shape[1]) for im in T.imgs]) * 100

# ---- miniaturas con la franja de examen ----
TH_ = 112
def thumbs(gray):
    out = []
    for c, im in enumerate(T.imgs):
        a = im.mean(-1, keepdims=True).repeat(3, -1) if gray else im
        t = Image.fromarray((a * 255).astype(np.uint8)); t = t.resize((int(t.width * TH_ / t.height), TH_), Image.LANCZOS).convert("RGBA")
        d_ = ImageDraw.Draw(t, "RGBA"); cut = int(t.width * (1 - T.HOLD))
        d_.rectangle((cut, 0, t.width, TH_), fill=(8, 12, 16, 120))
        for yy in range(0, TH_, 10): d_.line((cut, yy, cut, yy + 5), fill=GOLD + (255,), width=2)
        out.append(t)
    return out
TH_C, TH_G = thumbs(False), thumbs(True)

def header(d_):
    d_.text((W / 2, 112), f"REDES NEURONALES · EP. 2 · {HANDLE}", font=R.fMonoS, fill=(70, 84, 98), anchor="mm")
    d_.text((W / 2, 205), "¿puede una red neuronal", font=R.fTitle, fill=GOLD, anchor="mm")
    pill = "reconocer un cuadro por un trocito"
    f_ = R.fPill
    tb = d_.textbbox((0, 0), pill, font=f_); tw = tb[2] - tb[0]
    if tw > 820: f_ = R.font(R.JB, 42, 600); tb = d_.textbbox((0, 0), pill, font=f_); tw = tb[2] - tb[0]
    x0 = W / 2 - (tw + 60) / 2 - 22; y0 = 262
    d_.rounded_rectangle((x0, y0, x0 + tw + 60, y0 + 86), radius=12, fill=(22, 30, 40), outline=(62, 78, 96), width=2)
    d_.text((x0 + 30 + tw / 2, y0 + 43), pill, font=f_, fill=WHITE, anchor="mm")
    d_.text((x0 + tw + 60 + 34, y0 + 42), "?", font=R.fQ, fill=GOLD, anchor="mm")

def draw_thumbs(img, d_, gray_mix):
    ws = [t.width for t in TH_C]; gap = (880 - sum(ws)) / 4; x = 100
    for c in range(5):
        t = TH_C[c] if gray_mix <= 0 else Image.blend(TH_C[c], TH_G[c], gray_mix)
        img.alpha_composite(t, (int(x), 462)); d_ = ImageDraw.Draw(img, "RGBA")
        d_.rectangle((x, 580, x + ws[c], 586), fill=CCOL[c])
        d_.text((x + ws[c] / 2, 608), T.PAINT[c][0], font=fS, fill=WHITE, anchor="mm")
        x += ws[c] + gap
    return ImageDraw.Draw(img, "RGBA")

MX0, MY0, STEP = 100, 668, 110
def draw_mosaic(img, d_, preds, gray_mix, show):
    for j in range(40):
        r, c = divmod(j, 8); x = MX0 + c * STEP; y = MY0 + r * STEP
        t = TILES_C[j] if gray_mix <= 0 else (TILES_G[j] if gray_mix >= 1 else Image.blend(TILES_C[j], TILES_G[j], gray_mix))
        img.paste(t, (x + 6, y + 6)); d_ = ImageDraw.Draw(img, "RGBA")
        if show[j] > 0 and preds is not None:
            ok = preds[j] == TY[j]; col = (GREEN if ok else REDC) + (int(255 * show[j]),)
            d_.rectangle((x + 2, y + 2, x + TS + 9, y + TS + 9), outline=col, width=4)
            d_.ellipse((x + TS - 20, y + TS - 20, x + TS + 6, y + TS + 6), fill=CCOL[preds[j]] + (int(255 * show[j]),), outline=(8, 12, 16), width=2)
        else:
            d_.rectangle((x + 4, y + 4, x + TS + 7, y + TS + 7), outline=EDGE, width=2)
    return d_

LX0, LX1, LY0, LY1 = 150, 990, 1560, 1765
def lpx(it, v): return LX0 + np.log10(max(it, 1)) / np.log10(ITERS) * (LX1 - LX0), LY1 - (np.log10(max(v, 1e-3)) + 2) / (np.log10(3) + 2) * (LY1 - LY0)
def chart(d_, idx):
    R.panel(d_, (60, 1440, 1020, 1840))
    d_.text((90, 1478), "error (escala log)", font=R.fMono, fill=GREY, anchor="lm")
    it = KEYS[idx]; l, vl, a = LOG[it]
    d_.line((520, 1478, 550, 1478), fill=BLUE, width=4); d_.text((560, 1478), f"aprende {dec(l, 3)}", font=R.fMonoS, fill=WHITE, anchor="lm")
    d_.line((780, 1478, 810, 1478), fill=GOLD, width=4); d_.text((820, 1478), f"examen {dec(vl, 3)}", font=R.fMonoS, fill=WHITE, anchor="lm")
    for v, lab in ((1, "1"), (0.1, "0,1"), (0.01, "0,01")):
        _, yy = lpx(1, v); d_.line((LX0, yy, LX1, yy), fill=(26, 36, 46), width=1); d_.text((LX0 - 14, yy), lab, font=R.fMonoS, fill=GREY, anchor="rm")
    d_.line((LX0, LY1, LX1, LY1), fill=EDGE, width=2)
    for e in (1, 10, 100, ITERS):
        x, _ = lpx(e, 1); d_.text((min(x, LX1 - 16), LY1 + 26), fmt(e), font=R.fMonoS, fill=GREY, anchor="mm")
    d_.text(((LX0 + LX1) / 2, LY1 + 56), "iteración (escala log)", font=R.fMonoS, fill=GREY, anchor="mm")
    for j, col in ((2, GOLD), (1, BLUE)):
        pts = [lpx(k, LOG[k][j - 1]) for k in KEYS[:idx + 1]]
        if len(pts) > 1: d_.line(pts, fill=col, width=3, joint="curve")
        d_.ellipse((pts[-1][0] - 6, pts[-1][1] - 6, pts[-1][0] + 6, pts[-1][1] + 6), fill=col)

def bars(d_, p):
    R.panel(d_, (60, 1440, 1020, 1840))
    d_.text((90, 1478), "acierto con trozos nunca vistos", font=R.fMono, fill=GREY, anchor="lm")
    acc_c = LOG[KEYS[-1]][2]
    rows = (("con color", acc_c, GOLD), ("blanco y negro", gray_acc * ease(p), WHITE))
    for i, (lab, v, col) in enumerate(rows):
        y = 1550 + i * 110
        d_.text((90, y), lab, font=sg(34, 600), fill=col, anchor="lm")
        if v > 0.005:
            d_.rounded_rectangle((90, y + 30, 90 + 840 * v, y + 70), radius=8, fill=col)
            d_.text((90 + 840 * v + 14, y + 50), f"{dec(v * 100)} %", font=sg(34), fill=col, anchor="lm")
        else:
            d_.text((100, y + 50), "?", font=sg(40), fill=col, anchor="lm")
    xz = 90 + 840 * 0.2
    for yy in range(1570, 1780, 12): d_.line((xz, yy, xz, yy + 6), fill=(110, 124, 138), width=2)
    d_.text((xz, 1800), "azar: 20 %", font=fS, fill=GREY, anchor="mm")

def scene(f):
    img = R.BGIMG.copy(); d_ = ImageDraw.Draw(img, "RGBA"); header(d_)
    R.panel(d_, (60, 380, 1020, 1410))
    if f < S1:
        idx = int(round((f / (S1 - 1)) ** 0.9 * (NK - 1))); gm = 0
        d_.text((92, 425), f"iter {fmt(KEYS[idx])} / {fmt(ITERS)}", font=R.fMono, fill=GREY, anchor="lm")
    else:
        idx = NK - 1; zf = f - S1; gm = ease(zf / 18)
        d_.text((92, 425), "red idéntica, en B/N", font=R.fMono, fill=GOLD, anchor="lm")
    d_.text((988, 425), "Conv(16) · Conv(32) · Dense(5)", font=R.fMono, fill=GREY, anchor="rm")
    d_ = draw_thumbs(img, d_, gm)
    d_.text((540, 642), "aprende con la izquierda · examen con la franja marcada", font=fS, fill=GREY, anchor="mm")
    if f < S1:
        d_ = draw_mosaic(img, d_, PRED_RGB[idx], 0, np.ones(40))
        acc = LOG[KEYS[idx]][2]
        d_.text((540, 1270), f"acierta {dec(acc * 100)} %", font=sg(64), fill=GOLD if acc > .9 else WHITE, anchor="mm")
        d_.text((540, 1336), f"de 400 trozos que nunca vio · cada uno es ~{dec(PATCH_PCT)} % del cuadro", font=fS, fill=GREY, anchor="mm")
        chart(d_, idx)
    else:
        zf = f - S1
        show = np.clip((zf - 22 - np.arange(40) * 0.9) / 5, 0, 1) if zf >= 22 else np.zeros(40)
        d_ = draw_mosaic(img, d_, PRED_GRAY if zf >= 22 else None, gm, show)
        if zf < 40:
            d_.text((540, 1300), "¿y si le quitamos el color?", font=sg(56), fill=GOLD, anchor="mm")
        else:
            a = ease((zf - 40) / 10)
            lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
            ld.text((540, 1270), f"baja al {dec(gray_acc * 100)} %", font=sg(64), fill=WHITE, anchor="mm")
            ld.text((540, 1336), "el color le chivaba parte de la respuesta", font=sg(32, 600), fill=GOLD, anchor="mm")
            lay.putalpha(lay.getchannel("A").point(lambda v: int(v * a))); img.alpha_composite(lay); d_ = ImageDraw.Draw(img, "RGBA")
        bars(d_, (zf - 30) / 40)
    return img

fSub = R.font(R.SG, 44, 500); fFollow = R.font(R.JB, 26, 600); fHandle = R.font(R.SG, 50, 700); fBrand = R.font(R.SG, 96, 700)
def end_card():
    img = R.BGIMG.copy()
    gl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(gl)
    xs = [180, 360, 540, 720, 900]
    for x, c in zip(xs, CCOL): gd.rounded_rectangle((x - 70, 630, x + 70, 770), radius=10, fill=c + (120,))
    img.alpha_composite(gl.filter(ImageFilter.GaussianBlur(14))); d_ = ImageDraw.Draw(img, "RGBA")
    for j, (x, c) in enumerate(zip(xs, CCOL)):
        t = TILES_C[list(TY).index(j)].resize((136, 136), Image.BICUBIC); img.paste(t, (x - 68, 632)); d_ = ImageDraw.Draw(img, "RGBA")
        d_.rectangle((x - 70, 630, x + 70, 770), outline=c, width=4)
    tl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ImageDraw.Draw(tl).text((W / 2, 940), "Redes Neuronales", font=fBrand, fill=GOLD, anchor="mm")
    img.alpha_composite(tl.filter(ImageFilter.GaussianBlur(10))); img.alpha_composite(tl); d_ = ImageDraw.Draw(img, "RGBA")
    d_.text((W / 2, 1030), "Ep. 2 · Redes convolucionales", font=fSub, fill=WHITE, anchor="mm")
    R.panel(d_, (200, 1120, 880, 1380))
    d_.text((W / 2, 1180), "¿QUÉ CUADRO LE PONGO?", font=fFollow, fill=GOLD, anchor="mm")
    d_.text((W / 2, 1250), "comenta una obra de dominio público", font=R.fMono, fill=GREY, anchor="mm")
    d_.text((W / 2, 1318), HANDLE, font=fHandle, fill=WHITE, anchor="mm")
    return img

def audio():
    sr = 44100; n = int(N / FPS * sr); t = np.arange(n) / sr; a = np.zeros(n)
    env = np.clip(t / 1.2, 0, 1) * np.clip((N / FPS - t) / 1.0, 0, 1)
    for fr in (110, 164.8, 220, 277.2): a += 0.04 * env * np.sin(2 * np.pi * fr * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.25 * t))
    def at(fr_, arr, g):
        i0 = int(fr_ / FPS * sr); m = min(len(arr), n - i0)
        if m > 0: a[i0:i0 + m] += g * arr[:m]
    tt = np.arange(int(.25 * sr)) / sr
    prev = None
    for f in range(0, S1, 3):                      # un toque cuando cambia el número de aciertos
        idx = int(round((f / (S1 - 1)) ** 0.9 * (NK - 1))); ok = int((PRED_RGB[idx] == TY).sum())
        if prev is not None and ok != prev: at(f, np.exp(-tt * 18) * np.sin(2 * np.pi * (330 + 12 * ok) * tt), 0.07)
        prev = ok
    m = int(1.2 * sr); tt2 = np.arange(m) / sr                 # caída al quitar el color
    at(S1, np.exp(-tt2 * 2) * np.sin(2 * np.pi * (440 - 160 * tt2) * tt2), 0.18)
    for j in range(40):
        if PRED_GRAY[j] != TY[j]: at(S1 + 22 + j * 0.9, np.exp(-tt * 20) * np.sin(2 * np.pi * 196 * tt), 0.08)
    i0 = int((S1 + S2 + HOLD_F) / FPS * sr); tt3 = np.arange(n - i0) / sr
    for fr in (880, 1318.5, 1760): a[i0:] += 0.08 * np.exp(-tt3 * 2.5) * np.sin(2 * np.pi * fr * tt3)
    a = a / np.abs(a).max() * 0.8
    with wave.open("work/cuadros.wav", "w") as wv:
        wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(sr); wv.writeframes((a * 32767).astype(np.int16).tobytes())

if __name__ == "__main__":
    import sys
    END = end_card(); LASTF = S1 + S2 - 1; LAST = scene(LASTF)
    def frame(f):
        if f <= LASTF: return scene(f)
        if f <= LASTF + HOLD_F: return LAST
        return Image.blend(LAST, END, min(1, (f - LASTF - HOLD_F) / 14))
    if len(sys.argv) > 1 and sys.argv[1] == "prev":
        for f in map(int, sys.argv[2:]): frame(f).convert("RGB").save(f"work/k{f}.png")
        sys.exit()
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "work/cuadros.mp4"], stdin=subprocess.PIPE)
    for f in range(N): p.stdin.write(frame(f).convert("RGB").tobytes())
    p.stdin.close(); p.wait(); audio()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "work/cuadros.mp4", "-i", "work/cuadros.wav", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "160k", "-shortest", "out/ep2b_cnn_cuadros.mp4"])
