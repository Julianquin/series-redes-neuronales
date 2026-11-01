import numpy as np, pickle, subprocess, wave
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS = 1080, 1920, 30
TRAIN_F, HOLD_F, END_F = 390, 24, 66
N = TRAIN_F + HOLD_F + END_F
BG = (8, 12, 16); PANEL = (13, 19, 26); EDGE = (44, 58, 72); GREY = (120, 136, 150); WHITE = (226, 232, 238)
GOLD = (240, 186, 74); BLUE = (92, 156, 255); VALC = (150, 170, 190)
HANDLE = "@rojocuantico"

def font(path, size, wght):
    f = ImageFont.truetype(path, size)
    try: f.set_variation_by_axes([wght])
    except Exception: pass
    return f
SG = "fonts/SG.ttf"; JB = "fonts/JBM.ttf"
fTitle = font(SG, 66, 700); fPill = font(JB, 50, 600); fQ = font(SG, 74, 700)
fMono = font(JB, 26, 500); fMonoS = font(JB, 22, 400); fTiny = font(JB, 24, 400)

run = pickle.load(open("work/run.pkl", "rb"))
Xtr, ytr, Xva, yva, L, snaps = run["Xtr"], run["ytr"], run["Xva"], run["yva"], run["L"], run["snaps"]
keys = sorted(snaps); E = len(L)

# ---------- static background ----------
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
glow = np.exp(-(((xx - W / 2) / 520) ** 2 + ((yy - 300) / 260) ** 2)) * 0.10 + np.exp(-(((xx - W / 2) / 600) ** 2 + ((yy - 900) / 700) ** 2)) * 0.06
bg = np.zeros((H, W, 3), np.float32)
for i in range(3): bg[..., i] = BG[i] + glow * (GOLD[i] * 0.35 + BLUE[i] * 0.15)
BGIMG = Image.fromarray(bg.clip(0, 255).astype(np.uint8)).convert("RGBA")

def panel(d, box):
    d.rounded_rectangle(box, radius=18, fill=PANEL + (235,), outline=EDGE, width=2)

def header(img, d):
    d.text((W / 2, 112), f"REDES NEURONALES · EP. 1 · {HANDLE}", font=fMonoS, fill=(70, 84, 98), anchor="mm")
    d.text((W / 2, 205), "¿puede una red neuronal", font=fTitle, fill=GOLD, anchor="mm")
    pill = "separar 2 espirales"
    tb = d.textbbox((0, 0), pill, font=fPill); tw = tb[2] - tb[0]
    x0 = W / 2 - (tw + 60) / 2 - 22; y0 = 262
    d.rounded_rectangle((x0, y0, x0 + tw + 60, y0 + 86), radius=12, fill=(22, 30, 40), outline=(62, 78, 96), width=2)
    d.text((x0 + 30 + tw / 2, y0 + 43), pill, font=fPill, fill=WHITE, anchor="mm")
    d.text((x0 + tw + 60 + 34, y0 + 42), "?", font=fQ, fill=GOLD, anchor="mm")

# ---------- decision panel ----------
PX0, PY0, PS = 110, 470, 860      # plot square
LIM = 6.6
G = 220
gx = np.linspace(-LIM, LIM, G); GX, GY = np.meshgrid(gx, -gx)
GRID = np.c_[GX.ravel(), GY.ravel()]
def predict(P, X):
    h = X
    for i, (Wt, b) in enumerate(P):
        z = h @ Wt + b; h = np.maximum(z, 0) if i < len(P) - 1 else 1 / (1 + np.exp(-z))
    return h[:, 0]
to_px = lambda x, y: (PX0 + (x + LIM) / (2 * LIM) * PS, PY0 + (LIM - y) / (2 * LIM) * PS)

def decision_img(P):
    p = predict(P, GRID).reshape(G, G).astype(np.float32)
    pu = np.array(Image.fromarray(p, "F").resize((PS, PS), Image.BILINEAR))
    conf = np.abs(2 * pu - 1) ** 0.6
    base = np.array(PANEL, np.float32)
    tint = np.array(BLUE, np.float32) * (1 - pu[..., None]) + np.array(GOLD, np.float32) * pu[..., None]
    rgb = base + (tint - base) * (0.30 * conf[..., None])
    gy_, gx_ = np.gradient(pu); gm = np.hypot(gx_, gy_)
    band = np.clip(1.6 - np.abs(pu - 0.5) / (gm + 1e-6) / 1.4, 0, 1) * (gm > 1e-4)
    rgb = rgb * (1 - band[..., None] * 0.9) + np.array(WHITE, np.float32) * band[..., None] * 0.9
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).convert("RGBA")

def draw_points(d):
    for (x, y), c in zip(Xva, yva):
        px, py = to_px(x, y); col = GOLD if c else BLUE
        d.ellipse((px - 5, py - 5, px + 5, py + 5), outline=col + (150,), width=2)
    for (x, y), c in zip(Xtr, ytr):
        px, py = to_px(x, y); col = GOLD if c else BLUE
        d.ellipse((px - 6, py - 6, px + 6, py + 6), fill=col, outline=(5, 8, 12), width=2)

# ---------- loss panel ----------
LX0, LX1, LY0, LY1 = 150, 990, 1560, 1765
lmin, lmax = np.log10(0.008), np.log10(2.0)
def lpx(ep, v):
    return LX0 + (ep - 1) / (E - 1) * (LX1 - LX0), LY1 - (np.log10(max(v, 1e-4)) - lmin) / (lmax - lmin) * (LY1 - LY0)

def loss_panel(d, ep):
    panel(d, (60, 1440, 1020, 1840))
    d.text((90, 1478), "BCE loss (escala log)", font=fMono, fill=GREY, anchor="lm")
    tr, va, atr, ava = L[ep - 1]
    d.line((720, 1470, 750, 1470), fill=BLUE, width=4); d.text((762, 1470), f"train {tr:.4f}", font=fMonoS, fill=WHITE, anchor="lm")
    d.line((720, 1502, 750, 1502), fill=VALC, width=4); d.text((762, 1502), f"valid {va:.4f}", font=fMonoS, fill=WHITE, anchor="lm")
    for v, lab in [(1, "1"), (0.1, "0.1"), (0.01, "0.01")]:
        _, y = lpx(1, v); d.line((LX0, y, LX1, y), fill=(26, 36, 46), width=1)
        d.text((LX0 - 14, y), lab, font=fMonoS, fill=GREY, anchor="rm")
    d.line((LX0, LY1, LX1, LY1), fill=EDGE, width=2)
    for e in (0, 1000, 2000, 3000):
        x = LX0 + e / E * (LX1 - LX0); d.text((min(x, LX1 - 30), LY1 + 26), f"{e:,}", font=fMonoS, fill=GREY, anchor="mm")
    d.text(((LX0 + LX1) / 2, LY1 + 56), "epoch", font=fMonoS, fill=GREY, anchor="mm")
    step = max(1, ep // 600)
    for col, j in ((VALC, 1), (BLUE, 0)):
        pts = [lpx(e, L[e - 1, j]) for e in range(1, ep + 1, step)] + [lpx(ep, L[ep - 1, j])]
        if len(pts) > 1: d.line(pts, fill=col, width=3, joint="curve")
        d.ellipse((pts[-1][0] - 6, pts[-1][1] - 6, pts[-1][0] + 6, pts[-1][1] + 6), fill=col)

dcache = {}
def train_frame(fi):
    k = keys[min(len(keys) - 1, int(round(fi / (TRAIN_F - 1) * (len(keys) - 1))))]
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    header(img, d)
    panel(d, (60, 380, 1020, 1410))
    tr, va, atr, ava = L[k - 1]
    d.text((92, 425), f"epoch {k:,} / {E:,}", font=fMono, fill=GREY, anchor="lm")
    d.text((988, 425), "ReLU · BCE · [2,64,64,1]", font=fMono, fill=GREY, anchor="rm")
    if k not in dcache: dcache[k] = decision_img(snaps[k])
    img.alpha_composite(dcache[k], (PX0, PY0)); d = ImageDraw.Draw(img, "RGBA")
    d.rectangle((PX0, PY0, PX0 + PS, PY0 + PS), outline=EDGE, width=1)
    draw_points(d)
    acc_col = GOLD if ava > 0.9 else WHITE
    d.text((PX0 + PS - 18, PY0 + 26), f"acc valid {ava * 100:.1f}%", font=fMono, fill=acc_col, anchor="rm")
    ly = 1372
    d.ellipse((150, ly - 8, 166, ly + 8), fill=BLUE); d.text((176, ly), "clase A", font=fTiny, fill=GREY, anchor="lm")
    d.ellipse((330, ly - 8, 346, ly + 8), fill=GOLD); d.text((356, ly), "clase B", font=fTiny, fill=GREY, anchor="lm")
    d.ellipse((510, ly - 7, 524, ly + 7), outline=GREY, width=2); d.text((534, ly), "validación", font=fTiny, fill=GREY, anchor="lm")
    d.line((720, ly, 760, ly), fill=WHITE, width=4); d.text((772, ly), "frontera", font=fTiny, fill=GREY, anchor="lm")
    loss_panel(d, k)
    return img

# ---------- end card ----------
fBrand = font(SG, 96, 700); fSub = font(SG, 44, 500); fFollow = font(JB, 26, 600); fHandle = font(SG, 50, 700)
def end_card():
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    cx, cy = W / 2, 690
    layers = [[-1, 0, 1], [-1.5, -0.5, 0.5, 1.5], [-1.5, -0.5, 0.5, 1.5], [0]]
    xs = [-240, -80, 80, 240]; nodes = [[(cx + xs[i], cy + v * 70) for v in L_] for i, L_ in enumerate(layers)]
    for a, b in zip(nodes, nodes[1:]):
        for p in a:
            for q in b: d.line((p, q), fill=GOLD + (90,), width=2)
    glowl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(glowl)
    for col in nodes:
        for (x, y) in col: gd.ellipse((x - 26, y - 26, x + 26, y + 26), fill=GOLD + (140,))
    img.alpha_composite(glowl.filter(ImageFilter.GaussianBlur(14))); d = ImageDraw.Draw(img, "RGBA")
    for col in nodes:
        for (x, y) in col: d.ellipse((x - 17, y - 17, x + 17, y + 17), fill=GOLD, outline=(255, 226, 160), width=2)
    tl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); td = ImageDraw.Draw(tl)
    td.text((W / 2, 940), "Redes Neuronales", font=fBrand, fill=GOLD, anchor="mm")
    img.alpha_composite(tl.filter(ImageFilter.GaussianBlur(10))); img.alpha_composite(tl); d = ImageDraw.Draw(img, "RGBA")
    d.text((W / 2, 1030), "Ep. 1 · El perceptrón multicapa", font=fSub, fill=WHITE, anchor="mm")
    panel(d, (200, 1120, 880, 1380))
    d.text((W / 2, 1180), "PRÓXIMO EPISODIO: CNN", font=fFollow, fill=GOLD, anchor="mm")
    d.text((W / 2, 1250), "sígueme para no perdértelo", font=fMono, fill=GREY, anchor="mm")
    d.text((W / 2, 1318), HANDLE, font=fHandle, fill=WHITE, anchor="mm")
    return img

if __name__ == "__main__":
    import sys
    END = end_card()
    def frame(f):
        if f < TRAIN_F: return train_frame(f)
        last = train_frame(TRAIN_F - 1)
        if f < TRAIN_F + HOLD_F: return last
        a = min(1, (f - TRAIN_F - HOLD_F) / 14)
        return Image.blend(last, END, a)
    if len(sys.argv) > 1:
        for f in map(int, sys.argv[1:]): frame(f).convert("RGB").save(f"work/m{f}.png")
        sys.exit()
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "work/mlp.mp4"], stdin=subprocess.PIPE)
    lastimg = None
    for f in range(N):
        if TRAIN_F <= f < TRAIN_F + HOLD_F and lastimg is not None: im = lastimg
        else: im = frame(f)
        if f == TRAIN_F - 1: lastimg = im
        p.stdin.write(im.convert("RGB").tobytes())
    p.stdin.close(); p.wait()
    # audio: soft pad + rising arpeggio while training, chime on end card
    sr = 44100; n = int(N / FPS * sr); t = np.arange(n) / sr; a = np.zeros(n)
    env = np.clip(t / 1.2, 0, 1) * np.clip((N / FPS - t) / 1.0, 0, 1)
    for fr in (110, 164.8, 220, 277.2): a += 0.05 * env * np.sin(2 * np.pi * fr * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.2 * t))
    notes = [440, 554.4, 659.3, 880]
    for i in range(int(TRAIN_F / FPS / 0.25)):
        s = i * 0.25; idx = int(s * sr); m = int(0.5 * sr); tt = np.arange(min(m, n - idx)) / sr
        a[idx:idx + len(tt)] += 0.05 * (0.4 + s / 13) * np.exp(-tt * 7) * np.sin(2 * np.pi * notes[i % 4] * tt)
    s = (TRAIN_F + HOLD_F) / FPS; idx = int(s * sr); tt = np.arange(n - idx) / sr
    for fr in (880, 1318.5, 1760): a[idx:] += 0.08 * np.exp(-tt * 2.5) * np.sin(2 * np.pi * fr * tt)
    a = a / np.abs(a).max() * 0.8
    with wave.open("work/mlp.wav", "w") as wv:
        wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(sr); wv.writeframes((a * 32767).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "work/mlp.mp4", "-i", "work/mlp.wav", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "160k", "-shortest", "out/ep1_mlp_espirales.mp4"])
