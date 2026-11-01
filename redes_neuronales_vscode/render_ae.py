import numpy as np, pickle, subprocess, wave
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import render_mlp as R

W, H, FPS = 1080, 1920, 30
TRAIN_F, HOLD_F, END_F = 390, 24, 66
N = TRAIN_F + HOLD_F + END_F
GOLD, BLUE, WHITE, GREY, EDGE, PANEL = R.GOLD, R.BLUE, R.WHITE, R.GREY, R.EDGE, R.PANEL
HANDLE = R.HANDLE
CLS = [(240, 186, 74), (92, 156, 255), (236, 96, 96), (96, 206, 140), (190, 120, 255),
       (255, 140, 60), (80, 210, 220), (240, 110, 190), (205, 205, 120), (170, 180, 195)]
run = pickle.load(open("work/run_ae.pkl", "rb"))
X, y, va, L, snaps = run["X"], run["y"], run["va"], run["L"], run["snaps"]
keys = sorted(snaps); E = len(L)
sig = lambda z: 1 / (1 + np.exp(-z))
def enc(P, X):
    h = np.maximum(X @ P[0][0] + P[0][1], 0); return h @ P[1][0] + P[1][1]
def dec(P, Z):
    h = np.maximum(Z @ P[2][0] + P[2][1], 0); return sig(h @ P[3][0] + P[3][1])

SHOW = np.random.default_rng(1).permutation(len(X))[:1100]
EXS = [va[np.where(y[va] == c)[0][0]] for c in range(10)]
fH = R.font(R.JB, 24, 500); fLeg = R.font(R.SG, 34, 700); fCen = R.font(R.SG, 46, 700)

def kidx(fi): return keys[min(len(keys) - 1, int(round(fi / (TRAIN_F - 1) * (len(keys) - 1))))]
# smoothed latent view bounds per frame
bounds = []; cur = None
for fi in range(TRAIN_F):
    Z = enc(snaps[kidx(fi)], X[SHOW]); lo, hi = np.percentile(Z, 1, 0), np.percentile(Z, 99, 0)
    c = (lo + hi) / 2; r = (hi - lo).max() / 2 * 1.12; b = np.r_[c, r]
    cur = b if cur is None else 0.8 * cur + 0.2 * b; bounds.append(cur.copy())

def digit_img(x, size):
    a = x.reshape(8, 8); im = Image.fromarray((a * 255).clip(0, 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
    arr = np.array(im).astype(np.float32)[..., None] / 255
    return Image.fromarray((np.array(PANEL, np.float32) * (1 - arr) + np.array(WHITE, np.float32) * arr).astype(np.uint8))

def header(d_):
    d_.text((W / 2, 112), f"REDES NEURONALES · EP. 5 · {HANDLE}", font=R.fMonoS, fill=(70, 84, 98), anchor="mm")
    d_.text((W / 2, 205), "¿puede una red neuronal", font=R.fTitle, fill=GOLD, anchor="mm")
    pill = "comprimir en 2 números"
    tb = d_.textbbox((0, 0), pill, font=R.fPill); tw = tb[2] - tb[0]
    x0 = W / 2 - (tw + 60) / 2 - 22; y0 = 262
    d_.rounded_rectangle((x0, y0, x0 + tw + 60, y0 + 86), radius=12, fill=(22, 30, 40), outline=(62, 78, 96), width=2)
    d_.text((x0 + 30 + tw / 2, y0 + 43), pill, font=R.fPill, fill=WHITE, anchor="mm")
    d_.text((x0 + tw + 60 + 34, y0 + 42), "?", font=R.fQ, fill=GOLD, anchor="mm")

LX0, LX1, LY0, LY1 = 150, 990, 1560, 1765
lmin, lmax = np.log10(0.02), np.log10(0.15)
def lpx(it, v): return LX0 + (it - 1) / (E - 1) * (LX1 - LX0), LY1 - (np.log10(max(v, 1e-4)) - lmin) / (lmax - lmin) * (LY1 - LY0)
def loss_panel(d_, k):
    R.panel(d_, (60, 1440, 1020, 1840))
    d_.text((90, 1478), "error de reconstrucción (MSE, log)", font=R.fMono, fill=GREY, anchor="lm")
    tr, vl = L[k - 1]
    d_.line((720, 1470, 750, 1470), fill=BLUE, width=4); d_.text((762, 1470), f"train {tr:.4f}", font=R.fMonoS, fill=WHITE, anchor="lm")
    d_.line((720, 1502, 750, 1502), fill=R.VALC, width=4); d_.text((762, 1502), f"valid {vl:.4f}", font=R.fMonoS, fill=WHITE, anchor="lm")
    for v, lab in [(0.1, "0.1"), (0.05, "0.05"), (0.03, "0.03")]:
        _, yy = lpx(1, v); d_.line((LX0, yy, LX1, yy), fill=(26, 36, 46), width=1); d_.text((LX0 - 14, yy), lab, font=R.fMonoS, fill=GREY, anchor="rm")
    d_.line((LX0, LY1, LX1, LY1), fill=EDGE, width=2)
    for e in (0, 500, 1000, 1500, 2000):
        x = LX0 + e / E * (LX1 - LX0); d_.text((min(x, LX1 - 26), LY1 + 26), f"{e:,}", font=R.fMonoS, fill=GREY, anchor="mm")
    d_.text(((LX0 + LX1) / 2, LY1 + 56), "iteración", font=R.fMonoS, fill=GREY, anchor="mm")
    step = max(1, k // 500)
    for col, j in ((R.VALC, 1), (BLUE, 0)):
        pts = [lpx(e, L[e - 1, j]) for e in range(1, k + 1, step)] + [lpx(k, L[k - 1, j])]
        if len(pts) > 1: d_.line(pts, fill=col, width=3, joint="curve")
        d_.ellipse((pts[-1][0] - 6, pts[-1][1] - 6, pts[-1][0] + 6, pts[-1][1] + 6), fill=col)

SX0, SY0, SS = 95, 835, 560
GXN, GC = 5, 56; GX0, GY0 = 690, 835
def train_frame(fi):
    k = kidx(fi); P = snaps[k]; cx, cy, r = bounds[fi]
    img = R.BGIMG.copy(); d_ = ImageDraw.Draw(img, "RGBA"); header(d_)
    R.panel(d_, (60, 380, 1020, 1410))
    d_.text((92, 425), f"iter {k:,} / {E:,}", font=R.fMono, fill=GREY, anchor="lm")
    d_.text((988, 425), "64 → 128 → 2 → 128 → 64", font=R.fMono, fill=GREY, anchor="rm")
    # reconstructions
    rec = dec(P, enc(P, X[EXS]))
    d_.text((104, 478), "original", font=fH, fill=GREY, anchor="lm")
    for j in range(10):
        x = 104 + j * 88
        img.paste(digit_img(X[EXS[j]], 80), (x, 495)); img.paste(digit_img(rec[j], 80), (x, 655))
        d_ = ImageDraw.Draw(img, "RGBA")
        d_.rectangle((x, 495, x + 80, 575), outline=EDGE); d_.rectangle((x, 655, x + 80, 735), outline=CLS[j], width=2)
        d_.line((x + 40, 582, x + 40, 646), fill=(60, 74, 88), width=2)
        d_.ellipse((x + 33, 607, x + 47, 621), fill=CLS[j])
    d_.text((W / 2, 614), "", font=fH, fill=GREY)
    d_.text((104, 758), "reconstrucción desde solo 2 números", font=fH, fill=GREY, anchor="lm")
    d_.line((80, 788, 1000, 788), fill=(26, 36, 46), width=1)
    # latent scatter
    d_.text((95, 812), "espacio latente (z₁, z₂)", font=fH, fill=GREY, anchor="lm")
    d_.rectangle((SX0, SY0, SX0 + SS, SY0 + SS), fill=(11, 16, 22), outline=EDGE)
    Z = enc(P, X[SHOW]); to = lambda z: (SX0 + (z[0] - cx + r) / (2 * r) * SS, SY0 + (cy + r - z[1]) / (2 * r) * SS)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
    for z, c in zip(Z, y[SHOW]):
        px, py = to(z)
        if SX0 + 4 < px < SX0 + SS - 4 and SY0 + 4 < py < SY0 + SS - 4: ld.ellipse((px - 4, py - 4, px + 4, py + 4), fill=CLS[c] + (190,))
    img.alpha_composite(lay); d_ = ImageDraw.Draw(img, "RGBA")
    # manifold grid points + decoded grid
    gv = np.linspace(-0.7, 0.7, GXN); grid = np.array([[cx + a * r, cy - b * r] for b in gv for a in gv])
    for z in grid:
        px, py = to(z); d_.line((px - 6, py, px + 6, py), fill=WHITE + (160,), width=2); d_.line((px, py - 6, px, py + 6), fill=WHITE + (160,), width=2)
    for c in range(10):
        m = np.median(Z[y[SHOW] == c], 0); px, py = to(m)
        if SX0 + 20 < px < SX0 + SS - 20 and SY0 + 20 < py < SY0 + SS - 20 and fi > 60:
            d_.text((px, py), str(c), font=fCen, fill=CLS[c], anchor="mm", stroke_width=4, stroke_fill=(8, 12, 16))
    d_.text((GX0, 812), "lo que imagina el decoder", font=fH, fill=GREY, anchor="lm")
    imgs = dec(P, grid)
    for i, im in enumerate(imgs):
        rr, cc = divmod(i, GXN); img.paste(digit_img(im, GC), (GX0 + cc * (GC + 4), GY0 + rr * (GC + 4)))
    d_ = ImageDraw.Draw(img, "RGBA")
    d_.text((GX0, GY0 + GXN * (GC + 4) + 26), "= las + del mapa", font=R.fMonoS, fill=(80, 94, 108), anchor="lm")
    for c in range(10):
        rr, cc = divmod(c, 5); x = GX0 + cc * 60; yy = 1230 + rr * 60
        d_.text((x + 18, yy), str(c), font=fLeg, fill=CLS[c], anchor="mm")
    d_.text((GX0, 1340), "color = dígito real", font=R.fMonoS, fill=(80, 94, 108), anchor="lm")
    d_.text((GX0, 1366), "(no ve la etiqueta)", font=R.fMonoS, fill=(80, 94, 108), anchor="lm")
    loss_panel(d_, k)
    return img

fSub = R.font(R.SG, 44, 500); fFollow = R.font(R.JB, 26, 600); fHandle = R.font(R.SG, 50, 700); fBrand = R.font(R.SG, 96, 700)
def end_card():
    img = R.BGIMG.copy(); d_ = ImageDraw.Draw(img, "RGBA")
    cols = [(4, 260), (3, 400), (1, 540), (3, 680), (4, 820)]; y0 = 690
    nodes = [[(x, y0 + (j - (n - 1) / 2) * 60) for j in range(n)] for n, x in cols]
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
    for a, b in zip(nodes, nodes[1:]):
        for p in a:
            for q in b: ld.line((p, q), fill=GOLD + (90,), width=2)
    for col in nodes:
        for (x, yy) in col: ld.ellipse((x - 24, yy - 24, x + 24, yy + 24), fill=GOLD + (130,))
    img.alpha_composite(lay.filter(ImageFilter.GaussianBlur(10))); img.alpha_composite(lay); d_ = ImageDraw.Draw(img, "RGBA")
    for col in nodes:
        for (x, yy) in col: d_.ellipse((x - 16, yy - 16, x + 16, yy + 16), fill=GOLD, outline=(255, 226, 160), width=2)
    d_.text((540, y0 + 60), "z", font=R.fPill, fill=WHITE, anchor="mm")
    tl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ImageDraw.Draw(tl).text((W / 2, 940), "Redes Neuronales", font=fBrand, fill=GOLD, anchor="mm")
    img.alpha_composite(tl.filter(ImageFilter.GaussianBlur(10))); img.alpha_composite(tl); d_ = ImageDraw.Draw(img, "RGBA")
    d_.text((W / 2, 1030), "Ep. 5 · Autoencoders", font=fSub, fill=WHITE, anchor="mm")
    R.panel(d_, (200, 1120, 880, 1380))
    d_.text((W / 2, 1180), "PRÓXIMO EPISODIO: GAN", font=fFollow, fill=GOLD, anchor="mm")
    d_.text((W / 2, 1250), "sígueme para no perdértelo", font=R.fMono, fill=GREY, anchor="mm")
    d_.text((W / 2, 1318), HANDLE, font=fHandle, fill=WHITE, anchor="mm")
    return img

def audio():
    sr = 44100; n = int(N / FPS * sr); t = np.arange(n) / sr; a = np.zeros(n)
    env = np.clip(t / 1.2, 0, 1) * np.clip((N / FPS - t) / 1.0, 0, 1)
    # pad that "focuses": detuned chord converging to a clean chord as training progresses
    det = 6 * np.clip(1 - t / (TRAIN_F / FPS), 0, 1)
    for fr in (110, 164.8, 220, 277.2):
        a += 0.03 * env * (np.sin(2 * np.pi * fr * t) + np.sin(2 * np.pi * (fr + det) * t))
    i0 = int((TRAIN_F + HOLD_F) / FPS * sr); tt = np.arange(n - i0) / sr
    for fr in (880, 1318.5, 1760): a[i0:] += 0.08 * np.exp(-tt * 2.5) * np.sin(2 * np.pi * fr * tt)
    a = a / np.abs(a).max() * 0.8
    with wave.open("work/ae.wav", "w") as wv:
        wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(sr); wv.writeframes((a * 32767).astype(np.int16).tobytes())

if __name__ == "__main__":
    import sys
    END = end_card(); LAST = train_frame(TRAIN_F - 1)
    def frame(f):
        if f < TRAIN_F: return train_frame(f)
        if f < TRAIN_F + HOLD_F: return LAST
        return Image.blend(LAST, END, min(1, (f - TRAIN_F - HOLD_F) / 14))
    if len(sys.argv) > 1:
        for f in map(int, sys.argv[1:]): frame(f).convert("RGB").save(f"work/e{f}.png")
        sys.exit()
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "work/ae.mp4"], stdin=subprocess.PIPE)
    for f in range(N): p.stdin.write(frame(f).convert("RGB").tobytes())
    p.stdin.close(); p.wait(); audio()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "work/ae.mp4", "-i", "work/ae.wav", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "160k", "-shortest", "out/ep5_autoencoder.mp4"])
