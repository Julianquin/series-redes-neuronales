import numpy as np, pickle, subprocess, wave
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import render_mlp as R

W, H, FPS = 1080, 1920, 30
F1, F2 = 300, 90; TRAIN_F = F1 + F2; HOLD_F, END_F = 24, 66
N = TRAIN_F + HOLD_F + END_F
GOLD, BLUE, WHITE, GREY, EDGE, PANEL = R.GOLD, R.BLUE, R.WHITE, R.GREY, R.EDGE, R.PANEL
HANDLE = R.HANDLE
run = pickle.load(open("work/run_dbn.pkl", "rb"))
X, S1, S2, E1, E2, W1f = run["X"], run["S1"], run["S2"], run["E1"], run["E2"], run["W1"]
sig = lambda z: 1 / (1 + np.exp(-z))
def ema(a, k=0.95):
    o = a.copy()
    for i in range(1, len(a)): o[i] = k * o[i - 1] + (1 - k) * a[i]
    return o
E1s, E2s = ema(E1), ema(E2); T1, T2 = len(E1), len(E2); TT = T1 + T2
fH = R.font(R.JB, 24, 500); fS = R.font(R.JB, 22, 400); fBox = R.font(R.SG, 28, 700)
rng = np.random.default_rng(7)
REP = rng.choice(len(X), 8, replace=False); MASK = rng.random((8, 64)) < 0.5
CORR = np.where(MASK, rng.random((8, 64)) < 0.5, X[REP]).astype(float)

def gray_img(v, size):
    a = np.clip(v.reshape(8, 8), 0, 1)[..., None]
    rgb = np.array(PANEL, np.float32) * (1 - a) + np.array(WHITE, np.float32) * a
    return Image.fromarray(rgb.astype(np.uint8)).resize((size, size), Image.NEAREST)
def signed_img(w, size):
    v = np.clip(w / (np.abs(w).max() + 1e-9), -1, 1).reshape(8, 8)[..., None]
    base = np.array((18, 26, 35), np.float32)
    rgb = np.where(v > 0, base + (np.array(GOLD) - base) * v, base + (np.array(BLUE) - base) * (-v))
    return Image.fromarray(rgb.astype(np.uint8)).resize((size, size), Image.NEAREST)

def header(d_):
    d_.text((W / 2, 112), f"REDES NEURONALES · EP. 8 · {HANDLE}", font=R.fMonoS, fill=(70, 84, 98), anchor="mm")
    d_.text((W / 2, 205), "¿puede una red neuronal", font=R.fTitle, fill=GOLD, anchor="mm")
    pill = "aprender soñando"
    tb = d_.textbbox((0, 0), pill, font=R.fPill); tw = tb[2] - tb[0]
    x0 = W / 2 - (tw + 60) / 2 - 22; y0 = 262
    d_.rounded_rectangle((x0, y0, x0 + tw + 60, y0 + 86), radius=12, fill=(22, 30, 40), outline=(62, 78, 96), width=2)
    d_.text((x0 + 30 + tw / 2, y0 + 43), pill, font=R.fPill, fill=WHITE, anchor="mm")
    d_.text((x0 + tw + 60 + 34, y0 + 42), "?", font=R.fQ, fill=GOLD, anchor="mm")

LX0, LX1, LY0, LY1 = 150, 990, 1560, 1765
lmin, lmax = np.log10(0.004), np.log10(0.3)
def lpx(s, v): return LX0 + s / (TT - 1) * (LX1 - LX0), LY1 - (np.log10(max(v, 1e-4)) - lmin) / (lmax - lmin) * (LY1 - LY0)
def loss_panel(d_, s1, s2):
    R.panel(d_, (60, 1440, 1020, 1840))
    d_.text((90, 1478), "error de reconstrucción (log)", font=R.fMono, fill=GREY, anchor="lm")
    d_.line((720, 1470, 750, 1470), fill=BLUE, width=4); d_.text((762, 1470), f"capa 1 {E1s[s1 - 1]:.4f}", font=R.fMonoS, fill=WHITE, anchor="lm")
    if s2: d_.line((720, 1502, 750, 1502), fill=GOLD, width=4); d_.text((762, 1502), f"capa 2 {E2s[s2 - 1]:.4f}", font=R.fMonoS, fill=WHITE, anchor="lm")
    for v, lab in [(0.1, "0.1"), (0.01, "0.01")]:
        _, yy = lpx(0, v); d_.line((LX0, yy, LX1, yy), fill=(26, 36, 46), width=1); d_.text((LX0 - 14, yy), lab, font=R.fMonoS, fill=GREY, anchor="rm")
    d_.line((LX0, LY1, LX1, LY1), fill=EDGE, width=2)
    xs, _ = lpx(T1, 0.01)
    for yy in range(LY0, LY1, 14): d_.line((xs, yy, xs, yy + 7), fill=(90, 104, 118), width=1)
    d_.text(((LX0 + xs) / 2, LY1 + 26), "capa 1", font=R.fMonoS, fill=GREY, anchor="mm"); d_.text(((xs + LX1) / 2, LY1 + 26), "capa 2", font=R.fMonoS, fill=GREY, anchor="mm")
    d_.text(((LX0 + LX1) / 2, LY1 + 56), "se entrena una capa cada vez", font=R.fMonoS, fill=GREY, anchor="mm")
    st = max(1, s1 // 400)
    pts = [lpx(i, E1s[i]) for i in range(0, s1, st)] + [lpx(s1 - 1, E1s[s1 - 1])]
    d_.line(pts, fill=BLUE, width=3, joint="curve")
    if s2:
        pts2 = [lpx(T1 + i, E2s[i]) for i in range(0, s2)]
        if len(pts2) > 1: d_.line(pts2, fill=GOLD, width=3, joint="curve")
        d_.ellipse((pts2[-1][0] - 6, pts2[-1][1] - 6, pts2[-1][0] + 6, pts2[-1][1] + 6), fill=GOLD)
    else: d_.ellipse((pts[-1][0] - 6, pts[-1][1] - 6, pts[-1][0] + 6, pts[-1][1] + 6), fill=BLUE)

def snap_at(f):
    if f < F1: return 1, S1[min(len(S1) - 1, int(round((f / (F1 - 1)) ** 1.8 * (len(S1) - 1))))], None
    f2 = min(f - F1, F2 - 1); return 2, S1[-1], S2[min(len(S2) - 1, int(round(f2 / (F2 - 1) * (len(S2) - 1))))]

class Dreamer:
    def __init__(s): s.v = (np.random.default_rng(3).random((16, 64)) < 0.5).astype(float); s.r = np.random.default_rng(4)
    def step(s, Wt, b, c, k=2):
        for _ in range(k):
            h = s.r.random((16, Wt.shape[1])) < sig(s.v @ Wt + c); pv = sig(h @ Wt.T + b); s.v = (s.r.random(pv.shape) < pv).astype(float)
        return pv

def draw(f, pv_dream):
    stage, s1, s2 = snap_at(f); st1, Wt, b, c = s1
    img = R.BGIMG.copy(); d_ = ImageDraw.Draw(img, "RGBA"); header(d_)
    R.panel(d_, (60, 380, 1020, 1410))
    it_txt = f"capa 1 · paso {st1:,} / {T1:,}" if stage == 1 else f"capa 2 · paso {s2[0]:,} / {T2:,}"
    d_.text((92, 425), it_txt, font=R.fMono, fill=GREY, anchor="lm")
    d_.text((988, 425), "2 RBM apiladas", font=R.fMono, fill=GREY, anchor="rm")
    # stack diagram
    boxes = [("píxeles · 64", 0), ("capa 1 · 150", 1), ("capa 2 · 60", 2)]
    for i, (lab, lv) in enumerate(boxes):
        x0 = 100 + i * 310; on = lv == stage; done = lv < stage and lv > 0
        col = (BLUE if lv == 1 else GOLD) if on else ((80, 96, 112) if done or lv == 0 else (50, 62, 74))
        d_.rounded_rectangle((x0, 462, x0 + 260, 530), radius=12, fill=(18, 25, 33), outline=col, width=3 if on else 2)
        d_.text((x0 + 130, 496), lab, font=fBox, fill=col if on else GREY, anchor="mm")
        if i < 2:
            lc = (BLUE if i == 0 else GOLD) if stage == i + 1 else (60, 74, 88)
            d_.line((x0 + 268, 496, x0 + 302, 496), fill=lc, width=4)
            d_.polygon([(x0 + 270, 488), (x0 + 270, 504), (x0 + 260, 496)], fill=lc); d_.polygon([(x0 + 300, 488), (x0 + 300, 504), (x0 + 310, 496)], fill=lc)
    d_.line((80, 552, 1000, 552), fill=(26, 36, 46), width=1)
    # filters
    if stage == 1:
        d_.text((95, 578), "lo que detecta cada neurona de la capa 1", font=fH, fill=GREY, anchor="lm"); filt = [Wt[:, j] for j in range(24)]
    else:
        W2 = s2[1]; d_.text((95, 578), "capa 2: combina patrones de la capa 1", font=fH, fill=GOLD, anchor="lm"); filt = [W1f @ W2[:, j] for j in range(24)]
    for j, w in enumerate(filt):
        r_, c_ = divmod(j, 12); img.paste(signed_img(w, 64), (104 + c_ * 73, 598 + r_ * 73))
    d_ = ImageDraw.Draw(img, "RGBA")
    d_.line((80, 760, 1000, 760), fill=(26, 36, 46), width=1)
    d_.text((95, 786), "sueños: la red genera dígitos partiendo de ruido", font=fH, fill=GREY, anchor="lm")
    for j in range(16):
        r_, c_ = divmod(j, 8); img.paste(gray_img(pv_dream[j], 100), (104 + c_ * 110, 806 + r_ * 110))
    d_ = ImageDraw.Draw(img, "RGBA")
    d_.line((80, 1030, 1000, 1030), fill=(26, 36, 46), width=1)
    d_.text((95, 1056), "reparar: la mitad de los píxeles al azar → reconstrucción", font=fH, fill=GREY, anchor="lm")
    v = CORR.copy()
    for _ in range(4):
        h = sig(v @ Wt + c); v = np.where(MASK, sig(h @ Wt.T + b), CORR)
    for j in range(8):
        x = 104 + j * 110
        img.paste(gray_img(CORR[j], 100), (x, 1078)); img.paste(gray_img(v[j], 100), (x, 1246))
        d_ = ImageDraw.Draw(img, "RGBA"); d_.text((x + 50, 1212), "↓", font=fBox, fill=GREY, anchor="mm")
    loss_panel(d_, st1, s2[0] if stage == 2 else 0)
    return img

fSub = R.font(R.SG, 44, 500); fFollow = R.font(R.JB, 26, 600); fHandle = R.font(R.SG, 50, 700); fBrand = R.font(R.SG, 96, 700)
def end_card():
    img = R.BGIMG.copy(); d_ = ImageDraw.Draw(img, "RGBA")
    rows = [(5, 800), (4, 690), (3, 580)]
    nodes = [[(540 + (j - (n - 1) / 2) * 110, y_) for j in range(n)] for n, y_ in rows]
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
    for a, b in zip(nodes, nodes[1:]):
        for p in a:
            for q in b: ld.line((p, q), fill=GOLD + (110,), width=2)
    for row in nodes:
        for x, y_ in row: ld.ellipse((x - 26, y_ - 26, x + 26, y_ + 26), fill=GOLD + (130,))
    img.alpha_composite(lay.filter(ImageFilter.GaussianBlur(10))); img.alpha_composite(lay); d_ = ImageDraw.Draw(img, "RGBA")
    for row in nodes:
        for x, y_ in row: d_.ellipse((x - 17, y_ - 17, x + 17, y_ + 17), fill=GOLD, outline=(255, 226, 160), width=2)
    tl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ImageDraw.Draw(tl).text((W / 2, 940), "Redes Neuronales", font=fBrand, fill=GOLD, anchor="mm")
    img.alpha_composite(tl.filter(ImageFilter.GaussianBlur(10))); img.alpha_composite(tl); d_ = ImageDraw.Draw(img, "RGBA")
    d_.text((W / 2, 1030), "Ep. 8 · Deep Belief Networks", font=fSub, fill=WHITE, anchor="mm")
    R.panel(d_, (200, 1120, 880, 1380))
    d_.text((W / 2, 1180), "FIN DE LA TEMPORADA 1", font=fFollow, fill=GOLD, anchor="mm")
    d_.text((W / 2, 1250), "¿qué arquitectura quieres ver?", font=R.fMono, fill=GREY, anchor="mm")
    d_.text((W / 2, 1318), HANDLE, font=fHandle, fill=WHITE, anchor="mm")
    return img

def audio():
    sr = 44100; n_ = int(N / FPS * sr); t = np.arange(n_) / sr; a = np.zeros(n_)
    env = np.clip(t / 1.5, 0, 1) * np.clip((N / FPS - t) / 1.0, 0, 1)
    for fr in (98, 146.8, 196, 233.1): a += 0.04 * env * np.sin(2 * np.pi * fr * t) * (1 + 0.4 * np.sin(2 * np.pi * 0.15 * t + fr))
    shift = np.clip((t - F1 / FPS) * 3, 0, 1)
    for fr in (293.7, 349.2): a += 0.03 * env * shift * np.sin(2 * np.pi * fr * t)
    i0 = int((TRAIN_F + HOLD_F) / FPS * sr); tt = np.arange(n_ - i0) / sr
    for fr in (784, 1174.7, 1568): a[i0:] += 0.08 * np.exp(-tt * 2) * np.sin(2 * np.pi * fr * tt)
    a = a / np.abs(a).max() * 0.8
    with wave.open("work/dbn.wav", "w") as wv:
        wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(sr); wv.writeframes((a * 32767).astype(np.int16).tobytes())

def frames():
    dr = Dreamer(); END = end_card(); last = None
    for f in range(N):
        if f < TRAIN_F + HOLD_F:
            _, s1, _ = snap_at(min(f, TRAIN_F - 1)); pv = dr.step(s1[1], s1[2], s1[3])
            last = draw(min(f, TRAIN_F - 1), pv); yield f, last
        else:
            yield f, Image.blend(last, END, min(1, (f - TRAIN_F - HOLD_F) / 14))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        want = set(map(int, sys.argv[1:]))
        for f, im in frames():
            if f in want: im.convert("RGB").save(f"work/b{f}.png")
            if f >= max(want): break
        sys.exit()
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "work/dbn.mp4"], stdin=subprocess.PIPE)
    for f, im in frames(): p.stdin.write(im.convert("RGB").tobytes())
    p.stdin.close(); p.wait(); audio()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "work/dbn.mp4", "-i", "work/dbn.wav", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "160k", "-shortest", "out/ep8_dbn.mp4"])
