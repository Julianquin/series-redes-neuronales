import numpy as np, pickle, subprocess, wave
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import render_mlp as R

W, H, FPS = 1080, 1920, 30
TRAIN_F, HOLD_F, END_F = 390, 24, 66
N = TRAIN_F + HOLD_F + END_F
GOLD, BLUE, WHITE, GREY, EDGE, PANEL = R.GOLD, R.BLUE, R.WHITE, R.GREY, R.EDGE, R.PANEL
REDC = (236, 96, 96); MID = (110, 120, 132)
HANDLE = R.HANDLE
run = pickle.load(open("work/run_gnn.pkl", "rb"))
L, snaps, yt, Ah, edges, pos, LAB = run["L"], run["snaps"], run["y"], run["Ah"], run["edges"], run["pos"], run["LAB"]
keys = sorted(snaps); E = len(L); n = len(yt)
def fwd(P):
    H1 = np.tanh(Ah @ P['W1']); H2 = np.tanh(Ah @ H1 @ P['W2'])
    lg = H2 @ P['W3'] + P['b3']; lg -= lg.max(1, keepdims=True); p = np.exp(lg); return H2, p / p.sum(1, keepdims=True)
fH = R.font(R.JB, 24, 500); fS = R.font(R.JB, 22, 400); fTag = R.font(R.SG, 26, 700); fStat = R.font(R.SG, 72, 700)
lerp = lambda a, b, t: tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
def ncol(p1):
    c = lerp(BLUE, GOLD, p1); return lerp(MID, c, min(1, abs(2 * p1 - 1) * 1.3))

def header(d_):
    d_.text((W / 2, 112), f"REDES NEURONALES · EP. 7 · {HANDLE}", font=R.fMonoS, fill=(70, 84, 98), anchor="mm")
    d_.text((W / 2, 205), "¿puede una red neuronal", font=R.fTitle, fill=GOLD, anchor="mm")
    pill = "predecir una ruptura"
    tb = d_.textbbox((0, 0), pill, font=R.fPill); tw = tb[2] - tb[0]
    x0 = W / 2 - (tw + 60) / 2 - 22; y0 = 262
    d_.rounded_rectangle((x0, y0, x0 + tw + 60, y0 + 86), radius=12, fill=(22, 30, 40), outline=(62, 78, 96), width=2)
    d_.text((x0 + 30 + tw / 2, y0 + 43), pill, font=R.fPill, fill=WHITE, anchor="mm")
    d_.text((x0 + tw + 60 + 34, y0 + 42), "?", font=R.fQ, fill=GOLD, anchor="mm")

GX0, GX1, GY0, GY1 = 110, 970, 590, 1140
mn, mx = pos.min(0), pos.max(0)
P2 = np.c_[GX0 + (pos[:, 0] - mn[0]) / (mx[0] - mn[0]) * (GX1 - GX0), GY0 + (pos[:, 1] - mn[1]) / (mx[1] - mn[1]) * (GY1 - GY0)]
rng = np.random.default_rng(0); PH = rng.uniform(0, 1, len(edges)); DIR = rng.integers(0, 2, len(edges))

LX0, LX1, LY0, LY1 = 150, 990, 1560, 1765
lmin, lmax = np.log10(0.005), np.log10(1.5)
def lpx(it, v): return LX0 + (it - 1) / (E - 1) * (LX1 - LX0), LY1 - (np.log10(max(v, 1e-4)) - lmin) / (lmax - lmin) * (LY1 - LY0)
def loss_panel(d_, k):
    R.panel(d_, (60, 1440, 1020, 1840))
    d_.text((90, 1478), "error en los 2 nodos conocidos (log)", font=R.fMono, fill=GREY, anchor="lm")
    d_.line((800, 1478, 830, 1478), fill=BLUE, width=4); d_.text((842, 1478), f"{L[k - 1, 0]:.4f}", font=R.fMonoS, fill=WHITE, anchor="lm")
    for v, lab in [(1, "1"), (0.1, "0.1"), (0.01, "0.01")]:
        _, yy = lpx(1, v); d_.line((LX0, yy, LX1, yy), fill=(26, 36, 46), width=1); d_.text((LX0 - 14, yy), lab, font=R.fMonoS, fill=GREY, anchor="rm")
    d_.line((LX0, LY1, LX1, LY1), fill=EDGE, width=2)
    for e in (0, 50, 100, 150, 200):
        x = LX0 + e / E * (LX1 - LX0); d_.text((min(x, LX1 - 20), LY1 + 26), str(e), font=R.fMonoS, fill=GREY, anchor="mm")
    d_.text(((LX0 + LX1) / 2, LY1 + 56), "iteración", font=R.fMonoS, fill=GREY, anchor="mm")
    pts = [lpx(e, L[e - 1, 0]) for e in range(1, k + 1)]
    if len(pts) > 1: d_.line(pts, fill=BLUE, width=3, joint="curve")
    d_.ellipse((pts[-1][0] - 6, pts[-1][1] - 6, pts[-1][0] + 6, pts[-1][1] + 6), fill=BLUE)

def kidx(fi): return keys[min(len(keys) - 1, int(round((fi / (TRAIN_F - 1)) ** 1.8 * (len(keys) - 1))))]
def scene(fi, k):
    Hz, p = fwd(snaps[k]); p1 = p[:, 1]; pred = p.argmax(1)
    img = R.BGIMG.copy(); d_ = ImageDraw.Draw(img, "RGBA"); header(d_)
    R.panel(d_, (60, 380, 1020, 1410))
    d_.text((92, 425), f"iter {k} / {E}", font=R.fMono, fill=GREY, anchor="lm")
    d_.text((988, 425), "GCN · 2 capas · 34 nodos", font=R.fMono, fill=GREY, anchor="rm")
    d_.text((95, 478), "Club de kárate de Zachary (1977): una pelea partió el club en dos.", font=fS, fill=WHITE, anchor="lm")
    d_.text((95, 510), "La red solo sabe de qué lado están 2 personas. ¿Y el resto?", font=fS, fill=GREY, anchor="lm")
    d_.line((80, 545, 1000, 545), fill=(26, 36, 46), width=1)
    # edges + message pulses
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
    for (a, b) in edges: ld.line((tuple(P2[a]), tuple(P2[b])), fill=(90, 104, 118, 110), width=2)
    strength = min(1, fi / 120) if fi < TRAIN_F else 1
    t = fi / FPS
    for e_, (a, b) in enumerate(edges):
        if DIR[e_]: a, b = b, a
        ph = (t * 0.7 + PH[e_]) % 1; q = P2[a] + (P2[b] - P2[a]) * ph
        c = ncol(p1[a]); al = int(220 * strength * np.sin(np.pi * ph))
        ld.ellipse((q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4), fill=c + (al,))
    img.alpha_composite(lay); d_ = ImageDraw.Draw(img, "RGBA")
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(glow)
    for i in range(n):
        x, y_ = P2[i]; gd.ellipse((x - 26, y_ - 26, x + 26, y_ + 26), fill=ncol(p1[i]) + (int(150 * abs(2 * p1[i] - 1)),))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(10))); d_ = ImageDraw.Draw(img, "RGBA")
    for i in range(n):
        x, y_ = P2[i]; r = 24 if i in LAB else 17
        out = WHITE if i in LAB else (REDC if pred[i] != yt[i] and k > 1 else (8, 12, 16))
        d_.ellipse((x - r, y_ - r, x + r, y_ + r), fill=ncol(p1[i]), outline=out, width=5 if i in LAB or out == REDC else 2)
    for i, txt in zip(LAB, ("instructor", "presidente")):
        x, y_ = P2[i]; col = BLUE if yt[i] == 0 else GOLD
        d_.text((x, y_ + 46), txt, font=fTag, fill=col, anchor="mm", stroke_width=4, stroke_fill=(10, 14, 18))
    d_.line((80, 1165, 1000, 1165), fill=(26, 36, 46), width=1)
    # embedding
    d_.text((95, 1190), "lo que ve la red: 2 números por persona", font=fS, fill=GREY, anchor="lm")
    ex0, ey0, es = 95, 1210, 180
    d_.rectangle((ex0, ey0, ex0 + es, ey0 + es), fill=(11, 16, 22), outline=EDGE)
    for i in range(n):
        x = ex0 + (Hz[i, 0] + 1) / 2 * es; y_ = ey0 + (1 - Hz[i, 1]) / 2 * es
        d_.ellipse((x - 5, y_ - 5, x + 5, y_ + 5), fill=ncol(p1[i]))
    acc = (pred == yt).sum()
    d_.text((560, 1235), "aciertos", font=fH, fill=GREY, anchor="lm")
    d_.text((560, 1300), f"{acc}/{n}", font=fStat, fill=GOLD if acc >= 33 else WHITE, anchor="lm")
    d_.text((560, 1370), "frente al bando real", font=fS, fill=GREY, anchor="lm")
    ly = 1236
    d_.ellipse((330, ly - 9, 348, ly + 9), fill=BLUE); d_.text((358, ly), "instructor", font=fS, fill=GREY, anchor="lm")
    d_.ellipse((330, ly + 32, 348, ly + 50), fill=GOLD); d_.text((358, ly + 41), "presidente", font=fS, fill=GREY, anchor="lm")
    d_.ellipse((330, ly + 73, 348, ly + 91), fill=MID); d_.text((358, ly + 82), "dudoso", font=fS, fill=GREY, anchor="lm")
    d_.ellipse((330, ly + 114, 348, ly + 132), outline=REDC, width=4); d_.text((358, ly + 123), "error", font=fS, fill=GREY, anchor="lm")
    loss_panel(d_, k)
    return img

fSub = R.font(R.SG, 44, 500); fFollow = R.font(R.JB, 26, 600); fHandle = R.font(R.SG, 50, 700); fBrand = R.font(R.SG, 96, 700)
def end_card():
    img = R.BGIMG.copy(); d_ = ImageDraw.Draw(img, "RGBA")
    ang = np.linspace(0, 2 * np.pi, 7)[:-1]; pts = [(540 + 170 * np.cos(a), 690 + 130 * np.sin(a)) for a in ang] + [(540, 690)]
    E_ = [(6, i) for i in range(6)] + [(i, (i + 1) % 6) for i in range(0, 6, 2)]
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
    for a, b in E_: ld.line((pts[a], pts[b]), fill=GOLD + (140,), width=4)
    for x, y_ in pts: ld.ellipse((x - 30, y_ - 30, x + 30, y_ + 30), fill=GOLD + (130,))
    img.alpha_composite(lay.filter(ImageFilter.GaussianBlur(12))); img.alpha_composite(lay); d_ = ImageDraw.Draw(img, "RGBA")
    for x, y_ in pts: d_.ellipse((x - 20, y_ - 20, x + 20, y_ + 20), fill=GOLD, outline=(255, 226, 160), width=2)
    tl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ImageDraw.Draw(tl).text((W / 2, 940), "Redes Neuronales", font=fBrand, fill=GOLD, anchor="mm")
    img.alpha_composite(tl.filter(ImageFilter.GaussianBlur(10))); img.alpha_composite(tl); d_ = ImageDraw.Draw(img, "RGBA")
    d_.text((W / 2, 1030), "Ep. 7 · Redes neuronales de grafos", font=fSub, fill=WHITE, anchor="mm")
    R.panel(d_, (200, 1120, 880, 1380))
    d_.text((W / 2, 1180), "PRÓXIMO EPISODIO: DBN", font=fFollow, fill=GOLD, anchor="mm")
    d_.text((W / 2, 1250), "sígueme para no perdértelo", font=R.fMono, fill=GREY, anchor="mm")
    d_.text((W / 2, 1318), HANDLE, font=fHandle, fill=WHITE, anchor="mm")
    return img

def audio():
    sr = 44100; n_ = int(N / FPS * sr); t = np.arange(n_) / sr; a = np.zeros(n_)
    env = np.clip(t / 1.2, 0, 1) * np.clip((N / FPS - t) / 1.0, 0, 1)
    for fr in (110, 164.8, 220, 246.9): a += 0.04 * env * np.sin(2 * np.pi * fr * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.3 * t))
    rr = np.random.default_rng(1)
    for s in np.arange(0.3, TRAIN_F / FPS, 0.18):     # soft "messages" between neighbours
        i0 = int(s * sr); tt = np.arange(int(0.25 * sr)) / sr; fr = rr.choice([659.3, 740, 880, 987.8])
        a[i0:i0 + len(tt)] += 0.03 * min(1, s / 4) * np.exp(-tt * 14) * np.sin(2 * np.pi * fr * tt)
    i0 = int((TRAIN_F + HOLD_F) / FPS * sr); tt = np.arange(n_ - i0) / sr
    for fr in (880, 1318.5, 1760): a[i0:] += 0.08 * np.exp(-tt * 2.5) * np.sin(2 * np.pi * fr * tt)
    a = a / np.abs(a).max() * 0.8
    with wave.open("work/gnn.wav", "w") as wv:
        wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(sr); wv.writeframes((a * 32767).astype(np.int16).tobytes())

if __name__ == "__main__":
    import sys
    END = end_card()
    def frame(f):
        if f < TRAIN_F: return scene(f, kidx(f))
        if f < TRAIN_F + HOLD_F: return scene(f, keys[-1])
        return Image.blend(scene(TRAIN_F + HOLD_F - 1, keys[-1]), END, min(1, (f - TRAIN_F - HOLD_F) / 14))
    if len(sys.argv) > 1:
        for f in map(int, sys.argv[1:]): frame(f).convert("RGB").save(f"work/n{f}.png")
        sys.exit()
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "work/gnn.mp4"], stdin=subprocess.PIPE)
    for f in range(N): p.stdin.write(frame(f).convert("RGB").tobytes())
    p.stdin.close(); p.wait(); audio()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "work/gnn.mp4", "-i", "work/gnn.wav", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "160k", "-shortest", "out/ep7_gnn.mp4"])
