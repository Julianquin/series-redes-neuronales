import numpy as np, pickle, subprocess, wave
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import render_mlp as R

W, H, FPS = 1080, 1920, 30
TRAIN_F, HOLD_F, END_F = 390, 24, 66
N = TRAIN_F + HOLD_F + END_F
GOLD, BLUE, WHITE, GREY, EDGE, PANEL = R.GOLD, R.BLUE, R.WHITE, R.GREY, R.EDGE, R.PANEL
GREEN, REDC = (96, 206, 140), (236, 96, 96)
HANDLE = R.HANDLE
run = pickle.load(open("work/run_attn.pkl", "rb"))
L, snaps, T, d = run["L"], run["snaps"], run["T"], run["d"]
keys = sorted(snaps); E = len(L)
WORD = "aprender"; tok = np.array([[ord(c) - 97 for c in WORD]]); TARGET = WORD[::-1]

def fwd(P, tok):
    x = P['E'][tok] + P['Pe']; Q, K, Vv = x @ P['Wq'], x @ P['Wk'], x @ P['Wv']
    S = Q @ K.transpose(0, 2, 1) * d ** -0.5; S -= S.max(-1, keepdims=True); A = np.exp(S); A /= A.sum(-1, keepdims=True)
    lg = (A @ Vv) @ P['Wo'] + P['bo']; lg -= lg.max(-1, keepdims=True); p = np.exp(lg); p /= p.sum(-1, keepdims=True)
    return A[0], p[0]

fBox = R.font(R.SG, 54, 700); fH = R.font(R.JB, 24, 500); fLab = R.font(R.JB, 26, 600); fCell = R.font(R.JB, 20, 600)
fStat = R.font(R.SG, 64, 700)
lerp = lambda a, b, t: tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

def header(d_):
    d_.text((W / 2, 112), f"REDES NEURONALES · EP. 4 · {HANDLE}", font=R.fMonoS, fill=(70, 84, 98), anchor="mm")
    d_.text((W / 2, 205), "¿puede una red neuronal", font=R.fTitle, fill=GOLD, anchor="mm")
    pill = "aprender dónde mirar"
    tb = d_.textbbox((0, 0), pill, font=R.fPill); tw = tb[2] - tb[0]
    x0 = W / 2 - (tw + 60) / 2 - 22; y0 = 262
    d_.rounded_rectangle((x0, y0, x0 + tw + 60, y0 + 86), radius=12, fill=(22, 30, 40), outline=(62, 78, 96), width=2)
    d_.text((x0 + 30 + tw / 2, y0 + 43), pill, font=R.fPill, fill=WHITE, anchor="mm")
    d_.text((x0 + tw + 60 + 34, y0 + 42), "?", font=R.fQ, fill=GOLD, anchor="mm")

BW, GAP, BX0 = 90, 18, 117
bx = lambda i: BX0 + i * (BW + GAP)
IN_Y, OUT_Y = 512, 712
GX0, GY0, CS = 300, 905, 56

def bezier(p0, p1, p2, p3, n=28):
    t = np.linspace(0, 1, n)[:, None]
    pts = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
    return [tuple(p) for p in pts]

LX0, LX1, LY0, LY1 = 150, 990, 1560, 1765
lmin, lmax = np.log10(0.015), np.log10(4.5)
def lpx(it, v): return LX0 + (it - 1) / (E - 1) * (LX1 - LX0), LY1 - (np.log10(max(v, 1e-4)) - lmin) / (lmax - lmin) * (LY1 - LY0)
def loss_panel(d_, k):
    R.panel(d_, (60, 1440, 1020, 1840))
    d_.text((90, 1478), "cross-entropy (escala log)", font=R.fMono, fill=GREY, anchor="lm")
    d_.line((800, 1478, 830, 1478), fill=BLUE, width=4); d_.text((842, 1478), f"{L[k - 1, 0]:.4f}", font=R.fMonoS, fill=WHITE, anchor="lm")
    for v, lab in [(1, "1"), (0.1, "0.1")]:
        _, y = lpx(1, v); d_.line((LX0, y, LX1, y), fill=(26, 36, 46), width=1); d_.text((LX0 - 14, y), lab, font=R.fMonoS, fill=GREY, anchor="rm")
    d_.line((LX0, LY1, LX1, LY1), fill=EDGE, width=2)
    for e in (0, 100, 200, 300):
        x = LX0 + e / E * (LX1 - LX0); d_.text((min(x, LX1 - 20), LY1 + 26), str(e), font=R.fMonoS, fill=GREY, anchor="mm")
    d_.text(((LX0 + LX1) / 2, LY1 + 56), "iteración", font=R.fMonoS, fill=GREY, anchor="mm")
    pts = [lpx(i, L[i - 1, 0]) for i in range(1, k + 1)]
    if len(pts) > 1: d_.line(pts, fill=BLUE, width=3, joint="curve")
    d_.ellipse((pts[-1][0] - 6, pts[-1][1] - 6, pts[-1][0] + 6, pts[-1][1] + 6), fill=BLUE)

def train_frame(fi):
    k = keys[min(len(keys) - 1, int(round(fi / (TRAIN_F - 1) * (len(keys) - 1))))]
    A, p = fwd(snaps[k], tok); pred = p.argmax(-1)
    q = (fi // 14) % T
    img = R.BGIMG.copy(); d_ = ImageDraw.Draw(img, "RGBA"); header(d_)
    R.panel(d_, (60, 380, 1020, 1410))
    d_.text((92, 425), f"iter {k} / {E}", font=R.fMono, fill=GREY, anchor="lm")
    d_.text((988, 425), f"1 capa de self-attention · d={d}", font=R.fMono, fill=GREY, anchor="rm")
    d_.text((95, 482), "entrada", font=fH, fill=GREY, anchor="lm")
    # connections from the highlighted output to every input
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
    p3 = np.array([bx(q) + BW / 2, OUT_Y]); 
    for j in range(T):
        w = A[q, j]; p0 = np.array([bx(j) + BW / 2, IN_Y + BW])
        pts = bezier(p0, p0 + [0, 55], p3 - [0, 55], p3)
        ld.line(pts, fill=GOLD + (int(30 + 225 * w),), width=max(1, int(2 + 14 * w)), joint="curve")
    img.alpha_composite(lay); d_ = ImageDraw.Draw(img, "RGBA")
    for j, ch in enumerate(WORD):
        w = A[q, j]; fill = lerp((22, 30, 40), (90, 66, 20), w)
        d_.rounded_rectangle((bx(j), IN_Y, bx(j) + BW, IN_Y + BW), radius=12, fill=fill, outline=lerp(EDGE, GOLD, w), width=3)
        d_.text((bx(j) + BW / 2, IN_Y + BW / 2), ch.upper(), font=fBox, fill=WHITE, anchor="mm")
    for i in range(T):
        conf = p[i, pred[i]]; ok = pred[i] == ord(TARGET[i]) - 97; col = GREEN if ok else REDC
        d_.rounded_rectangle((bx(i), OUT_Y, bx(i) + BW, OUT_Y + BW), radius=12, fill=(22, 30, 40),
                             outline=GOLD if i == q else EDGE, width=4 if i == q else 2)
        d_.text((bx(i) + BW / 2, OUT_Y + BW / 2), chr(97 + pred[i]).upper(), font=fBox, fill=lerp((60, 72, 84), col, min(1, conf * 1.2)), anchor="mm")
    d_.text((95, OUT_Y + BW + 30), "salida: la palabra al revés", font=fH, fill=GREY, anchor="lm")
    d_.text((988, OUT_Y + BW + 30), f"objetivo: {TARGET.upper()}", font=fH, fill=GREY, anchor="rm")
    d_.line((80, 860, 1000, 860), fill=(26, 36, 46), width=1)
    d_.text((95, 882), "mapa de atención", font=fH, fill=GREY, anchor="lm")
    # heatmap
    for i in range(T):
        d_.text((GX0 - 20, GY0 + i * CS + CS / 2), chr(97 + pred[i]).upper(), font=fLab, fill=GOLD if i == q else GREY, anchor="rm")
        for j in range(T):
            w = A[i, j]; col = lerp((18, 26, 35), GOLD, w ** 0.7)
            d_.rectangle((GX0 + j * CS + 2, GY0 + i * CS + 2, GX0 + (j + 1) * CS - 2, GY0 + (i + 1) * CS - 2), fill=col)
            if i == q and w > 0.08: d_.text((GX0 + j * CS + CS / 2, GY0 + i * CS + CS / 2), f"{w * 100:.0f}", font=fCell, fill=(10, 12, 16) if w > 0.5 else WHITE, anchor="mm")
    for j, ch in enumerate(WORD):
        d_.text((GX0 + j * CS + CS / 2, GY0 + T * CS + 22), ch.upper(), font=fLab, fill=GREY, anchor="mm")
    d_.rectangle((GX0, GY0 + q * CS, GX0 + T * CS, GY0 + (q + 1) * CS), outline=GOLD, width=3)
    d_.text((GX0 + T * CS, GY0 + T * CS + 22), "  ← entrada", font=R.fMonoS, fill=(80, 94, 108), anchor="lm")
    # stats
    acc, word_acc = L[k - 1, 1], L[k - 1, 2]
    sx = 800
    d_.text((sx, 960), "letras bien", font=fH, fill=GREY, anchor="lm")
    d_.text((sx, 1010), f"{acc * 100:.0f}%", font=fStat, fill=GOLD if acc > 0.99 else WHITE, anchor="lm")
    d_.text((sx, 1110), "palabras", font=fH, fill=GREY, anchor="lm"); d_.text((sx, 1138), "perfectas", font=fH, fill=GREY, anchor="lm")
    d_.text((sx, 1190), f"{word_acc * 100:.0f}%", font=fStat, fill=GOLD if word_acc > 0.99 else WHITE, anchor="lm")
    d_.text((sx, 1262), "(500 palabras", font=R.fMonoS, fill=(80, 94, 108), anchor="lm"); d_.text((sx, 1288), " de prueba)", font=R.fMonoS, fill=(80, 94, 108), anchor="lm")
    loss_panel(d_, k)
    return img

fSub = R.font(R.SG, 44, 500); fFollow = R.font(R.JB, 26, 600); fHandle = R.font(R.SG, 50, 700); fBrand = R.font(R.SG, 96, 700)
def end_card():
    img = R.BGIMG.copy(); d_ = ImageDraw.Draw(img, "RGBA")
    xs = [220 + i * 160 for i in range(5)]; y = 760
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
    pairs = [(0, 2, .9), (0, 4, .5), (1, 3, .8), (2, 4, .7), (1, 4, .4), (0, 3, .3), (2, 3, .5)]
    for a, b, w in pairs:
        p0 = np.array([xs[a], y - 45]); p3 = np.array([xs[b], y - 45]); hgt = 70 + 45 * (b - a)
        ld.line(bezier(p0, p0 - [0, hgt], p3 - [0, hgt], p3, 40), fill=GOLD + (int(80 + 170 * w),), width=int(3 + 6 * w), joint="curve")
    img.alpha_composite(lay.filter(ImageFilter.GaussianBlur(6))); img.alpha_composite(lay); d_ = ImageDraw.Draw(img, "RGBA")
    for i, x in enumerate(xs):
        d_.rounded_rectangle((x - 45, y - 45, x + 45, y + 45), radius=12, fill=(20, 26, 34), outline=GOLD, width=3)
        d_.text((x, y), "QKVAO"[i] if False else "•", font=fBox, fill=GOLD, anchor="mm")
    tl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ImageDraw.Draw(tl).text((W / 2, 940), "Redes Neuronales", font=fBrand, fill=GOLD, anchor="mm")
    img.alpha_composite(tl.filter(ImageFilter.GaussianBlur(10))); img.alpha_composite(tl); d_ = ImageDraw.Draw(img, "RGBA")
    d_.text((W / 2, 1030), "Ep. 4 · Transformers y atención", font=fSub, fill=WHITE, anchor="mm")
    R.panel(d_, (200, 1120, 880, 1380))
    d_.text((W / 2, 1180), "PRÓXIMO EPISODIO: AUTOENCODERS", font=fFollow, fill=GOLD, anchor="mm")
    d_.text((W / 2, 1250), "sígueme para no perdértelo", font=R.fMono, fill=GREY, anchor="mm")
    d_.text((W / 2, 1318), HANDLE, font=fHandle, fill=WHITE, anchor="mm")
    return img

def audio():
    sr = 44100; n = int(N / FPS * sr); t = np.arange(n) / sr; a = np.zeros(n)
    env = np.clip(t / 1.2, 0, 1) * np.clip((N / FPS - t) / 1.0, 0, 1)
    for fr in (110, 138.6, 164.8, 220): a += 0.045 * env * np.sin(2 * np.pi * fr * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.25 * t))
    scale = [440, 493.9, 554.4, 659.3, 740, 880, 987.8, 1108.7]
    for f in range(0, TRAIN_F, 14):   # one soft note per highlighted query
        q = (f // 14) % T; i0 = int(f / FPS * sr); tt = np.arange(min(int(0.6 * sr), n - i0)) / sr
        a[i0:i0 + len(tt)] += 0.06 * (0.5 + f / TRAIN_F) * np.exp(-tt * 6) * np.sin(2 * np.pi * scale[q] * tt)
    i0 = int((TRAIN_F + HOLD_F) / FPS * sr); tt = np.arange(n - i0) / sr
    for fr in (880, 1318.5, 1760): a[i0:] += 0.08 * np.exp(-tt * 2.5) * np.sin(2 * np.pi * fr * tt)
    a = a / np.abs(a).max() * 0.8
    with wave.open("work/attn.wav", "w") as wv:
        wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(sr); wv.writeframes((a * 32767).astype(np.int16).tobytes())

if __name__ == "__main__":
    import sys
    END = end_card(); LAST = train_frame(TRAIN_F - 1)
    def frame(f):
        if f < TRAIN_F: return train_frame(f)
        if f < TRAIN_F + HOLD_F: return LAST
        return Image.blend(LAST, END, min(1, (f - TRAIN_F - HOLD_F) / 14))
    if len(sys.argv) > 1:
        for f in map(int, sys.argv[1:]): frame(f).convert("RGB").save(f"work/a{f}.png")
        sys.exit()
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "work/attn.mp4"], stdin=subprocess.PIPE)
    for f in range(N): p.stdin.write(frame(f).convert("RGB").tobytes())
    p.stdin.close(); p.wait(); audio()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "work/attn.mp4", "-i", "work/attn.wav", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "160k", "-shortest", "out/ep4_transformer_atencion.mp4"])
