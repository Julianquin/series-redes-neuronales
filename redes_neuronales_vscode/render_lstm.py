import numpy as np, pickle, subprocess, wave
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import render_mlp as R

W, H, FPS = 1080, 1920, 30
SEG, NCK = 78, 5
TRAIN_F, HOLD_F, END_F = SEG * NCK, 24, 66
N = TRAIN_F + HOLD_F + END_F
GOLD, BLUE, WHITE, GREY, EDGE, PANEL = R.GOLD, R.BLUE, R.WHITE, R.GREY, R.EDGE, R.PANEL
HANDLE = R.HANDLE
run = pickle.load(open("work/run_lstm.pkl", "rb"))
L, gens, CK, SEED, V, Hd = run["L"], run["gens"], run["CK"], run["SEED"], run["V"], run["Hd"]
E = len(L)
Ls = np.copy(L)
for i in range(1, E): Ls[i] = 0.9 * Ls[i - 1] + 0.1 * L[i]

fTxt = R.font(R.JB, 40, 500); fH = R.font(R.JB, 24, 500); fCk = R.font(R.JB, 24, 600); fTop = R.font(R.JB, 32, 600)
ease = lambda t: 0 if t <= 0 else 1 if t >= 1 else t * t * (3 - 2 * t)
lerp = lambda a, b, t: tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

def header(d):
    d.text((W / 2, 112), f"REDES NEURONALES · EP. 3 · {HANDLE}", font=R.fMonoS, fill=(70, 84, 98), anchor="mm")
    d.text((W / 2, 205), "¿puede una red neuronal", font=R.fTitle, fill=GOLD, anchor="mm")
    pill = "escribir letra a letra"
    tb = d.textbbox((0, 0), pill, font=R.fPill); tw = tb[2] - tb[0]
    x0 = W / 2 - (tw + 60) / 2 - 22; y0 = 262
    d.rounded_rectangle((x0, y0, x0 + tw + 60, y0 + 86), radius=12, fill=(22, 30, 40), outline=(62, 78, 96), width=2)
    d.text((x0 + 30 + tw / 2, y0 + 43), pill, font=R.fPill, fill=WHITE, anchor="mm")
    d.text((x0 + tw + 60 + 34, y0 + 42), "?", font=R.fQ, fill=GOLD, anchor="mm")

LX0, LX1, LY0, LY1 = 150, 990, 1560, 1765
lmin, lmax = np.log10(0.05), np.log10(4.5)
def lpx(it, v): return LX0 + (it - 1) / (E - 1) * (LX1 - LX0), LY1 - (np.log10(max(v, 1e-3)) - lmin) / (lmax - lmin) * (LY1 - LY0)
def loss_panel(d, upto, ck_i):
    R.panel(d, (60, 1440, 1020, 1840))
    d.text((90, 1478), "cross-entropy por letra (escala log)", font=R.fMono, fill=GREY, anchor="lm")
    d.line((800, 1478, 830, 1478), fill=BLUE, width=4); d.text((842, 1478), f"{Ls[upto - 1]:.3f}", font=R.fMonoS, fill=WHITE, anchor="lm")
    for v, lab in [(1, "1"), (0.1, "0.1")]:
        _, y = lpx(1, v); d.line((LX0, y, LX1, y), fill=(26, 36, 46), width=1); d.text((LX0 - 14, y), lab, font=R.fMonoS, fill=GREY, anchor="rm")
    d.line((LX0, LY1, LX1, LY1), fill=EDGE, width=2)
    for e in (0, 100, 200, 300, 400, 500):
        x = LX0 + e / E * (LX1 - LX0); d.text((min(x, LX1 - 20), LY1 + 26), str(e), font=R.fMonoS, fill=GREY, anchor="mm")
    d.text(((LX0 + LX1) / 2, LY1 + 56), "iteración", font=R.fMonoS, fill=GREY, anchor="mm")
    pts = [lpx(i, Ls[i - 1]) for i in range(1, upto + 1)]
    if len(pts) > 1: d.line(pts, fill=BLUE, width=3, joint="curve")
    for j, ck in enumerate(CK[:ck_i + 1]):
        if ck <= upto:
            x, y = lpx(ck, Ls[ck - 1]); col = GOLD if j == ck_i else GREY
            d.ellipse((x - 8, y - 8, x + 8, y + 8), fill=col, outline=(8, 12, 16), width=2)
    x, y = pts[-1]; d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=BLUE)

def signed_cells(d, vec, x0, y0, cols, cell, scale):
    base = np.array((18, 26, 35), np.float32)
    for j, val in enumerate(vec):
        r, c = divmod(j, cols); v = float(np.clip(val / scale, -1, 1))
        tgt = np.array(GOLD if v > 0 else BLUE, np.float32); col = tuple((base + (tgt - base) * abs(v) ** 0.8).astype(int))
        d.rectangle((x0 + c * cell + 1, y0 + r * cell + 1, x0 + (c + 1) * cell - 2, y0 + (r + 1) * cell - 2), fill=col)

CPL = 36; CW = 24.0; TX0, TY0, LH = 108, 572, 57
def text_layout(s):
    pos = []; x = y = 0
    words = []; cur = ""
    for ch in s:
        cur += ch
        if ch == " ": words.append(cur); cur = ""
    words.append(cur)
    for w_ in words:
        if x + len(w_.rstrip()) > CPL: x = 0; y += 1
        for ch in w_:
            if x >= CPL and ch != " ": x = 0; y += 1
            pos.append((x, y)); x += 1
    return pos

def train_frame(f):
    ci, lf = divmod(f, SEG); ck = CK[ci]; prev = CK[ci - 1] if ci else 1
    g = gens[ck]
    upto = int(prev + (ck - prev) * ease(lf / 12))
    ntyped = int(np.clip((lf - 8) * 2.6, 0, len(g)))
    img = R.BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA"); header(d)
    R.panel(d, (60, 380, 1020, 1410))
    d.text((92, 425), f"iter {ck} / {E}", font=R.fMono, fill=GREY, anchor="lm")
    d.text((988, 425), f"LSTM({Hd}) · por caracteres · {V} símbolos", font=R.fMono, fill=GREY, anchor="rm")
    # checkpoint pills
    for j, c in enumerate(CK):
        x = 100 + j * 180; on = j == ci; done = j < ci
        d.rounded_rectangle((x, 462, x + 160, 504), radius=21, fill=(40, 32, 14) if on else (18, 25, 33),
                            outline=GOLD if on else (EDGE if not done else (80, 96, 112)), width=2)
        d.text((x + 80, 483), f"iter {c}", font=fCk, fill=GOLD if on else (GREY if done else (70, 84, 98)), anchor="mm")
    d.text((95, 540), "texto generado (semilla en gris)", font=fH, fill=GREY, anchor="lm")
    full = SEED + "".join(o["ch"] for o in g)
    pos = text_layout(full)
    for k in range(len(SEED) + ntyped):
        cx, cy = pos[k]; x = TX0 + cx * CW; y = TY0 + cy * LH
        if k < len(SEED): col = (92, 106, 120)
        else:
            p = g[k - len(SEED)]["prob"]; col = lerp((214, 96, 96), WHITE, min(1, p / 0.9)) if p < 0.9 else WHITE
        d.text((x, y), full[k], font=fTxt, fill=col, anchor="lm")
    if ntyped < len(g) or (lf // 8) % 2 == 0:
        k = len(SEED) + ntyped; cx, cy = pos[min(k, len(pos) - 1)] if k < len(pos) else (pos[-1][0] + 1, pos[-1][1])
        d.rectangle((TX0 + cx * CW, TY0 + cy * LH + 22, TX0 + cx * CW + 20, TY0 + cy * LH + 27), fill=GOLD)
    d.text((988, 540), "confianza:", font=fH, fill=GREY, anchor="rm") if False else None
    # legend for confidence
    d.text((710, 540), "baja", font=fH, fill=(214, 96, 96), anchor="lm"); d.text((800, 540), "→", font=fH, fill=GREY, anchor="lm"); d.text((840, 540), "alta", font=fH, fill=WHITE, anchor="lm")
    d.line((80, 900, 1000, 900), fill=(26, 36, 46), width=1)
    # next-letter distribution
    st = g[max(0, ntyped - 1)] if ntyped else None
    d.text((95, 930), "¿qué letra viene ahora?", font=fH, fill=GREY, anchor="lm")
    if st:
        for r, (ch, p) in enumerate(st["top"]):
            y = 965 + r * 46; lab = "_" if ch == " " else ch
            d.rounded_rectangle((100, y, 142, y + 38), radius=6, fill=(22, 30, 40), outline=EDGE)
            d.text((121, y + 19), lab, font=fTop, fill=WHITE, anchor="mm")
            d.rounded_rectangle((160, y + 6, 160 + max(4, p * 720), y + 32), radius=5, fill=GOLD if r == 0 else (60, 74, 88))
            d.text((160 + max(4, p * 720) + 14, y + 19), f"{p * 100:.0f}%", font=R.fMonoS, fill=WHITE if r == 0 else GREY, anchor="lm")
    d.line((80, 1203, 1000, 1203), fill=(26, 36, 46), width=1)
    d.text((95, 1228), f"memoria · estado oculto h ({Hd} valores)", font=fH, fill=GREY, anchor="lm")
    hv = st["h"] if st else np.zeros(Hd)
    signed_cells(d, hv, 108, 1252, 24, 36, 0.8)
    loss_panel(d, upto, ci)
    return img

fSub = R.font(R.SG, 44, 500); fFollow = R.font(R.JB, 26, 600); fHandle = R.font(R.SG, 50, 700); fBrand = R.font(R.SG, 96, 700)
def end_card():
    img = R.BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    glowl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(glowl)
    xs = [270, 450, 630, 810]; y = 690
    for x in xs: gd.rounded_rectangle((x - 55, y - 55, x + 55, y + 55), radius=16, fill=GOLD + (120,))
    img.alpha_composite(glowl.filter(ImageFilter.GaussianBlur(14))); d = ImageDraw.Draw(img, "RGBA")
    letters = "hola"
    for j, x in enumerate(xs):
        d.line((x, y + 120, x, y + 58), fill=GREY, width=3)
        d.text((x, y + 140), letters[j], font=R.fMono, fill=GREY, anchor="mm")
        d.line((x, y - 58, x, y - 110), fill=GOLD, width=3); d.polygon([(x - 8, y - 104), (x + 8, y - 104), (x, y - 118)], fill=GOLD)
        if j < 3:
            d.line((x + 58, y, xs[j + 1] - 62, y), fill=GOLD, width=4)
            d.polygon([(xs[j + 1] - 70, y - 9), (xs[j + 1] - 70, y + 9), (xs[j + 1] - 57, y)], fill=GOLD)
        d.rounded_rectangle((x - 50, y - 50, x + 50, y + 50), radius=14, fill=(20, 26, 34), outline=GOLD, width=3)
        d.text((x, y), "h", font=R.fPill, fill=GOLD, anchor="mm")
    tl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ImageDraw.Draw(tl).text((W / 2, 940), "Redes Neuronales", font=fBrand, fill=GOLD, anchor="mm")
    img.alpha_composite(tl.filter(ImageFilter.GaussianBlur(10))); img.alpha_composite(tl); d = ImageDraw.Draw(img, "RGBA")
    d.text((W / 2, 1030), "Ep. 3 · Redes recurrentes (LSTM)", font=fSub, fill=WHITE, anchor="mm")
    R.panel(d, (200, 1120, 880, 1380))
    d.text((W / 2, 1180), "PRÓXIMO EPISODIO: TRANSFORMERS", font=fFollow, fill=GOLD, anchor="mm")
    d.text((W / 2, 1250), "sígueme para no perdértelo", font=R.fMono, fill=GREY, anchor="mm")
    d.text((W / 2, 1318), HANDLE, font=fHandle, fill=WHITE, anchor="mm")
    return img

def audio():
    sr = 44100; n = int(N / FPS * sr); t = np.arange(n) / sr; a = np.zeros(n)
    env = np.clip(t / 1.2, 0, 1) * np.clip((N / FPS - t) / 1.0, 0, 1)
    for fr in (98, 146.8, 196, 246.9): a += 0.04 * env * np.sin(2 * np.pi * fr * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.2 * t))
    rng = np.random.default_rng(0); click = rng.normal(0, 1, int(0.012 * sr)) * np.exp(-np.arange(int(0.012 * sr)) / (0.002 * sr))
    for f in range(TRAIN_F):
        ci, lf = divmod(f, SEG); g = gens[CK[ci]]
        typed = int(np.clip((lf - 8) * 2.6, 0, len(g))); prevt = int(np.clip((lf - 9) * 2.6, 0, len(g)))
        for k in range(typed - prevt):
            i0 = int((f + k / 3) / FPS * sr); a[i0:i0 + len(click)] += 0.12 * click
        if lf == 0 and ci:
            i0 = int(f / FPS * sr); tt = np.arange(int(0.6 * sr)) / sr
            a[i0:i0 + len(tt)] += 0.12 * np.exp(-tt * 6) * np.sin(2 * np.pi * (440 + 110 * ci) * tt)
    i0 = int((TRAIN_F + HOLD_F) / FPS * sr); tt = np.arange(n - i0) / sr
    for fr in (880, 1318.5, 1760): a[i0:] += 0.08 * np.exp(-tt * 2.5) * np.sin(2 * np.pi * fr * tt)
    a = a / np.abs(a).max() * 0.8
    with wave.open("work/lstm.wav", "w") as wv:
        wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(sr); wv.writeframes((a * 32767).astype(np.int16).tobytes())

if __name__ == "__main__":
    import sys
    END = end_card(); LAST = train_frame(TRAIN_F - 1)
    def frame(f):
        if f < TRAIN_F: return train_frame(f)
        if f < TRAIN_F + HOLD_F: return LAST
        return Image.blend(LAST, END, min(1, (f - TRAIN_F - HOLD_F) / 14))
    if len(sys.argv) > 1:
        for f in map(int, sys.argv[1:]): frame(f).convert("RGB").save(f"work/l{f}.png")
        sys.exit()
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "work/lstm.mp4"], stdin=subprocess.PIPE)
    for f in range(N): p.stdin.write(frame(f).convert("RGB").tobytes())
    p.stdin.close(); p.wait(); audio()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "work/lstm.mp4", "-i", "work/lstm.wav", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "160k", "-shortest", "out/ep3_lstm_letra_a_letra.mp4"])
