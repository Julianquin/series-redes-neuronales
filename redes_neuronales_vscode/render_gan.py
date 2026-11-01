import numpy as np, pickle, subprocess, wave
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import render_mlp as R
from train_gan import fwd, bwd
sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))

W, H, FPS = 1080, 1920, 30
TRAIN_F, HOLD_F, END_F = 390, 24, 66
N = TRAIN_F + HOLD_F + END_F
GOLD, BLUE, WHITE, GREY, EDGE, PANEL = R.GOLD, R.BLUE, R.WHITE, R.GREY, R.EDGE, R.PANEL
REDC = (236, 96, 96)
HANDLE = R.HANDLE
run = pickle.load(open("work/run_gan.pkl", "rb"))
L, snaps, Zfix, Rfix = run["L"], run["snaps"], run["Zfix"], run["Rfix"]
keys = sorted(snaps); E = len(L)
Ls = L.copy()
for i in range(1, E): Ls[i] = 0.97 * Ls[i - 1] + 0.03 * L[i]
fH = R.font(R.JB, 24, 500); fCard = R.font(R.SG, 30, 700); fCardS = R.font(R.JB, 20, 400)

def header(d_):
    d_.text((W / 2, 112), f"REDES NEURONALES · EP. 6 · {HANDLE}", font=R.fMonoS, fill=(70, 84, 98), anchor="mm")
    d_.text((W / 2, 205), "¿puede una red neuronal", font=R.fTitle, fill=GOLD, anchor="mm")
    pill = "aprender a falsificar"
    tb = d_.textbbox((0, 0), pill, font=R.fPill); tw = tb[2] - tb[0]
    x0 = W / 2 - (tw + 60) / 2 - 22; y0 = 262
    d_.rounded_rectangle((x0, y0, x0 + tw + 60, y0 + 86), radius=12, fill=(22, 30, 40), outline=(62, 78, 96), width=2)
    d_.text((x0 + 30 + tw / 2, y0 + 43), pill, font=R.fPill, fill=WHITE, anchor="mm")
    d_.text((x0 + tw + 60 + 34, y0 + 42), "?", font=R.fQ, fill=GOLD, anchor="mm")

PX0, PY0, PS, LIM = 130, 560, 820, 3.0
G_ = 180; gx = np.linspace(-LIM, LIM, G_); GX, GY = np.meshgrid(gx, -gx); GRID = np.c_[GX.ravel(), GY.ravel()]
to = lambda p: (PX0 + (p[0] + LIM) / (2 * LIM) * PS, PY0 + (LIM - p[1]) / (2 * LIM) * PS)
ARR = np.random.default_rng(0).choice(len(Zfix), 70, replace=False)

def dmap(D):
    p = sig(fwd(D, GRID)[-1][:, 0]).reshape(G_, G_).astype(np.float32)
    pu = np.array(Image.fromarray(p, "F").resize((PS, PS), Image.BILINEAR))[..., None]
    base = np.array(PANEL, np.float32)
    tint = np.array(BLUE, np.float32) * pu + np.array(REDC, np.float32) * (1 - pu)
    rgb = base + (tint - base) * 0.30 * np.abs(2 * pu - 1) ** 0.7
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).convert("RGBA")

LX0, LX1, LY0, LY1 = 150, 990, 1560, 1765
ymin, ymax = 0.3, 1.5
def lpx(it, v): return LX0 + (it - 1) / (E - 1) * (LX1 - LX0), LY1 - (np.clip(v, ymin, ymax) - ymin) / (ymax - ymin) * (LY1 - LY0)
def loss_panel(d_, k):
    R.panel(d_, (60, 1440, 1020, 1840))
    d_.text((90, 1478), "el duelo: pérdida de cada red", font=R.fMono, fill=GREY, anchor="lm")
    d_.line((700, 1470, 730, 1470), fill=GOLD, width=4); d_.text((742, 1470), f"generador {Ls[k - 1, 1]:.3f}", font=R.fMonoS, fill=WHITE, anchor="lm")
    d_.line((700, 1502, 730, 1502), fill=BLUE, width=4); d_.text((742, 1502), f"discriminador {Ls[k - 1, 0]:.3f}", font=R.fMonoS, fill=WHITE, anchor="lm")
    for v in (0.5, 1.0):
        _, yy = lpx(1, v); d_.line((LX0, yy, LX1, yy), fill=(26, 36, 46), width=1); d_.text((LX0 - 14, yy), f"{v:.1f}", font=R.fMonoS, fill=GREY, anchor="rm")
    _, yy = lpx(1, np.log(2))
    for x in range(LX0, LX1, 18): d_.line((x, yy, x + 9, yy), fill=(90, 104, 118), width=1)
    d_.text((LX1, yy - 14), "equilibrio (ln 2)", font=R.fMonoS, fill=(90, 104, 118), anchor="rm")
    d_.line((LX0, LY1, LX1, LY1), fill=EDGE, width=2)
    for e in (0, 1000, 2000, 3000, 4000):
        x = LX0 + e / E * (LX1 - LX0); d_.text((min(x, LX1 - 26), LY1 + 26), f"{e:,}", font=R.fMonoS, fill=GREY, anchor="mm")
    d_.text(((LX0 + LX1) / 2, LY1 + 56), "iteración", font=R.fMonoS, fill=GREY, anchor="mm")
    step = max(1, k // 500)
    for col, j in ((BLUE, 0), (GOLD, 1)):
        pts = [lpx(e, Ls[e - 1, j]) for e in range(1, k + 1, step)] + [lpx(k, Ls[k - 1, j])]
        if len(pts) > 1: d_.line(pts, fill=col, width=3, joint="curve")
        d_.ellipse((pts[-1][0] - 6, pts[-1][1] - 6, pts[-1][0] + 6, pts[-1][1] + 6), fill=col)

def train_frame(fi):
    k = keys[min(len(keys) - 1, int(round(fi / (TRAIN_F - 1) * (len(keys) - 1))))]
    G, D = snaps[k]
    img = R.BGIMG.copy(); d_ = ImageDraw.Draw(img, "RGBA"); header(d_)
    R.panel(d_, (60, 380, 1020, 1410))
    d_.text((92, 425), f"iter {k:,} / {E:,}", font=R.fMono, fill=GREY, anchor="lm")
    d_.text((988, 425), "G y D: 2 capas de 128", font=R.fMono, fill=GREY, anchor="rm")
    # role cards
    for x0, title, sub, col in ((92, "GENERADOR", "crea puntos falsos", GOLD), (580, "DISCRIMINADOR", "¿real o falso?", BLUE)):
        d_.rounded_rectangle((x0, 460, x0 + 408, 540), radius=12, fill=(18, 25, 33), outline=col, width=2)
        d_.text((x0 + 24, 488), title, font=fCard, fill=col, anchor="lm"); d_.text((x0 + 24, 520), sub, font=fCardS, fill=GREY, anchor="lm")
    d_.text((540, 500), "vs", font=fCard, fill=WHITE, anchor="mm")
    img.alpha_composite(dmap(D), (PX0, PY0)); d_ = ImageDraw.Draw(img, "RGBA")
    d_.rectangle((PX0, PY0, PX0 + PS, PY0 + PS), outline=EDGE)
    for p in Rfix: px, py = to(p); d_.ellipse((px - 3, py - 3, px + 3, py + 3), fill=WHITE + (200,))
    hg = fwd(G, Zfix); fake = hg[-1]
    hd = fwd(D, fake[ARR]); _, gx_ = bwd(D, hd, np.ones((len(ARR), 1)))
    for p, gvec in zip(fake[ARR], gx_):
        n = np.linalg.norm(gvec)
        if n < 1e-6: continue
        q = p + gvec / n * 0.28; a, b = to(p), to(q)
        if PX0 < a[0] < PX0 + PS and PY0 < a[1] < PY0 + PS: d_.line((a, b), fill=GOLD + (170,), width=2)
    for p in fake:
        px, py = to(p)
        if PX0 + 4 < px < PX0 + PS - 4 and PY0 + 4 < py < PY0 + PS - 4: d_.ellipse((px - 5, py - 5, px + 5, py + 5), fill=GOLD, outline=(8, 12, 16), width=1)
    acc = np.mean(L[max(0, k - 30):k, 2])
    d_.rounded_rectangle((PX0 + PS - 330, PY0 + 14, PX0 + PS - 14, PY0 + 60), radius=10, fill=(8, 12, 16, 200))
    d_.text((PX0 + PS - 28, PY0 + 37), f"el detective acierta {acc * 100:.0f}%", font=R.fMonoS, fill=WHITE, anchor="rm")
    ly = 1392
    d_.ellipse((130, ly - 5, 140, ly + 5), fill=WHITE); d_.text((150, ly), "datos reales", font=fCardS, fill=GREY, anchor="lm")
    d_.ellipse((330, ly - 7, 344, ly + 7), fill=GOLD); d_.text((354, ly), "falsos (con su próximo paso)", font=fCardS, fill=GREY, anchor="lm")
    d_.rectangle((720, ly - 8, 740, ly + 8), fill=(40, 62, 100)); d_.text((748, ly), "\"real\"", font=fCardS, fill=GREY, anchor="lm")
    d_.rectangle((840, ly - 8, 860, ly + 8), fill=(100, 40, 44)); d_.text((868, ly), "\"falso\"", font=fCardS, fill=GREY, anchor="lm")
    loss_panel(d_, k)
    return img

fSub = R.font(R.SG, 44, 500); fFollow = R.font(R.JB, 26, 600); fHandle = R.font(R.SG, 50, 700); fBrand = R.font(R.SG, 96, 700)
def end_card():
    img = R.BGIMG.copy(); d_ = ImageDraw.Draw(img, "RGBA")
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
    ld.rounded_rectangle((200, 620, 400, 760), radius=20, fill=GOLD + (120,)); ld.rounded_rectangle((680, 620, 880, 760), radius=20, fill=BLUE + (120,))
    img.alpha_composite(lay.filter(ImageFilter.GaussianBlur(16))); d_ = ImageDraw.Draw(img, "RGBA")
    d_.rounded_rectangle((200, 620, 400, 760), radius=20, fill=(20, 26, 34), outline=GOLD, width=3); d_.text((300, 690), "G", font=fBrand, fill=GOLD, anchor="mm")
    d_.rounded_rectangle((680, 620, 880, 760), radius=20, fill=(20, 26, 34), outline=BLUE, width=3); d_.text((780, 690), "D", font=fBrand, fill=BLUE, anchor="mm")
    d_.line((410, 665, 660, 665), fill=GOLD, width=4); d_.polygon([(660, 655), (660, 675), (676, 665)], fill=GOLD)
    d_.line((670, 715, 420, 715), fill=BLUE, width=4); d_.polygon([(420, 705), (420, 725), (404, 715)], fill=BLUE)
    d_.text((540, 645), "falsos", font=fCardS, fill=GREY, anchor="mm"); d_.text((540, 740), "feedback", font=fCardS, fill=GREY, anchor="mm")
    tl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ImageDraw.Draw(tl).text((W / 2, 940), "Redes Neuronales", font=fBrand, fill=GOLD, anchor="mm")
    img.alpha_composite(tl.filter(ImageFilter.GaussianBlur(10))); img.alpha_composite(tl); d_ = ImageDraw.Draw(img, "RGBA")
    d_.text((W / 2, 1030), "Ep. 6 · Redes generativas (GAN)", font=fSub, fill=WHITE, anchor="mm")
    R.panel(d_, (200, 1120, 880, 1380))
    d_.text((W / 2, 1180), "PRÓXIMO EPISODIO: GNN", font=fFollow, fill=GOLD, anchor="mm")
    d_.text((W / 2, 1250), "sígueme para no perdértelo", font=R.fMono, fill=GREY, anchor="mm")
    d_.text((W / 2, 1318), HANDLE, font=fHandle, fill=WHITE, anchor="mm")
    return img

def audio():
    sr = 44100; n = int(N / FPS * sr); t = np.arange(n) / sr; a = np.zeros(n)
    env = np.clip(t / 1.2, 0, 1) * np.clip((N / FPS - t) / 1.0, 0, 1)
    # two voices "arguing": alternating tremolos that settle into the same chord
    tr = TRAIN_F / FPS; mix = np.clip(t / tr, 0, 1)
    for fr in (110, 164.8): a += 0.05 * env * np.sin(2 * np.pi * fr * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t))
    for fr in (116.5, 155.6): a += 0.04 * env * (1 - mix) * np.sin(2 * np.pi * fr * t) * (0.6 - 0.4 * np.sin(2 * np.pi * 2 * t))
    for fr in (220, 277.2): a += 0.04 * env * mix * np.sin(2 * np.pi * fr * t)
    i0 = int((TRAIN_F + HOLD_F) / FPS * sr); tt = np.arange(n - i0) / sr
    for fr in (880, 1318.5, 1760): a[i0:] += 0.08 * np.exp(-tt * 2.5) * np.sin(2 * np.pi * fr * tt)
    a = a / np.abs(a).max() * 0.8
    with wave.open("work/gan.wav", "w") as wv:
        wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(sr); wv.writeframes((a * 32767).astype(np.int16).tobytes())

if __name__ == "__main__":
    import sys
    END = end_card(); LAST = train_frame(TRAIN_F - 1)
    def frame(f):
        if f < TRAIN_F: return train_frame(f)
        if f < TRAIN_F + HOLD_F: return LAST
        return Image.blend(LAST, END, min(1, (f - TRAIN_F - HOLD_F) / 14))
    if len(sys.argv) > 1:
        for f in map(int, sys.argv[1:]): frame(f).convert("RGB").save(f"work/g{f}.png")
        sys.exit()
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "work/gan.mp4"], stdin=subprocess.PIPE)
    for f in range(N): p.stdin.write(frame(f).convert("RGB").tobytes())
    p.stdin.close(); p.wait(); audio()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "work/gan.mp4", "-i", "work/gan.wav", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "160k", "-shortest", "out/ep6_gan.mp4"])
