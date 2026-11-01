import numpy as np, pickle, subprocess, wave
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import render_mlp as R
import train_dbn_ola as T
import repair_ola as RP

W, H, FPS = 1080, 1920, 30
F1, F2, F3 = 300, 90, 120
SHOW_F = F1 + F2 + F3; HOLD_F, END_F = 20, 66
N = SHOW_F + HOLD_F + END_F
GOLD, BLUE, WHITE, GREY, EDGE, PANEL = R.GOLD, R.BLUE, R.WHITE, R.GREY, R.EDGE, R.PANEL
GREEN = (96, 206, 140); PAPER = (240, 234, 220); INKC = (22, 26, 34)
HANDLE = R.HANDLE; P = T.PS; sig = T.sig
RUN = pickle.load(open("work/run_dbn_ola_v2.pkl", "rb"))
S1, S2, E1, E2 = RUN["S1"], RUN["S2"], RUN["E1"], RUN["E2"]; W1f = RUN["W1"]
T1, T2 = len(E1), len(E2)
def ema(a, k=0.98):
    o = a.copy()
    for i in range(1, len(a)): o[i] = k * o[i - 1] + (1 - k) * a[i]
    return o
E1s, E2s = ema(E1), ema(E2)
fmt = lambda n: f"{n:,}".replace(",", ".")
dec = lambda v, k=1: f"{v:.{k}f}".replace(".", ",")
ease = lambda t: 0 if t <= 0 else 1 if t >= 1 else t * t * (3 - 2 * t)
_fc = {}
def sg(size, w=700):
    if (size, w) not in _fc: _fc[(size, w)] = R.font(R.SG, size, w)
    return _fc[(size, w)]
fS = R.font(R.JB, 22, 400); fH = R.font(R.JB, 24, 500); fBox = R.font(R.SG, 28, 700)

def ink_img(a, w, h=None):
    a = (np.asarray(a) > 0.5)[..., None]; h = h or w
    rgb = np.where(a, np.array(INKC, np.uint8), np.array(PAPER, np.uint8)).astype(np.uint8)
    return Image.fromarray(rgb).resize((w, h), Image.NEAREST)
def signed_img(w_, size):
    v = np.clip(w_ / (np.abs(w_).max() + 1e-9), -1, 1).reshape(P, P)[..., None]; base = np.array((18, 26, 35), np.float32)
    rgb = np.where(v > 0, base + (np.array(GOLD) - base) * v, base + (np.array(BLUE) - base) * (-v))
    return Image.fromarray(rgb.astype(np.uint8)).resize((size, size), Image.NEAREST)

# restauración (precalculada con el modelo final)
RREG, _ = RP.region(); RDMG, RMASK = RP.damage(RREG)
RHIST = RP.repair(RUN["W1"], RUN["b1"], RUN["c1"], RDMG, RMASK, hist=True)
RACC = [float((RDMG == RREG)[RMASK].mean())] + [float(((e > .5) == (RREG > .5))[RMASK].mean()) for e in RHIST]
# miniatura del grabado con la franja de examen
PH = 300; PW = int(T.WID * PH / T.HT)
PRINT = ink_img(T.INK, PW, PH).convert("RGBA")
_d = ImageDraw.Draw(PRINT, "RGBA"); cutp = int(PW * T.CUT / T.WID)
_d.rectangle((cutp, 0, PW, PH), fill=(8, 12, 16, 70))
for yy in range(0, PH, 12): _d.line((cutp, yy, cutp, yy + 6), fill=GOLD + (255,), width=3)

def header(d_):
    d_.text((W / 2, 112), f"REDES NEURONALES · EP. 8 · {HANDLE}", font=R.fMonoS, fill=(70, 84, 98), anchor="mm")
    d_.text((W / 2, 205), "¿puede una red neuronal", font=R.fTitle, fill=GOLD, anchor="mm")
    pill = "aprender soñando"
    tb = d_.textbbox((0, 0), pill, font=R.fPill); tw = tb[2] - tb[0]
    x0 = W / 2 - (tw + 60) / 2 - 22; y0 = 262
    d_.rounded_rectangle((x0, y0, x0 + tw + 60, y0 + 86), radius=12, fill=(22, 30, 40), outline=(62, 78, 96), width=2)
    d_.text((x0 + 30 + tw / 2, y0 + 43), pill, font=R.fPill, fill=WHITE, anchor="mm")
    d_.text((x0 + tw + 60 + 34, y0 + 42), "?", font=R.fQ, fill=GOLD, anchor="mm")

def stack(d_, stage):
    boxes = [("píxeles · 400", 0), ("capa 1 · 300", 1), ("capa 2 · 120", 2)]
    for i, (lab, lv) in enumerate(boxes):
        x0 = 100 + i * 310; on = lv == stage or (stage == 3 and lv > 0)
        col = (BLUE if lv == 1 else GOLD) if on else ((80, 96, 112) if lv < stage else (50, 62, 74))
        d_.rounded_rectangle((x0, 462, x0 + 260, 530), radius=12, fill=(18, 25, 33), outline=col, width=3 if on else 2)
        d_.text((x0 + 130, 496), lab, font=fBox, fill=col if on else GREY, anchor="mm")
        if i < 2:
            lc = (BLUE if i == 0 else GOLD) if (stage == i + 1 or stage == 3) else (60, 74, 88)
            d_.line((x0 + 268, 496, x0 + 302, 496), fill=lc, width=4)
            d_.polygon([(x0 + 270, 488), (x0 + 270, 504), (x0 + 260, 496)], fill=lc); d_.polygon([(x0 + 300, 488), (x0 + 300, 504), (x0 + 310, 496)], fill=lc)
    d_.line((80, 552, 1000, 552), fill=(26, 36, 46), width=1)

LX0, LX1, LY0, LY1 = 150, 990, 1560, 1765
lmin, lmax = np.log10(0.02), np.log10(0.3)
def lpx(s, v): return LX0 + s / (T1 + T2 - 1) * (LX1 - LX0), LY1 - (np.log10(max(v, 1e-3)) - lmin) / (lmax - lmin) * (LY1 - LY0)
def chart(d_, s1, s2):
    R.panel(d_, (60, 1440, 1020, 1840))
    d_.text((90, 1478), "error de reconstrucción (log)", font=R.fMono, fill=GREY, anchor="lm")
    d_.line((640, 1470, 670, 1470), fill=BLUE, width=4); d_.text((680, 1470), f"capa 1 {dec(E1s[s1 - 1], 3)}", font=R.fMonoS, fill=WHITE, anchor="lm")
    if s2: d_.line((640, 1502, 670, 1502), fill=GOLD, width=4); d_.text((680, 1502), f"capa 2 {dec(E2s[s2 - 1], 3)}", font=R.fMonoS, fill=WHITE, anchor="lm")
    for v, lab in ((0.1, "0,1"), (0.03, "0,03")):
        _, yy = lpx(0, v); d_.line((LX0, yy, LX1, yy), fill=(26, 36, 46), width=1); d_.text((LX0 - 14, yy), lab, font=R.fMonoS, fill=GREY, anchor="rm")
    d_.line((LX0, LY1, LX1, LY1), fill=EDGE, width=2)
    xs_, _ = lpx(T1, 0.1)
    for yy in range(LY0, LY1, 14): d_.line((xs_, yy, xs_, yy + 7), fill=(90, 104, 118), width=1)
    d_.text(((LX0 + xs_) / 2, LY1 + 26), "capa 1", font=R.fMonoS, fill=GREY, anchor="mm"); d_.text(((xs_ + LX1) / 2, LY1 + 26), "capa 2", font=R.fMonoS, fill=GREY, anchor="mm")
    d_.text(((LX0 + LX1) / 2, LY1 + 56), "una capa cada vez · cada una mide su propio error", font=R.fMonoS, fill=GREY, anchor="mm")
    st = max(1, s1 // 400); pts = [lpx(i, E1s[i]) for i in range(0, s1, st)] + [lpx(s1 - 1, E1s[s1 - 1])]
    d_.line(pts, fill=BLUE, width=3, joint="curve")
    if s2:
        pts2 = [lpx(T1 + i, E2s[i]) for i in range(0, s2, max(1, s2 // 300))] + [lpx(T1 + s2 - 1, E2s[s2 - 1])]
        if len(pts2) > 1: d_.line(pts2, fill=GOLD, width=3, joint="curve")
        d_.ellipse((pts2[-1][0] - 6, pts2[-1][1] - 6, pts2[-1][0] + 6, pts2[-1][1] + 6), fill=GOLD)
    else: d_.ellipse((pts[-1][0] - 6, pts[-1][1] - 6, pts[-1][0] + 6, pts[-1][1] + 6), fill=BLUE)

def bars(d_, k):
    R.panel(d_, (60, 1440, 1020, 1840))
    d_.text((90, 1478), "píxeles dañados que acierta", font=R.fMono, fill=GREY, anchor="lm")
    for i, (lab, v, col) in enumerate((("antes (moneda al aire)", RACC[0], WHITE), (f"tras {k} pasadas de la red", RACC[k], GOLD))):
        y = 1550 + i * 110
        d_.text((90, y), lab, font=sg(32, 600), fill=col, anchor="lm")
        d_.rounded_rectangle((90, y + 30, 90 + 760 * v, y + 70), radius=8, fill=col)
        d_.text((90 + 760 * v + 14, y + 50), f"{dec(v * 100)} %", font=sg(34), fill=col, anchor="lm")

def snap_at(f):
    if f < F1: return 1, S1[min(len(S1) - 1, int(round((f / (F1 - 1)) ** 1.6 * (len(S1) - 1))))], None
    if f < F1 + F2: return 2, S1[-1], S2[min(len(S2) - 1, int(round((f - F1) / (F2 - 1) * (len(S2) - 1))))]
    return 3, S1[-1], S2[-1]

CYC, STEPS = 40, 30
class Dreamer:
    def __init__(s): s.r = np.random.default_rng(11); s.reset()
    def reset(s): s.v = (s.r.random((16, P * P)) < T.X.mean()).astype(np.float32); s.pv = s.v.copy()
    def step(s, Wt, b, c):
        h = (s.r.random((16, Wt.shape[1])) < sig(s.v @ Wt + c)).astype(np.float32); s.pv = sig(h @ Wt.T + b)
        s.v = (s.r.random(s.pv.shape) < s.pv).astype(np.float32)

def draw(f, dreams):
    stage, s1, s2 = snap_at(f); st1, Wt, b, c = s1
    img = R.BGIMG.copy(); d_ = ImageDraw.Draw(img, "RGBA"); header(d_)
    R.panel(d_, (60, 380, 1020, 1410))
    left = {1: f"capa 1 · paso {fmt(st1)} / {fmt(T1)}", 2: f"capa 2 · paso {fmt(s2[0]) if s2 else 0} / {fmt(T2)}", 3: "restaurar un grabado dañado"}[stage]
    d_.text((92, 425), left, font=R.fMono, fill=GOLD if stage == 3 else GREY, anchor="lm")
    d_.text((988, 425), "La gran ola · Hokusai, ~1831", font=R.fMono, fill=GREY, anchor="rm")
    stack(d_, stage)
    if stage < 3:
        if stage == 1:
            d_.text((95, 578), "lo que detecta cada neurona de la capa 1", font=fH, fill=GREY, anchor="lm"); filt = [Wt[:, j] for j in range(24)]
        else:
            d_.text((95, 578), "capa 2: combina patrones de la capa 1", font=fH, fill=GOLD, anchor="lm"); filt = [W1f @ s2[1][:, j] for j in range(24)]
        for j, w_ in enumerate(filt):
            r_, c_ = divmod(j, 12); img.paste(signed_img(w_, 64), (104 + c_ * 73, 598 + r_ * 73))
        d_ = ImageDraw.Draw(img, "RGBA"); d_.line((80, 760, 1000, 760), fill=(26, 36, 46), width=1)
        d_.text((95, 786), "sueños: parte de ruido y la red lo va convirtiendo en grabado", font=fH, fill=GREY, anchor="lm")
        for j in range(16):
            r_, c_ = divmod(j, 8); img.paste(ink_img(dreams[j].reshape(P, P), 100), (104 + c_ * 110, 806 + r_ * 110))
        d_ = ImageDraw.Draw(img, "RGBA"); d_.line((80, 1032, 1000, 1032), fill=(26, 36, 46), width=1)
        d_.text((95, 1058), "aprende con la izquierda del grabado · la franja marcada, nunca", font=fH, fill=GREY, anchor="lm")
        img.alpha_composite(PRINT, (540 - PW // 2, 1078)); d_ = ImageDraw.Draw(img, "RGBA")
        chart(d_, st1, s2[0] if stage == 2 else 0)
    else:
        zf = f - F1 - F2; k = int(np.clip((zf - 20) // 12 + 1, 0, 6))
        cur = RDMG if k == 0 else RHIST[k - 1]
        hR, wR = RREG.shape; sc = 2.5; iw, ih = int(wR * sc), int(hR * sc); xs = [104, 404, 704]
        labs = (("dañado", "40 % de píxeles al azar", GREY), ("la red restaura", None, GOLD), ("original", "para comparar", GREY))
        for x, (lab, sub, col) in zip(xs, labs):
            d_.text((x + iw / 2, 590), lab, font=sg(30, 600), fill=col, anchor="mm")
            if sub: d_.text((x + iw / 2, 618), sub, font=fS, fill=GREY, anchor="mm")
        d_.text((xs[1] + iw / 2, 618), f"pasada {k}/6", font=fS, fill=GOLD, anchor="mm")
        for x, arr in zip(xs, (RDMG, cur, RREG)):
            img.paste(ink_img(arr, iw, ih), (x, 640))
        d_ = ImageDraw.Draw(img, "RGBA")
        for x in xs: d_.rectangle((x - 2, 638, x + iw + 1, 641 + ih), outline=EDGE, width=2)
        d_.rectangle((xs[1] - 3, 637, xs[1] + iw + 2, 642 + ih), outline=GOLD, width=3)
        d_.text((540, 1250), f"recupera el {dec(RACC[k] * 100)} %", font=sg(64), fill=GOLD if k >= 5 else WHITE, anchor="mm")
        d_.text((540, 1315), "de los píxeles dañados · la red nunca vio esta zona", font=fS, fill=GREY, anchor="mm")
        bars(d_, k)
    return img

fSub = R.font(R.SG, 44, 500); fFollow = R.font(R.JB, 26, 600); fHandle = R.font(R.SG, 50, 700); fBrand = R.font(R.SG, 96, 700)
def end_card():
    img = R.BGIMG.copy(); pw, ph = 700, int(700 * T.HT / T.WID)
    gl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ImageDraw.Draw(gl).rectangle((540 - pw / 2 - 10, 700 - ph / 2 - 10, 540 + pw / 2 + 10, 700 + ph / 2 + 10), fill=GOLD + (150,))
    img.alpha_composite(gl.filter(ImageFilter.GaussianBlur(18)))
    img.paste(ink_img(T.INK, pw, ph), (540 - pw // 2, 700 - ph // 2))
    tl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ImageDraw.Draw(tl).text((W / 2, 940), "Redes Neuronales", font=fBrand, fill=GOLD, anchor="mm")
    img.alpha_composite(tl.filter(ImageFilter.GaussianBlur(10))); img.alpha_composite(tl); d_ = ImageDraw.Draw(img, "RGBA")
    d_.text((W / 2, 1030), "Ep. 8 · Deep Belief Networks", font=fSub, fill=WHITE, anchor="mm")
    R.panel(d_, (200, 1120, 880, 1380))
    d_.text((W / 2, 1180), "FIN DE LA TEMPORADA 1", font=fFollow, fill=GOLD, anchor="mm")
    d_.text((W / 2, 1250), "¿qué obra restauramos?", font=R.fMono, fill=GREY, anchor="mm")
    d_.text((W / 2, 1318), HANDLE, font=fHandle, fill=WHITE, anchor="mm")
    return img

def audio():
    sr = 44100; n = int(N / FPS * sr); t = np.arange(n) / sr; a = np.zeros(n)
    env = np.clip(t / 1.5, 0, 1) * np.clip((N / FPS - t) / 1.0, 0, 1)
    for fr in (98, 146.8, 196, 233.1): a += 0.04 * env * np.sin(2 * np.pi * fr * t) * (1 + 0.4 * np.sin(2 * np.pi * 0.15 * t + fr))
    def at(fr_, arr, g):
        i0 = int(fr_ / FPS * sr); m = min(len(arr), n - i0)
        if m > 0: a[i0:i0 + m] += g * arr[:m]
    rng = np.random.default_rng(0); m = int(1.0 * sr); tt = np.arange(m) / sr
    breath = np.convolve(rng.normal(0, 1, m), np.ones(60) / 60, "same") * np.sin(np.pi * tt) ** 2
    for f0 in range(0, F1 + F2, CYC): at(f0, breath, 0.25)
    tt2 = np.arange(int(.5 * sr)) / sr
    for k in range(1, 7): at(F1 + F2 + 20 + (k - 1) * 12, np.exp(-tt2 * 7) * np.sin(2 * np.pi * (392 * 2 ** ((k - 1) / 6)) * tt2), 0.12)
    i0 = int((SHOW_F + HOLD_F) / FPS * sr); tt3 = np.arange(n - i0) / sr
    for fr in (784, 1174.7, 1568): a[i0:] += 0.08 * np.exp(-tt3 * 2) * np.sin(2 * np.pi * fr * tt3)
    a = a / np.abs(a).max() * 0.8
    with wave.open("work/dbo.wav", "w") as wv:
        wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(sr); wv.writeframes((a * 32767).astype(np.int16).tobytes())

def frames():
    dr = Dreamer(); END = end_card(); last = None
    for f in range(N):
        if f < SHOW_F:
            _, s1, _ = snap_at(f); ph = f % CYC
            if ph == 0: dr.reset()
            elif ph <= STEPS: dr.step(s1[1], s1[2], s1[3])
            last = draw(f, dr.pv if ph > 0 else dr.v); yield f, last
        elif f < SHOW_F + HOLD_F: yield f, last
        else: yield f, Image.blend(last, END, min(1, (f - SHOW_F - HOLD_F) / 14))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "prev":
        want = set(map(int, sys.argv[2:]))
        for f, im in frames():
            if f in want: im.convert("RGB").save(f"work/o{f}.png")
            if f >= max(want): break
        sys.exit()
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "work/dbo.mp4"], stdin=subprocess.PIPE)
    for f, im in frames(): p.stdin.write(im.convert("RGB").tobytes())
    p.stdin.close(); p.wait(); audio()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "work/dbo.mp4", "-i", "work/dbo.wav", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "160k", "-shortest", "out/ep8b_dbn_gran_ola.mp4"])
