import numpy as np, pickle, subprocess, wave
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import render_mlp as R
import train_ae_cuadros as T

W, H, FPS = 1080, 1920, 30
S1, S2, HOLD_F, END_F = 390, 105, 20, 66
N = S1 + S2 + HOLD_F + END_F
GOLD, BLUE, WHITE, GREY, EDGE, PANEL = R.GOLD, R.BLUE, R.WHITE, R.GREY, R.EDGE, R.PANEL
CCOL = [(92, 156, 255), (80, 210, 220), (230, 150, 70), (240, 110, 190), (190, 120, 255)]
NAMES = [p[0] for p in T.PAINT]
HANDLE = R.HANDLE
RUN = pickle.load(open("work/run_ae_cuadros.pkl", "rb"))
KEYS = sorted(RUN["snaps"]); NK = len(KEYS); ITERS = RUN["ITERS"]
LD = {k: dict(RUN["L"][k]) for k in (2, 32)}
fmt = lambda n: f"{n:,}".replace(",", ".")
dec = lambda v, k=4: f"{v:.{k}f}".replace(".", ",")
ease = lambda t: 0 if t <= 0 else 1 if t >= 1 else t * t * (3 - 2 * t)
_fc = {}
def sg(size, w=700):
    if (size, w) not in _fc: _fc[(size, w)] = R.font(R.SG, size, w)
    return _fc[(size, w)]
fS = R.font(R.JB, 22, 400)

SEL = [c * 200 + i for c in range(5) for i in (3, 50)]; XR = T.XVA[SEL]
SHOW = np.random.default_rng(3).choice(len(T.XTR), 1500, replace=False); XS, YS = T.XTR[SHOW], T.YTR[SHOW]
def kidx(f): return int(round((min(f, S1 - 1) / (S1 - 1)) ** 0.9 * (NK - 1)))
bounds, cur = [], None                       # vista del mapa latente, suavizada entre fotogramas
for f in range(S1):
    Z = T.encode(RUN["snaps"][KEYS[kidx(f)]][2], XS); lo, hi = np.percentile(Z, 1, 0), np.percentile(Z, 99, 0)
    b = np.r_[(lo + hi) / 2, (hi - lo).max() / 2 * 1.12]; cur = b if cur is None else 0.85 * cur + 0.15 * b; bounds.append(cur.copy())
PF = RUN["snaps"][KEYS[-1]]
ZF = T.encode(PF[2], XS); CEN = np.median(ZF, 0); MED = []
for c in range(5):                      # parada = punto característico del cuadro, hacia fuera del centro
    zc = ZF[YS == c]; d = np.median(zc, 0) - CEN; d /= np.linalg.norm(d) + 1e-9
    pr = (zc - CEN) @ d; MED.append(np.median(zc[pr >= np.percentile(pr, 80)], 0))
ROUTE = [1, 0, 2, 3]; SEG = 30

def patch(x, size, nearest=True):
    return Image.fromarray((np.clip(x, 0, 1).reshape(16, 16, 3) * 255).astype(np.uint8)).resize((size, size), Image.NEAREST if nearest else Image.BICUBIC)

def header(d_):
    d_.text((W / 2, 112), f"REDES NEURONALES · EP. 5 · {HANDLE}", font=R.fMonoS, fill=(70, 84, 98), anchor="mm")
    d_.text((W / 2, 205), "¿puede una red neuronal", font=R.fTitle, fill=GOLD, anchor="mm")
    pill = "guardar un cuadro en 2 números"; f_ = R.fPill
    tb = d_.textbbox((0, 0), pill, font=f_); tw = tb[2] - tb[0]
    if tw > 820: f_ = R.font(R.JB, 42, 600); tb = d_.textbbox((0, 0), pill, font=f_); tw = tb[2] - tb[0]
    x0 = W / 2 - (tw + 60) / 2 - 22; y0 = 262
    d_.rounded_rectangle((x0, y0, x0 + tw + 60, y0 + 86), radius=12, fill=(22, 30, 40), outline=(62, 78, 96), width=2)
    d_.text((x0 + 30 + tw / 2, y0 + 43), pill, font=f_, fill=WHITE, anchor="mm")
    d_.text((x0 + tw + 60 + 34, y0 + 42), "?", font=R.fQ, fill=GOLD, anchor="mm")

LX0, LX1, LY0, LY1 = 150, 990, 1560, 1765
lmin, lmax = np.log10(0.008), np.log10(0.12)
def lpx(it, v): return LX0 + np.log10(max(it, 1)) / np.log10(ITERS) * (LX1 - LX0), LY1 - (np.log10(v) - lmin) / (lmax - lmin) * (LY1 - LY0)
def chart(d_, k):
    R.panel(d_, (60, 1440, 1020, 1840))
    d_.text((90, 1478), "error de reconstrucción", font=R.fMono, fill=GREY, anchor="lm")
    d_.line((560, 1478, 590, 1478), fill=BLUE, width=4); d_.text((600, 1478), f"2 núm. {dec(LD[2][k])}", font=R.fMonoS, fill=WHITE, anchor="lm")
    d_.line((790, 1478, 820, 1478), fill=GOLD, width=4); d_.text((830, 1478), f"32 núm. {dec(LD[32][k])}", font=R.fMonoS, fill=WHITE, anchor="lm")
    for v, lab in ((0.1, "0,1"), (0.05, "0,05"), (0.02, "0,02"), (0.01, "0,01")):
        _, yy = lpx(1, v); d_.line((LX0, yy, LX1, yy), fill=(26, 36, 46), width=1); d_.text((LX0 - 14, yy), lab, font=R.fMonoS, fill=GREY, anchor="rm")
    d_.line((LX0, LY1, LX1, LY1), fill=EDGE, width=2)
    for e in (1, 10, 100, 1000, ITERS):
        x, _ = lpx(e, 0.01); d_.text((min(x, LX1 - 20), LY1 + 26), fmt(e), font=R.fMonoS, fill=GREY, anchor="mm")
    d_.text(((LX0 + LX1) / 2, LY1 + 56), "iteración (escala log)", font=R.fMonoS, fill=GREY, anchor="mm")
    for kk, col in ((2, BLUE), (32, GOLD)):
        pts = [lpx(it, LD[kk][it]) for it in KEYS if it <= k]
        if len(pts) > 1: d_.line(pts, fill=col, width=3, joint="curve")
        d_.ellipse((pts[-1][0] - 6, pts[-1][1] - 6, pts[-1][0] + 6, pts[-1][1] + 6), fill=col)

SX0, SY0, SS = 95, 872, 430
GX0, GN, GT = 555, 5, 80
def scene(f):
    fin = f >= S1; k = KEYS[kidx(f)]; P = RUN["snaps"][k]; cx, cy, r = bounds[min(f, S1 - 1)]
    img = R.BGIMG.copy(); d_ = ImageDraw.Draw(img, "RGBA"); header(d_)
    R.panel(d_, (60, 380, 1020, 1410))
    d_.text((92, 425), f"iter {fmt(k)} / {fmt(ITERS)}", font=R.fMono, fill=GREY, anchor="lm")
    d_.text((988, 425), "768 → 2 (o 32) → 768", font=R.fMono, fill=GREY, anchor="rm")
    rec = {kk: T.forward(P[kk], XR)[-1] for kk in (2, 32)}
    rows = (("trozos que nunca vio (16×16 px)", XR, None), ("reconstruido con 2 números", rec[2], 2), ("reconstruido con 32 números", rec[32], 32))
    for ri, (lab, arr, kk) in enumerate(rows):
        y = 470 + ri * 118
        d_.text((100, y), lab, font=fS, fill=GREY if kk is None else (BLUE if kk == 2 else GOLD), anchor="lm")
        if kk: d_.text((980, y), f"error {dec(LD[kk][k])}", font=fS, fill=GREY, anchor="rm")
        for j in range(10):
            x = 100 + j * 88; img.paste(patch(arr[j], 80), (x, y + 16))
        d_ = ImageDraw.Draw(img, "RGBA")
    d_.line((80, 826, 1000, 826), fill=(26, 36, 46), width=1)
    # mapa latente
    d_.text((SX0, 850), "mapa de 2 números (z₁, z₂)", font=fS, fill=GREY, anchor="lm")
    d_.rectangle((SX0, SY0, SX0 + SS, SY0 + SS), fill=(11, 16, 22), outline=EDGE)
    Z = T.encode(P[2], XS); to = lambda z: (SX0 + (z[0] - cx + r) / (2 * r) * SS, SY0 + (cy + r - z[1]) / (2 * r) * SS)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
    for z, c in zip(Z, YS):
        px, py = to(z)
        if SX0 + 3 < px < SX0 + SS - 3 and SY0 + 3 < py < SY0 + SS - 3: ld.ellipse((px - 3, py - 3, px + 3, py + 3), fill=CCOL[c] + (170,))
    img.alpha_composite(lay); d_ = ImageDraw.Draw(img, "RGBA")
    gv = np.linspace(-0.75, 0.75, GN); grid = np.array([[cx + a * r, cy - b * r] for b in gv for a in gv])
    if not fin:
        for z in grid:
            px, py = to(z); d_.line((px - 6, py, px + 6, py), fill=WHITE + (170,), width=2); d_.line((px, py - 6, px, py + 6), fill=WHITE + (170,), width=2)
        d_.text((GX0, 850), "lo que dibuja el decoder en cada +", font=fS, fill=GREY, anchor="lm")
        for i, x in enumerate(T.decode(P[2], grid)):
            rr, cc = divmod(i, GN); img.paste(patch(x, GT, False), (GX0 + cc * (GT + 6), SY0 + rr * (GT + 6)))
        d_ = ImageDraw.Draw(img, "RGBA")
    else:
        zf = f - S1; s = min(len(ROUTE) - 1.0001, max(0, (zf - 6) / SEG)); a, t = int(s), ease(s - int(s))
        za, zb = MED[ROUTE[a]], MED[ROUTE[a + 1]]; z = za + (zb - za) * t
        pts = [to(MED[c]) for c in ROUTE]
        d_.line(pts, fill=WHITE + (110,), width=3, joint="curve")
        for c in ROUTE:
            px, py = to(MED[c]); d_.ellipse((px - 7, py - 7, px + 7, py + 7), fill=CCOL[c], outline=(8, 12, 16), width=2)
        px, py = to(z); d_.ellipse((px - 13, py - 13, px + 13, py + 13), outline=GOLD, width=4)
        d_.text((GX0, 850), "paseo por el mapa", font=fS, fill=GOLD, anchor="lm")
        big = patch(T.decode(PF[2], z[None])[0], 400, True)
        img.paste(big, (GX0 + 15, SY0 + 15)); d_ = ImageDraw.Draw(img, "RGBA")
        d_.rectangle((GX0 + 13, SY0 + 13, GX0 + 417, SY0 + 417), outline=GOLD, width=3)
        A_, B_ = NAMES[ROUTE[a]], NAMES[ROUTE[a + 1]]
        d_.rounded_rectangle((GX0 + 30, SY0 + 340, GX0 + 400, SY0 + 400), radius=12, fill=(8, 12, 16, 200))
        d_.text((GX0 + 215, SY0 + 370), f"de {A_} a {B_}", font=sg(28, 600), fill=WHITE, anchor="mm")
    # leyenda
    x = 100
    for c in range(5):
        d_.ellipse((x, 1338, x + 16, 1354), fill=CCOL[c]); d_.text((x + 24, 1346), NAMES[c], font=fS, fill=WHITE, anchor="lm")
        x += 24 + fS.getlength(NAMES[c]) + 40
    d_.text((540, 1385), "color = cuadro de origen · la red nunca ve la etiqueta", font=fS, fill=GREY, anchor="mm")
    chart(d_, k)
    return img

fSub = R.font(R.SG, 44, 500); fFollow = R.font(R.JB, 26, 600); fHandle = R.font(R.SG, 50, 700); fBrand = R.font(R.SG, 96, 700)
def end_card():
    img = R.BGIMG.copy(); n = 9; t = 58; x0 = 540 - n * (t + 4) / 2; y0 = 700 - 3 * (t + 4) / 2 - 40
    cx, cy, r = bounds[-1]; gv = np.linspace(-0.8, 0.8, n); gr = np.array([[cx + a * r, cy - b * r] for b in np.linspace(-0.8, 0.8, 4) for a in gv])
    gl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ImageDraw.Draw(gl).rounded_rectangle((x0 - 12, y0 - 12, x0 + n * (t + 4) + 8, y0 + 4 * (t + 4) + 8), radius=12, fill=GOLD + (140,))
    img.alpha_composite(gl.filter(ImageFilter.GaussianBlur(18)))
    for i, x in enumerate(T.decode(PF[2], gr)):
        rr, cc = divmod(i, n); img.paste(patch(x, t, False), (int(x0 + cc * (t + 4)), int(y0 + rr * (t + 4))))
    tl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ImageDraw.Draw(tl).text((W / 2, 940), "Redes Neuronales", font=fBrand, fill=GOLD, anchor="mm")
    img.alpha_composite(tl.filter(ImageFilter.GaussianBlur(10))); img.alpha_composite(tl); d_ = ImageDraw.Draw(img, "RGBA")
    d_.text((W / 2, 1030), "Ep. 5 · Autoencoders", font=fSub, fill=WHITE, anchor="mm")
    R.panel(d_, (200, 1120, 880, 1380))
    d_.text((W / 2, 1180), "¿QUÉ CUADROS MEZCLO?", font=fFollow, fill=GOLD, anchor="mm")
    d_.text((W / 2, 1250), "comenta dos obras de dominio público", font=R.fMono, fill=GREY, anchor="mm")
    d_.text((W / 2, 1318), HANDLE, font=fHandle, fill=WHITE, anchor="mm")
    return img

def audio():
    sr = 44100; n = int(N / FPS * sr); t = np.arange(n) / sr; a = np.zeros(n)
    env = np.clip(t / 1.2, 0, 1) * np.clip((N / FPS - t) / 1.0, 0, 1)
    det = 5 * np.clip(1 - t / (S1 / FPS), 0, 1)            # acorde que se afina mientras aprende
    for fr in (110, 164.8, 220, 277.2): a += 0.03 * env * (np.sin(2 * np.pi * fr * t) + np.sin(2 * np.pi * (fr + det) * t))
    fi = t * FPS; zf = np.clip(fi - S1 - 6, 0, None) / SEG; on = ((fi >= S1) & (fi < S1 + S2)).astype(float)
    fr2 = 330 * 2 ** (np.clip(zf, 0, 4) / 4); ph = 2 * np.pi * np.cumsum(fr2) / sr
    a += 0.06 * on * np.sin(ph) * (0.7 + 0.3 * np.sin(2 * np.pi * 3 * t))
    i0 = int((S1 + S2 + HOLD_F) / FPS * sr); tt = np.arange(n - i0) / sr
    for fr in (880, 1318.5, 1760): a[i0:] += 0.08 * np.exp(-tt * 2.5) * np.sin(2 * np.pi * fr * tt)
    a = a / np.abs(a).max() * 0.8
    with wave.open("work/aec.wav", "w") as wv:
        wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(sr); wv.writeframes((a * 32767).astype(np.int16).tobytes())

if __name__ == "__main__":
    import sys
    END = end_card(); LASTF = S1 + S2 - 1; LAST = scene(LASTF)
    def frame(f):
        if f <= LASTF: return scene(f)
        if f <= LASTF + HOLD_F: return LAST
        return Image.blend(LAST, END, min(1, (f - LASTF - HOLD_F) / 14))
    if len(sys.argv) > 1 and sys.argv[1] == "prev":
        for f in map(int, sys.argv[2:]): frame(f).convert("RGB").save(f"work/e{f}.png")
        sys.exit()
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "work/aec.mp4"], stdin=subprocess.PIPE)
    for f in range(N): p.stdin.write(frame(f).convert("RGB").tobytes())
    p.stdin.close(); p.wait(); audio()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "work/aec.mp4", "-i", "work/aec.wav", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "160k", "-shortest", "out/ep5b_autoencoder_cuadros.mp4"])
