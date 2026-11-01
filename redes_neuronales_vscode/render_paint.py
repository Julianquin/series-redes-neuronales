import numpy as np, pickle, subprocess, wave
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import render_mlp as R

W, H, FPS = 1080, 1920, 30
TRAIN_F, ZOOM_F, HOLD_F, END_F = 390, 100, 20, 66
N = TRAIN_F + ZOOM_F + HOLD_F + END_F
GOLD, BLUE, WHITE, GREY, EDGE, PANEL = R.GOLD, R.BLUE, R.WHITE, R.GREY, R.EDGE, R.PANEL
HANDLE = R.HANDLE
META = pickle.load(open("work/paint_meta.pkl", "rb"))
KEYS, PS, BOX, ITERS = META["keys"], META["ps"], META["BOX"], META["ITERS"]; NK = len(KEYS)
PL = np.load("work/paint_plain.npy"); FO = np.load("work/paint_fourier.npy")
ZPIX = np.load("work/paint_zoom_pix.npy"); ZNET = np.load("work/paint_zoom_net.npy")
ORIG = np.load("work/paint_orig.npy")
fmt = lambda n: f"{n:,}".replace(",", ".")
dec = lambda v, k=1: f"{v:.{k}f}".replace(".", ",")
ease = lambda t: 0 if t <= 0 else 1 if t >= 1 else t * t * (3 - 2 * t)

_fc = {}
def sg(size, w=700):
    if (size, w) not in _fc: _fc[(size, w)] = R.font(R.SG, size, w)
    return _fc[(size, w)]
fS = R.font(R.JB, 22, 400); fM = R.font(R.JB, 24, 500)

def header(d_):
    d_.text((W / 2, 112), f"REDES NEURONALES · EP. 1 · {HANDLE}", font=R.fMonoS, fill=(70, 84, 98), anchor="mm")
    d_.text((W / 2, 205), "¿puede una red neuronal", font=R.fTitle, fill=GOLD, anchor="mm")
    pill = "pintar un cuadro"
    tb = d_.textbbox((0, 0), pill, font=R.fPill); tw = tb[2] - tb[0]
    x0 = W / 2 - (tw + 60) / 2 - 22; y0 = 262
    d_.rounded_rectangle((x0, y0, x0 + tw + 60, y0 + 86), radius=12, fill=(22, 30, 40), outline=(62, 78, 96), width=2)
    d_.text((x0 + 30 + tw / 2, y0 + 43), pill, font=R.fPill, fill=WHITE, anchor="mm")
    d_.text((x0 + tw + 60 + 34, y0 + 42), "?", font=R.fQ, fill=GOLD, anchor="mm")

# ---- layout ----
CL, CR = 95, 545; AY, BY = 490, 915; IW, IH = 440, 275
LX0, LX1, LY0, LY1 = 150, 990, 1560, 1765
LOGMAX = np.log10(ITERS)
def lpx(it, v): return LX0 + np.log10(max(it, 1)) / LOGMAX * (LX1 - LX0), LY1 - (v - 10) / 26 * (LY1 - LY0)
def chart(d_, idx):
    R.panel(d_, (60, 1440, 1020, 1840))
    d_.text((90, 1478), "calidad del cuadro (dB)", font=R.fMono, fill=GREY, anchor="lm")
    for v in (15, 20, 25, 30, 35):
        _, yy = lpx(1, v); d_.line((LX0, yy, LX1, yy), fill=(26, 36, 46), width=1); d_.text((LX0 - 14, yy), str(v), font=R.fMonoS, fill=GREY, anchor="rm")
    d_.line((LX0, LY1, LX1, LY1), fill=EDGE, width=2)
    for e in (1, 10, 100, 1000, ITERS):
        x, _ = lpx(e, 10); d_.text((min(x, LX1 - 20), LY1 + 26), fmt(e), font=R.fMonoS, fill=GREY, anchor="mm")
    d_.text(((LX0 + LX1) / 2, LY1 + 56), "iteración (escala log)", font=R.fMonoS, fill=GREY, anchor="mm")
    for kind, col in (("plain", BLUE), ("fourier", GOLD)):
        pts = [lpx(it, v) for it, v in PS[kind][:idx + 1]]
        if len(pts) > 1: d_.line(pts, fill=col, width=4, joint="curve")
        d_.ellipse((pts[-1][0] - 7, pts[-1][1] - 7, pts[-1][0] + 7, pts[-1][1] + 7), fill=col)
    v_p, v_f = PS["plain"][idx][1], PS["fourier"][idx][1]
    d_.line((560, 1478, 590, 1478), fill=BLUE, width=4); d_.text((600, 1478), f"normal {dec(v_p)}", font=R.fMonoS, fill=WHITE, anchor="lm")
    d_.line((790, 1478, 820, 1478), fill=GOLD, width=4); d_.text((830, 1478), f"Fourier {dec(v_f)}", font=R.fMonoS, fill=WHITE, anchor="lm")

def img_at(d_, img, arr, x, y, outline=EDGE):
    img.paste(Image.fromarray(arr), (x, y)); d_.rectangle((x - 1, y - 1, x + IW, y + IH), outline=outline, width=2)

def frame_scene(f):
    zf = max(0, f - TRAIN_F); in_zoom = f >= TRAIN_F
    idx = int(round(min(1.0, f / (TRAIN_F - 1)) ** 0.85 * (NK - 1)))
    img = R.BGIMG.copy(); d_ = ImageDraw.Draw(img, "RGBA"); header(d_)
    R.panel(d_, (60, 380, 1020, 1410))
    d_.text((92, 425), f"iter {fmt(KEYS[idx])} / {fmt(ITERS)}", font=R.fMono, fill=GREY, anchor="lm")
    d_.text((988, 425), "MLP · 3 capas ocultas × 192", font=R.fMono, fill=GREY, anchor="rm")
    mix = ease((zf - 25) / 20) if in_zoom else 0
    # row A labels
    lt = ("red normal", "red + Fourier") if mix < .5 else ("los píxeles ×8", "la red ×8")
    for cx, a, col in ((CL + IW / 2, lt[0], WHITE), (CR + IW / 2, lt[1], GOLD)):
        d_.text((cx, 462), a, font=sg(32), fill=col, anchor="mm")
    # row A images
    if mix <= 0: aL, aR = PL[idx], FO[idx]
    else:
        aL = (PL[idx] * (1 - mix) + ZPIX * mix).astype(np.uint8); aR = (FO[idx] * (1 - mix) + ZNET * mix).astype(np.uint8)
    img_at(d_, img, aL, CL, AY); img_at(d_, img, aR, CR, AY, GOLD if mix > 0.5 else EDGE)
    # stats under images
    v_p, v_f = PS["plain"][idx][1], PS["fourier"][idx][1]
    if mix < .5:
        d_.text((CL + IW / 2, 808), f"{dec(v_p)} dB", font=sg(54), fill=WHITE, anchor="mm"); d_.text((CL + IW / 2, 850), "solo ve curvas suaves", font=fS, fill=GREY, anchor="mm")
        d_.text((CR + IW / 2, 808), f"{dec(v_f)} dB", font=sg(54), fill=GOLD, anchor="mm"); d_.text((CR + IW / 2, 850), "detalle fino desde el principio", font=fS, fill=GREY, anchor="mm")
    else:
        cw, ch = META["crop_px"]
        d_.text((CL + IW / 2, 808), f"{cw}×{ch} px", font=sg(54), fill=WHITE, anchor="mm"); d_.text((CL + IW / 2, 850), "un recorte del lienzo", font=fS, fill=GREY, anchor="mm")
        d_.text((CR + IW / 2, 808), "sin píxeles", font=sg(54), fill=GOLD, anchor="mm"); d_.text((CR + IW / 2, 850), "una función continua", font=fS, fill=GREY, anchor="mm")
    d_.line((80, 878, 1000, 878), fill=(26, 36, 46), width=1)
    # row B
    d_.text((CL + IW / 2, 898), "original · Van Gogh, 1889", font=fS, fill=GREY, anchor="mm")
    img_at(d_, img, ORIG, CL, BY)
    if in_zoom:
        a = ease(zf / 15); x0, y0, x1, y1 = BOX
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
        ld.rectangle((CL + x0 * IW, BY + y0 * IH, CL + x1 * IW, BY + y1 * IH), outline=GOLD + (int(255 * a),), width=4)
        img.alpha_composite(lay); d_ = ImageDraw.Draw(img, "RGBA")
    d_.text((CR + IW / 2, 898), "lo que sabe hacer la red", font=fS, fill=GREY, anchor="mm")
    d_.rounded_rectangle((CR, BY, CR + IW, BY + IH), radius=14, fill=(18, 25, 33), outline=EDGE, width=2)
    d_.text((CR + 30, BY + 38), "ENTRADA", font=fS, fill=GREY, anchor="lm"); d_.text((CR + 30, BY + 82), "(x, y)", font=sg(50), fill=WHITE, anchor="lm")
    d_.text((CR + 30, BY + 120), "una posición del lienzo", font=fS, fill=GREY, anchor="lm")
    d_.text((CR + 30, BY + 168), "SALIDA", font=fS, fill=GREY, anchor="lm"); d_.text((CR + 30, BY + 212), "(r, g, b)", font=sg(50), fill=GOLD, anchor="lm")
    d_.text((CR + 30, BY + 250), "el color en ese punto", font=fS, fill=GREY, anchor="lm")
    # row C message
    if not in_zoom: l1, l2 = "La red no guarda píxeles:", "guarda una función de (x, y) a color."
    else: l1, l2 = "No inventa detalle nuevo:", "interpola entre lo que aprendió."
    d_.text((540, 1262), l1, font=sg(42, 600), fill=WHITE, anchor="mm"); d_.text((540, 1322), l2, font=sg(42, 600), fill=GOLD, anchor="mm")
    chart(d_, idx)
    return img

fSub = R.font(R.SG, 44, 500); fFollow = R.font(R.JB, 26, 600); fHandle = R.font(R.SG, 50, 700); fBrand = R.font(R.SG, 96, 700)
def end_card():
    img = R.BGIMG.copy()
    fw, fh = 640, 400; gl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(gl)
    gd.rounded_rectangle((540 - fw / 2 - 12, 700 - fh / 2 - 12, 540 + fw / 2 + 12, 700 + fh / 2 + 12), radius=14, fill=GOLD + (170,))
    img.alpha_composite(gl.filter(ImageFilter.GaussianBlur(18))); d_ = ImageDraw.Draw(img, "RGBA")
    d_.rectangle((540 - fw / 2 - 10, 700 - fh / 2 - 10, 540 + fw / 2 + 10, 700 + fh / 2 + 10), fill=(26, 20, 8), outline=GOLD, width=4)
    img.paste(Image.fromarray(FO[-1]).resize((fw, fh), Image.LANCZOS), (540 - fw // 2, 700 - fh // 2))
    tl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ImageDraw.Draw(tl).text((W / 2, 1000), "Redes Neuronales", font=fBrand, fill=GOLD, anchor="mm")
    img.alpha_composite(tl.filter(ImageFilter.GaussianBlur(10))); img.alpha_composite(tl); d_ = ImageDraw.Draw(img, "RGBA")
    d_.text((W / 2, 1090), "Ep. 1 · El perceptrón multicapa", font=fSub, fill=WHITE, anchor="mm")
    R.panel(d_, (200, 1170, 880, 1410))
    d_.text((W / 2, 1230), "¿QUÉ OBRA QUIERES VER?", font=fFollow, fill=GOLD, anchor="mm")
    d_.text((W / 2, 1296), "comenta una de dominio público", font=R.fMono, fill=GREY, anchor="mm")
    d_.text((W / 2, 1356), HANDLE, font=fHandle, fill=WHITE, anchor="mm")
    return img

def audio():
    sr = 44100; n = int(N / FPS * sr); t = np.arange(n) / sr; a = np.zeros(n)
    env = np.clip(t / 1.2, 0, 1) * np.clip((N / FPS - t) / 1.0, 0, 1)
    for fr in (110, 164.8, 220, 277.2): a += 0.035 * env * np.sin(2 * np.pi * fr * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.25 * t))
    # voz de "pincel": su altura sigue la calidad de la red de Fourier
    fi = np.arange(n) / sr * FPS; idx = np.clip((np.clip(fi, 0, TRAIN_F - 1) / (TRAIN_F - 1)) ** 0.85 * (NK - 1), 0, NK - 1)
    q = np.interp(idx, np.arange(NK), [v for _, v in PS["fourier"]]); fr = 220 * 2 ** ((q - 12) / 12 * 1.1 / 1.0)
    ph = 2 * np.pi * np.cumsum(fr) / sr; amp = 0.05 * (fi < TRAIN_F + 30) * env
    a += amp * (np.sin(ph) + 0.4 * np.sin(2 * ph))
    rng = np.random.default_rng(1)
    for s in np.arange(0.2, TRAIN_F / FPS, 0.22):          # pinceladas
        i0 = int(s * sr); m = int(0.12 * sr); tt = np.arange(m) / sr
        a[i0:i0 + m] += 0.05 * np.exp(-tt * 28) * np.convolve(rng.normal(0, 1, m), np.ones(8) / 8, "same")
    z0 = int((TRAIN_F + 20) / FPS * sr); m = int(0.9 * sr); tt = np.arange(m) / sr
    sw = np.convolve(rng.normal(0, 1, m), np.ones(40) / 40, "same") * np.sin(np.pi * tt / 0.9) ** 2; a[z0:z0 + m] += 0.18 * sw
    i0 = int((TRAIN_F + ZOOM_F + HOLD_F) / FPS * sr); tt = np.arange(n - i0) / sr
    for f_ in (784, 1174.7, 1568): a[i0:] += 0.08 * np.exp(-tt * 2.2) * np.sin(2 * np.pi * f_ * tt)
    a = a / np.abs(a).max() * 0.8
    with wave.open("work/paint.wav", "w") as wv:
        wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(sr); wv.writeframes((a * 32767).astype(np.int16).tobytes())

if __name__ == "__main__":
    import sys
    END = end_card(); LASTF = TRAIN_F + ZOOM_F - 1; LAST = frame_scene(LASTF)
    def frame(f):
        if f <= LASTF: return frame_scene(f)
        if f < LASTF + 1 + HOLD_F: return LAST
        return Image.blend(LAST, END, min(1, (f - LASTF - 1 - HOLD_F) / 14))
    if len(sys.argv) > 1:
        for f in map(int, sys.argv[1:]): frame(f).convert("RGB").save(f"work/q{f}.png")
        sys.exit()
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "work/paint.mp4"], stdin=subprocess.PIPE)
    for f in range(N): p.stdin.write(frame(f).convert("RGB").tobytes())
    p.stdin.close(); p.wait(); audio()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "work/paint.mp4", "-i", "work/paint.wav", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "160k", "-shortest", "out/ep1b_la_noche_estrellada.mp4"])
