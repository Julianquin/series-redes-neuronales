"""Ep. 1b rediseñado para Instagram Reels. Uso: python3 reel_paint.py A|B|C [prev f1 f2 ...]"""
import numpy as np, pickle, subprocess, wave, sys, math
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS = 1080, 1920, 30
F2 = "fonts/"
FONTSETS = {
    "A": dict(name="Instrument Serif + IBM Plex Mono",
              head=(F2 + "instrumentserif_InstrumentSerif-Regular.ttf", {}), accent=(F2 + "instrumentserif_InstrumentSerif-Italic.ttf", {}),
              body=(F2 + "instrumentserif_InstrumentSerif-Italic.ttf", {}), mono=(F2 + "ibmplexmono_IBMPlexMono-Medium.ttf", {}),
              monob=(F2 + "ibmplexmono_IBMPlexMono-SemiBold.ttf", {}), hs=118, bs=58, upper=False),
    "B": dict(name="Bricolage Grotesque + JetBrains Mono",
              head=(F2 + "bricolagegrotesque_BricolageGrotesque.ttf", {"Weight": 800, "Width": 100}),
              accent=(F2 + "bricolagegrotesque_BricolageGrotesque.ttf", {"Weight": 800, "Width": 100}),
              body=(F2 + "bricolagegrotesque_BricolageGrotesque.ttf", {"Weight": 500, "Width": 100}),
              mono=("fonts/JBM.ttf", {"Weight": 500}), monob=("fonts/JBM.ttf", {"Weight": 700}), hs=98, bs=48, upper=False),
    "C": dict(name="Archivo Black + Inter",
              head=(F2 + "archivoblack_ArchivoBlack-Regular.ttf", {}), accent=(F2 + "archivoblack_ArchivoBlack-Regular.ttf", {}),
              body=(F2 + "inter_Inter.ttf", {"Weight": 500}), mono=(F2 + "inter_Inter.ttf", {"Weight": 600}),
              monob=(F2 + "inter_Inter.ttf", {"Weight": 800}), hs=86, bs=46, upper=True),
}
BG = (8, 12, 18); GOLD = (244, 190, 78); BLUE = (110, 168, 255); WHITE = (238, 242, 246); SOFT = (196, 206, 218); MUTED = (140, 154, 170)
HANDLE = "@rojocuantico"
ease = lambda t: 0 if t <= 0 else 1 if t >= 1 else t * t * (3 - 2 * t)
fmt = lambda n: f"{n:,}".replace(",", ".")
dec = lambda v, k=1: f"{v:.{k}f}".replace(".", ",")

_fc = {}
def font(spec, size):
    key = (spec[0], tuple(sorted(spec[1].items())), size)
    if key in _fc: return _fc[key]
    f = ImageFont.truetype(spec[0], size)
    try:
        axes = f.get_variation_axes(); vals = []
        for a in axes:
            nm = a["name"].decode() if isinstance(a["name"], bytes) else a["name"]
            if nm == "Optical size": vals.append(max(a["minimum"], min(a["maximum"], size / 2.2)))
            else: vals.append(spec[1].get(nm, a["default"]))
        f.set_variation_by_axes(vals)
    except Exception: pass
    _fc[key] = f; return f

def fit(spec, size, lines, maxw):
    while size > 20 and max(font(spec, size).getlength(l) for l in lines) > maxw: size -= 2
    return font(spec, size)

# ---------------- datos ----------------
META = pickle.load(open("work/hd_meta.pkl", "rb"))
IDX, KEYS, PS, ITERS = META["IDX"], META["keys"], META["ps"], META["ITERS"]
class _Lazy:
    def __init__(s, kind): s.kind = kind; s.c = {}
    def __getitem__(s, j):
        if j not in s.c: s.c = {j: np.load(f"work/hd/{s.kind}_{j:03d}.npy")}
        return s.c[j]
HDF, HDP = _Lazy("fourier"), _Lazy("plain")
HERO = np.load("work/hd_hero.npy"); ZNET = np.load("work/hd_zoom_net.npy"); ZPIX = np.load("work/hd_zoom_pix.npy")
HERO_IMG = Image.fromarray(HERO); NPIX = 230 * 144
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
_g = np.exp(-(((xx - W / 2) / 650) ** 2 + ((yy - 820) / 900) ** 2))
_bg = np.stack([BG[i] + _g * c for i, c in enumerate((10, 16, 30))], -1)
BGIMG = Image.fromarray(_bg.clip(0, 255).astype(np.uint8)).convert("RGBA")

# ---------------- escenas ----------------
SCENES = [("hook", 66), ("how", 96), ("race", 180), ("curve", 90), ("trick", 96), ("zoom", 96), ("end", 75)]
STARTS = np.cumsum([0] + [d for _, d in SCENES]); N = int(STARTS[-1])
def scene_of(f):
    i = int(np.searchsorted(STARTS, f, side="right") - 1); return SCENES[i][0], f - int(STARTS[i]), SCENES[i][1]

class Ctx:
    def __init__(s, fs): s.fs = fs
    def T(s, t): return t.upper() if s.fs["upper"] else t
    def headline(s, d, lines, y, accent_last=True, size=None, maxw=920, gap=1.02):
        size = size or s.fs["hs"]; ls = [s.T(l) for l in lines]
        fh = fit(s.fs["head"], size, ls, maxw); fa = fit(s.fs["accent"], size, ls, maxw)
        for i, l in enumerate(ls):
            acc = accent_last and i == len(ls) - 1 and len(ls) > 1
            d.text((W / 2, y + i * fh.size * gap), l, font=fa if acc else fh, fill=GOLD if acc else WHITE, anchor="mm")
        return y + len(ls) * fh.size * gap
    def body(s, d, lines, y, col=SOFT, size=None, maxw=900, gap=1.25):
        f = fit(s.fs["body"], size or s.fs["bs"], lines, maxw)
        for i, l in enumerate(lines): d.text((W / 2, y + i * f.size * gap), l, font=f, fill=col, anchor="mm")
    def chip(s, d, x, y, text, col=WHITE, size=30, anchor="lt", bold=True, pad=14, bg=(8, 12, 18, 200)):
        f = font(s.fs["monob" if bold else "mono"], size); tw = f.getlength(text); h = size + pad * 1.2
        x0 = x if anchor[0] == "l" else (x - tw - 2 * pad if anchor[0] == "r" else x - tw / 2 - pad)
        y0 = y if anchor[1] == "t" else y - h / 2
        d.rounded_rectangle((x0, y0, x0 + tw + 2 * pad, y0 + h), radius=h / 2, fill=bg)
        d.text((x0 + pad, y0 + h / 2), text, font=f, fill=col, anchor="lm")

def paste_img(img, arr_or_img, x, y, w, h, radius=26, alpha=1.0, outline=None):
    im = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(np.asarray(arr_or_img))
    if im.size != (w, h): im = im.resize((w, h), Image.LANCZOS)
    m = Image.new("L", (w, h), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=int(255 * alpha))
    img.paste(im, (x, y), m)
    if outline: ImageDraw.Draw(img).rounded_rectangle((x - 2, y - 2, x + w + 1, y + h + 1), radius=radius + 2, outline=outline, width=4)

def hero_full(img, y0, zoom=1.0, dim=0.0):
    h = int(1080 * 675 / 1080); im = HERO_IMG
    if zoom != 1.0:
        cw, ch = 1080 / zoom, 675 / zoom; im = im.crop(((1080 - cw) / 2, (675 - ch) / 2, (1080 + cw) / 2, (675 + ch) / 2)).resize((1080, 675), Image.LANCZOS)
    if dim > 0: im = Image.blend(im, Image.new("RGB", im.size, BG), dim)
    img.paste(im, (0, y0))
    fade = Image.new("L", (1, 90)); [fade.putpixel((0, i), int(255 * (1 - i / 89))) for i in range(90)]
    fade = fade.resize((1080, 90)); top = Image.new("RGBA", (1080, 90), BG + (255,)); top.putalpha(fade)
    img.alpha_composite(top, (0, y0)); img.alpha_composite(top.transpose(Image.FLIP_TOP_BOTTOM), (0, y0 + 675 - 90))

HERO_Y = 520
def s_hook(c, rf, n):
    img = BGIMG.copy(); hero_full(img, HERO_Y, zoom=1.0 + 0.05 * rf / n); d = ImageDraw.Draw(img, "RGBA")
    a = ease(rf / 6)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
    c.headline(ld, ["Esto lo ha pintado", "una red neuronal"], 330)
    c.body(ld, ["Solo sabe responder una pregunta:", "¿de qué color es el punto (x, y)?"], 1290)
    ld.text((W / 2, 1210), "La noche estrellada (Van Gogh, 1889) · versión de la red", font=font(c.fs["mono"], 24), fill=MUTED, anchor="mm")
    lay.putalpha(lay.getchannel("A").point(lambda v: int(v * a))); img.alpha_composite(lay)
    return img

def s_how(c, rf, n):
    img = BGIMG.copy(); hero_full(img, HERO_Y, zoom=1.05, dim=0.45); d = ImageDraw.Draw(img, "RGBA")
    c.headline(d, ["Le das una posición…", "…y te dice un color"] if rf > 20 else ["Le das una posición…"], 330)
    t = rf / FPS; u = 0.5 + 0.36 * math.sin(2 * math.pi * 0.42 * t + 0.6); v = 0.5 + 0.30 * math.sin(2 * math.pi * 0.67 * t + 1.9)
    px, py = u * 1080, HERO_Y + v * 675; col = tuple(int(x) for x in HERO[min(674, int(v * 675)), min(1079, int(u * 1080))])
    for r_, a_ in ((34, 90), (20, 255)): d.ellipse((px - r_, py - r_, px + r_, py + r_), outline=WHITE + (a_,), width=3)
    d.line((px - 52, py, px - 22, py), fill=WHITE, width=3); d.line((px + 22, py, px + 52, py), fill=WHITE, width=3)
    d.line((px, py - 52, px, py - 22), fill=WHITE, width=3); d.line((px, py + 22, px, py + 52), fill=WHITE, width=3)
    # ecuación bajo la imagen
    fy = 1300; f = font(c.fs["monob"], 54); txt = f"({dec(u, 2)} ; {dec(v, 2)})"
    tw = f.getlength(txt); x0 = W / 2 - (tw + 200) / 2
    d.text((x0, fy), txt, font=f, fill=WHITE, anchor="lm"); d.text((x0 + tw + 40, fy), "→", font=f, fill=GOLD, anchor="lm")
    d.ellipse((x0 + tw + 120, fy - 46, x0 + tw + 212, fy + 46), fill=col, outline=WHITE, width=5)
    d.text((W / 2, 1405), f"{fmt(NPIX)} preguntas así = un cuadro", font=font(c.fs["mono"], 34), fill=SOFT, anchor="mm")
    return img

SLOT = (140, 462, 800, 500); SLOT2 = (140, 978, 800, 500)
def race_idx(rf, n): return int(round(min(1.0, rf / (n - 36)) * (len(IDX) - 1)))
def s_race(c, rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    c.headline(d, ["Dos redes idénticas.", "Solo cambia la entrada"], 292, size=int(c.fs["hs"] * 0.78))
    j = race_idx(rf, n); si = IDX[j]; it = KEYS[si]; vp, vf = PS["plain"][si][1], PS["fourier"][si][1]
    for (x, y, w, h), arr, lab, val, col, stamp in ((SLOT, HDP[j], ("Red normal", " · (x, y)"), vp, WHITE, "se queda borrosa"),
                                                    (SLOT2, HDF[j], ("Red + Fourier", " · ondas"), vf, GOLD, "nítida")):
        paste_img(img, arr, x, y, w, h, outline=GOLD if col == GOLD else None); d = ImageDraw.Draw(img, "RGBA")
        c.chip(d, x + 18, y + 18, c.T(lab[0]) + lab[1], col=col, size=28)
        c.chip(d, x + w - 18, y + 18, f"{dec(val)} dB", col=col, size=36, anchor="rt")
        if rf > n - 34:
            a = ease((rf - (n - 34)) / 8); f = font(c.fs["accent"], int(c.fs["hs"] * 0.62)); s_ = c.T(stamp); tw = f.getlength(s_)
            d.rounded_rectangle((x + w / 2 - tw / 2 - 26, y + h - 120, x + w / 2 + tw / 2 + 26, y + h - 30), radius=20, fill=(8, 12, 18, int(210 * a)))
            d.text((x + w / 2, y + h - 75), s_, font=f, fill=col + (int(255 * a),), anchor="mm")
    c.chip(d, W / 2, (SLOT[1] + SLOT[3] + SLOT2[1]) / 2, f"iteración {fmt(it)}", col=WHITE, size=30, anchor="mm", bg=(30, 40, 54, 255))
    return img

def s_trick(c, rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    c.headline(d, ["El truco:", "darle la posición en ondas"], 320, size=int(c.fs["hs"] * 0.86))
    freqs = [1, 3, 7, 15, 31]; x0, x1 = 210, 960; t = rf / FPS
    cx = x0 + (x1 - x0) * (0.5 + 0.45 * math.sin(2 * math.pi * 0.35 * t - 1.2))
    for k, fr in enumerate(freqs):
        yc = 620 + k * 140; amp = 46; a_in = ease((rf - k * 5) / 10)
        if a_in <= 0: continue
        d.text((x0 - 30, yc), f"{fr}×", font=font(c.fs["monob"], 34), fill=SOFT + (int(255 * a_in),), anchor="rm")
        xs = np.linspace(x0, x1, 400); ys = yc - amp * np.sin(2 * np.pi * fr * (xs - x0) / (x1 - x0) * 0.5 + rf * 0.04 * (k + 1))
        col = GOLD if k >= 2 else BLUE
        d.line(list(zip(xs, ys)), fill=col + (int(255 * a_in),), width=4, joint="curve")
        vy = yc - amp * np.sin(2 * np.pi * fr * (cx - x0) / (x1 - x0) * 0.5 + rf * 0.04 * (k + 1))
        d.ellipse((cx - 11, vy - 11, cx + 11, vy + 11), fill=WHITE + (int(255 * a_in),))
    d.line((cx, 560, cx, 620 + 4 * 140 + 60), fill=WHITE + (120,), width=2)
    c.body(d, ["Las ondas rápidas le permiten", "dibujar el detalle fino."], 1318)
    d.text((W / 2, 1468), "(en realidad usa 96 ondas de frecuencias al azar)", font=font(c.fs["mono"], 26), fill=MUTED, anchor="mm")
    return img

def s_zoom(c, rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    c.headline(d, ["Ampliamos ×8"], 330, size=int(c.fs["hs"] * 0.86))
    m = ease((rf - 14) / 12)
    x, y, w, h = SLOT; hero = HERO_IMG.resize((w, h), Image.LANCZOS)
    top = hero if m <= 0 else Image.blend(hero, Image.fromarray(ZPIX).resize((w, h), Image.NEAREST), m)
    paste_img(img, top, x, y, w, h); d = ImageDraw.Draw(img, "RGBA")
    if m < 1:
        bx0, by0, bx1, by1 = META["BOX"]; a = 1 - m
        d.rounded_rectangle((x + bx0 * w, y + by0 * h, x + bx1 * w, y + by1 * h), radius=6, outline=GOLD + (int(255 * a),), width=5)
    if m > 0:
        x2, y2, w2, h2 = SLOT2; paste_img(img, ZNET, x2, y2, w2, h2, alpha=m, outline=GOLD if m > .9 else None); d = ImageDraw.Draw(img, "RGBA")
        c.chip(d, x + 18, y + 18, c.T("Los píxeles: se rompen"), col=WHITE, size=30)
        c.chip(d, x2 + 18, y2 + 18, c.T("La red: sigue continua"), col=GOLD, size=30)
        if rf > 40:
            a2 = ease((rf - 40) / 8); c.chip(d, x2 + w2 / 2, y2 + h2 - 52, "no inventa detalle: rellena entre lo que aprendió", col=SOFT + (int(255 * a2),), size=24, anchor="mm", bold=False, bg=(8, 12, 18, int(210 * a2)))
    return img

def s_end(c, rf, n):
    img = BGIMG.copy(); hero_full(img, HERO_Y, zoom=1.0 + 0.05 * (1 - rf / n), dim=0.15); d = ImageDraw.Draw(img, "RGBA")
    c.headline(d, ["¿Qué cuadro", "le pongo?"], 330)
    c.body(d, ["Comenta una obra de dominio público"], 1270)
    d.text((W / 2, 1350), c.T("Redes Neuronales · Ep. 1"), font=font(c.fs["monob"], 32), fill=GOLD, anchor="mm")
    d.text((W / 2, 1420), HANDLE, font=font(c.fs["head"], int(c.fs["hs"] * 0.55)), fill=WHITE, anchor="mm")
    return img

def s_curve(c, rf, n):
    import reel_common as RC
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    c.headline(d, ["La diferencia,", "en una curva"], 320, size=int(c.fs["hs"] * 0.9))
    up = 10 ** (np.log10(ITERS) * ease((rf - 4) / 56)) if rf > 4 else 0
    sp = np.array(PS["plain"]); sf = np.array(PS["fourier"])
    RC.chart(img, (90, 470, 990, 1010), [dict(xs=sp[:, 0], ys=sp[:, 1], upto=up, col=WHITE, name="normal", val_fmt=lambda v: dec(v) + " dB"),
                                         dict(xs=sf[:, 0], ys=sf[:, 1], upto=up, col=GOLD, name="Fourier", val_fmt=lambda v: dec(v) + " dB")],
             "calidad del cuadro (dB)", xmax=ITERS, ylim=(10, 36), yticks=[15, 25, 35], logy=False, yfmt=lambda v: str(int(v)))
    d = ImageDraw.Draw(img, "RGBA")
    if rf > 50:
        a = ease((rf - 50) / 10); lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
        for x, arr, lab, col in ((90, HDP[len(IDX) - 1], "normal", WHITE), (550, HDF[len(IDX) - 1], "Fourier", GOLD)):
            paste_img(lay, arr, x, 1060, 440, 275, radius=18, outline=col if col == GOLD else None)
            c.chip(ImageDraw.Draw(lay, "RGBA"), x + 14, 1074, c.T(lab), col=col, size=26)
        lay.putalpha(lay.getchannel("A").point(lambda v: int(v * a))); img.alpha_composite(lay)
        d = ImageDraw.Draw(img, "RGBA")
    c.body(d, ["Más dB = más parecido al original"], 1400, size=40)
    return img

FN = dict(curve=s_curve, hook=s_hook, how=s_how, race=s_race, trick=s_trick, zoom=s_zoom, end=s_end)
def frame(c, f):
    name, rf, n = scene_of(f); img = FN[name](c, rf, n)
    if rf < 6 and f > 0:                       # «punch-in» en cada corte
        s = 1 + 0.035 * (1 - ease(rf / 6)); cw, ch = W / s, H / s
        img = img.crop(((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2)).resize((W, H), Image.LANCZOS)
    return img.convert("RGB")

def audio(path):
    sr = 44100; n = int(N / FPS * sr); t = np.arange(n) / sr; a = np.zeros(n); rng = np.random.default_rng(1)
    env = np.clip(t / 0.6, 0, 1) * np.clip((N / FPS - t) / 0.8, 0, 1)
    for fr in (110, 164.8, 220, 277.2): a += 0.035 * env * np.sin(2 * np.pi * fr * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.25 * t))
    def at(fr_, arr, g):
        i0 = int(fr_ / FPS * sr); m = min(len(arr), n - i0)
        if m > 0: a[i0:i0 + m] += g * arr[:m]
    m = int(0.5 * sr); tt = np.arange(m) / sr
    whoosh = np.convolve(rng.normal(0, 1, m), np.ones(30) / 30, "same") * np.sin(np.pi * tt / 0.5) ** 2
    for s0 in STARTS[1:-1]: at(s0 - 6, whoosh, 0.22)
    hit = np.exp(-tt * 9) * np.sin(2 * np.pi * 55 * tt); at(0, hit, 0.5)
    r0, rn = int(STARTS[2]), SCENES[2][1]                          # tono que sube con la calidad de la red Fourier
    q = np.array([PS["fourier"][IDX[race_idx(rf, rn)]][1] for rf in range(rn)])
    fr = 196 * 2 ** ((np.interp(np.arange(int(rn / FPS * sr)) / sr * FPS, np.arange(rn), q) - 12) / 14)
    seg = 0.05 * np.sin(2 * np.pi * np.cumsum(fr) / sr) * np.clip(np.arange(len(fr)) / (0.3 * sr), 0, 1); at(r0, seg, 1.0)
    tk = np.exp(-np.arange(int(.08 * sr)) / (0.01 * sr)) * np.sin(2 * np.pi * 1200 * np.arange(int(.08 * sr)) / sr)
    for k in range(0, SCENES[1][1], 9): at(STARTS[1] + k, tk, 0.06)
    e0 = int(STARTS[[nm for nm, _ in SCENES].index("end")]); tt3 = np.arange(int(2.5 * sr)) / sr
    at(e0, sum(np.exp(-tt3 * 2.2) * np.sin(2 * np.pi * f_ * tt3) for f_ in (784, 1174.7, 1568)), 0.08)
    a = a / np.abs(a).max() * 0.85
    with wave.open(path, "w") as wv:
        wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(sr); wv.writeframes((a * 32767).astype(np.int16).tobytes())

if __name__ == "__main__":
    key = sys.argv[1]; c = Ctx(FONTSETS[key])
    if len(sys.argv) > 2 and sys.argv[2] == "prev":
        for f in map(int, sys.argv[3:]): frame(c, f).save(f"work/r{key}_{f}.png")
        sys.exit()
    import os, time
    if len(sys.argv) > 2 and sys.argv[2] == "chunk":
        out = f"work/frames_{key}"; os.makedirs(out, exist_ok=True); t0 = time.time(); budget = float(sys.argv[3])
        for f in range(N):
            p_ = f"{out}/{f:04d}.jpg"
            if os.path.exists(p_): continue
            if time.time() - t0 > budget: print("pausa", f, N); sys.exit(0)
            frame(c, f).save(p_, quality=95)
        print("completo", N); sys.exit(0)
    if len(sys.argv) > 2 and sys.argv[2] == "encode":
        audio(f"work/reel_{key}.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", f"work/frames_{key}/%04d.jpg", "-i", f"work/reel_{key}.wav",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-b:a", "160k", "-shortest", f"out/ep1b_reel_fuente_{key}.mp4"])
        sys.exit(0)
    tmp = f"work/reel_{key}.mp4"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", tmp], stdin=subprocess.PIPE)
    for f in range(N): p.stdin.write(frame(c, f).tobytes())
    p.stdin.close(); p.wait(); audio(f"work/reel_{key}.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-i", f"work/reel_{key}.wav", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                    "-shortest", f"out/ep1b_reel_fuente_{key}.mp4"])
