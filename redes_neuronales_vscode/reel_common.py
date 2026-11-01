"""Piezas comunes de los Reels de «Redes Neuronales» (tipografía C: Archivo Black + Inter)."""
import numpy as np, subprocess, wave, os, sys, time
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
FD = "fonts/"
FS = dict(head=(FD + "archivoblack_ArchivoBlack-Regular.ttf", {}), accent=(FD + "archivoblack_ArchivoBlack-Regular.ttf", {}),
          body=(FD + "inter_Inter.ttf", {"Weight": 500}), mono=(FD + "inter_Inter.ttf", {"Weight": 600}),
          monob=(FD + "inter_Inter.ttf", {"Weight": 800}), hs=86, bs=46, upper=True)
BG = (8, 12, 18); GOLD = (244, 190, 78); BLUE = (110, 168, 255); WHITE = (238, 242, 246); SOFT = (196, 206, 218); MUTED = (140, 154, 170)
GREEN = (96, 214, 140); RED = (240, 96, 96)
# MARCA es la identidad del canal (cabecera y progreso); RED sigue significando error.
# Separados a propósito: si fueran el mismo color, un titular de marca parecería una alarma.
MARCA = (235, 58, 52)
HANDLE = "@rojocuantico"
SAFE_TOP, SAFE_BOTTOM = 260, 1480          # Instagram tapa por encima/debajo
ease = lambda t: 0 if t <= 0 else 1 if t >= 1 else t * t * (3 - 2 * t)
fmt = lambda n: f"{n:,}".replace(",", ".")
dec = lambda v, k=1: f"{v:.{k}f}".replace(".", ",")

_fc = {}
def font(spec, size):
    key = (spec[0], tuple(sorted(spec[1].items())), size)
    if key in _fc: return _fc[key]
    f = ImageFont.truetype(spec[0], size)
    try:
        vals = []
        for a in f.get_variation_axes():
            nm = a["name"].decode() if isinstance(a["name"], bytes) else a["name"]
            vals.append(max(a["minimum"], min(a["maximum"], size / 2.2)) if nm == "Optical size" else spec[1].get(nm, a["default"]))
        f.set_variation_by_axes(vals)
    except Exception: pass
    _fc[key] = f; return f
def fit(spec, size, lines, maxw):
    while size > 20 and max(font(spec, size).getlength(l) for l in lines) > maxw: size -= 2
    return font(spec, size)

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
_g = np.exp(-(((xx - W / 2) / 650) ** 2 + ((yy - 820) / 900) ** 2))
BGIMG = Image.fromarray(np.stack([BG[i] + _g * c for i, c in enumerate((10, 16, 30))], -1).clip(0, 255).astype(np.uint8)).convert("RGBA")

T_ = lambda t: t.upper() if FS["upper"] else t
def headline(d, lines, y, size=None, maxw=920, gap=1.04, accent_last=True):
    ls = [T_(l) for l in lines]; f = fit(FS["head"], size or FS["hs"], ls, maxw)
    for i, l in enumerate(ls):
        acc = accent_last and i == len(ls) - 1 and len(ls) > 1
        d.text((W / 2, y + i * f.size * gap), l, font=f, fill=GOLD if acc else WHITE, anchor="mm")
    return y + len(ls) * f.size * gap
def body(d, lines, y, col=SOFT, size=None, maxw=900, gap=1.25):
    f = fit(FS["body"], size or FS["bs"], lines, maxw)
    for i, l in enumerate(lines): d.text((W / 2, y + i * f.size * gap), l, font=f, fill=col, anchor="mm")
def small(d, text, y, col=MUTED, size=26, x=W / 2, anchor="mm"):
    d.text((x, y), text, font=font(FS["mono"], size), fill=col, anchor=anchor)
def chip(d, x, y, text, col=WHITE, size=30, anchor="lt", bold=True, pad=14, bg=(8, 12, 18, 210)):
    f = font(FS["monob" if bold else "mono"], size); tw = f.getlength(text); h = size + pad * 1.2
    x0 = x if anchor[0] == "l" else (x - tw - 2 * pad if anchor[0] == "r" else x - tw / 2 - pad)
    y0 = y if anchor[1] == "t" else y - h / 2
    d.rounded_rectangle((x0, y0, x0 + tw + 2 * pad, y0 + h), radius=h / 2, fill=bg)
    d.text((x0 + pad, y0 + h / 2), text, font=f, fill=col, anchor="lm")
def paste_img(img, src, x, y, w, h, radius=22, alpha=1.0, outline=None, owidth=4, resample=Image.LANCZOS):
    im = src if isinstance(src, Image.Image) else Image.fromarray(np.asarray(src))
    if im.size != (int(w), int(h)): im = im.resize((int(w), int(h)), resample)
    m = Image.new("L", im.size, 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, im.size[0] - 1, im.size[1] - 1), radius=radius, fill=int(255 * alpha))
    img.paste(im.convert("RGB"), (int(x), int(y)), m)
    if outline: ImageDraw.Draw(img).rounded_rectangle((x - 2, y - 2, x + w + 1, y + h + 1), radius=radius + 2, outline=outline, width=owidth)
def shade(img, box, rgba):
    """Rectángulo semitransparente bien mezclado sobre una imagen RGBA."""
    lay = Image.new("RGBA", img.size, (0, 0, 0, 0)); ImageDraw.Draw(lay).rectangle(box, fill=rgba); img.alpha_composite(lay)
    return ImageDraw.Draw(img, "RGBA")
def faded(lay, a):
    lay.putalpha(lay.getchannel("A").point(lambda v: int(v * a))); return lay

class Timeline:
    def __init__(self, scenes, fns): self.sc = scenes; self.st = np.cumsum([0] + [d for _, d in scenes]); self.N = int(self.st[-1]); self.fns = fns
    def start(self, name): return int(self.st[[n for n, _ in self.sc].index(name)])
    def frame(self, f):
        i = int(np.searchsorted(self.st, f, side="right") - 1); name, n = self.sc[i]; rf = f - int(self.st[i])
        img = self.fns[name](rf, n)
        if rf < 6 and f > 0:                                     # «punch-in» en cada corte
            s = 1 + 0.035 * (1 - ease(rf / 6)); cw, ch = W / s, H / s
            img = img.crop(((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2)).resize((W, H), Image.LANCZOS)
        return img.convert("RGB")

class Audio:
    def __init__(self, nframes, sr=44100):
        self.sr = sr; self.n = int(nframes / FPS * sr); self.t = np.arange(self.n) / sr; self.a = np.zeros(self.n); self.rng = np.random.default_rng(1)
    def pad(self, freqs=(110, 164.8, 220, 277.2), g=0.035):
        env = np.clip(self.t / 0.6, 0, 1) * np.clip((self.t[-1] - self.t) / 0.8, 0, 1)
        for fr in freqs: self.a += g * env * np.sin(2 * np.pi * fr * self.t) * (1 + 0.3 * np.sin(2 * np.pi * 0.25 * self.t))
    def at(self, frame, arr, g):
        i0 = int(frame / FPS * self.sr); m = min(len(arr), self.n - i0)
        if m > 0 and i0 >= 0: self.a[i0:i0 + m] += g * arr[:m]
    def tone(self, fr, dur, decay): tt = np.arange(int(dur * self.sr)) / self.sr; return np.exp(-tt * decay) * np.sin(2 * np.pi * fr * tt)
    def whoosh(self, dur=0.5):
        m = int(dur * self.sr); tt = np.arange(m) / self.sr
        return np.convolve(self.rng.normal(0, 1, m), np.ones(30) / 30, "same") * np.sin(np.pi * tt / dur) ** 2
    def chime(self, frame, g=0.08): self.at(frame, sum(self.tone(f, 2.5, 2.2) for f in (784, 1174.7, 1568)), g)
    def save(self, path):
        a = self.a / (np.abs(self.a).max() + 1e-9) * 0.85
        with wave.open(path, "w") as wv:
            wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(self.sr); wv.writeframes((a * 32767).astype(np.int16).tobytes())

def cli(tl, name, audio_fn):
    """python3 x.py prev f1 f2 | chunk SEG | encode"""
    mode = sys.argv[1] if len(sys.argv) > 1 else "chunk"
    if mode == "prev":
        for f in map(int, sys.argv[2:]): tl.frame(f).save(f"work/{name}_{f}.png")
    elif mode == "chunk":
        out = f"work/frames_{name}"; os.makedirs(out, exist_ok=True); t0 = time.time(); budget = float(sys.argv[2]) if len(sys.argv) > 2 else 85
        for f in range(tl.N):
            p = f"{out}/{f:04d}.jpg"
            if os.path.exists(p): continue
            if time.time() - t0 > budget: print("pausa", f, tl.N); return
            tl.frame(f).save(p, quality=95)
        print("completo", tl.N)
    elif mode == "encode":
        wav = f"work/{name}.wav"; audio_fn().save(wav)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", f"work/frames_{name}/%04d.jpg", "-i", wav,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-b:a", "160k", "-shortest", f"out/{name}.mp4"])
        print("vídeo:", f"out/{name}.mp4")

# ---------------- gráfica de pérdida legible en móvil ----------------
from PIL import ImageFilter
def chart(img, box, series, title, xmax, ylim, yticks, logx=True, logy=True, xlabel="iteración", yfmt=None, legend_right=True):
    """series: dicts {xs, ys, upto, col, name, val_fmt}. Dibuja hasta x<=upto. Devuelve ImageDraw."""
    x0, y0, x1, y1 = box; d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle(box, radius=24, fill=(16, 22, 32, 255), outline=(44, 58, 74, 255), width=2)
    d.text((x0 + 28, y0 + 34), title, font=font(FS["mono"], 26), fill=SOFT, anchor="lm")
    px0, px1, py0, py1 = x0 + 92, x1 - 36, y0 + 92, y1 - 58
    fx = (lambda v: np.log10(max(v, 1)) / np.log10(xmax)) if logx else (lambda v: v / xmax)
    fy = (lambda v: (np.log10(max(v, 1e-9)) - np.log10(ylim[0])) / (np.log10(ylim[1]) - np.log10(ylim[0]))) if logy else (lambda v: (v - ylim[0]) / (ylim[1] - ylim[0]))
    X = lambda v: px0 + np.clip(fx(v), 0, 1) * (px1 - px0); Y = lambda v: py1 - np.clip(fy(v), 0, 1) * (py1 - py0)
    yfmt = yfmt or (lambda v: dec(v, 2 if v < 1 else 0) if v != int(v) else str(int(v)))
    for v in yticks:
        d.line((px0, Y(v), px1, Y(v)), fill=(36, 48, 62, 255), width=2); d.text((px0 - 14, Y(v)), yfmt(v), font=font(FS["mono"], 22), fill=MUTED, anchor="rm")
    d.text((px0, py1 + 30), "1" if logx else "0", font=font(FS["mono"], 22), fill=MUTED, anchor="lm")
    d.text((px1, py1 + 30), fmt(xmax), font=font(FS["mono"], 22), fill=MUTED, anchor="rm")
    d.text(((px0 + px1) / 2, py1 + 30), xlabel + (" (escala log)" if logx else ""), font=font(FS["mono"], 22), fill=MUTED, anchor="mm")
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0)); gd = ImageDraw.Draw(glow); lines = []
    for s in series:
        m = np.asarray(s["xs"]) <= s["upto"]; xs, ys = np.asarray(s["xs"])[m], np.asarray(s["ys"])[m]
        if len(xs) < 1: continue
        pts = [(X(a), Y(b)) for a, b in zip(xs, ys)]; lines.append((s, pts, ys[-1]))
        if len(pts) > 1: gd.line(pts, fill=s["col"] + (140,), width=16, joint="curve")
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(8))); d = ImageDraw.Draw(img, "RGBA")
    for s, pts, last in lines:
        if len(pts) > 1: d.line(pts, fill=s["col"] + (255,), width=6, joint="curve")
        ex, ey = pts[-1]; d.ellipse((ex - 10, ey - 10, ex + 10, ey + 10), fill=s["col"], outline=WHITE, width=3)
    lx = x1 - 28
    for s, pts, last in reversed(lines):                       # leyenda con el valor actual, arriba a la derecha
        txt = f'{s["name"]} {s.get("val_fmt", lambda v: dec(v, 3))(last)}'; f = font(FS["monob"], 26)
        d.text((lx, y0 + 34), txt, font=f, fill=s["col"], anchor="rm"); lx -= f.getlength(txt) + 26
        d.ellipse((lx + 4, y0 + 26, lx + 20, y0 + 42), fill=s["col"]); lx -= 22
    return d
