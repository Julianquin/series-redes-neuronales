"""Motor de los episodios largos (2–3 min) de «Redes Neuronales».
Marco fijo: navegación de la serie, título, sección, panel, tarjeta (usos / a tener en cuenta) y subtítulos.
Piezas: Formula (LaTeX con matplotlib, término a término), CodeBlock (Pygments, resaltado de líneas), Timeline (dirigida por la voz).
Tiempos: si existe voz/timings.json (tts.py) se usan los reales; si no, se estiman a WPS palabras por segundo."""
import json, os, re, sys, subprocess, time, unicodedata, wave
import numpy as np, yaml
from PIL import Image, ImageDraw, ImageFont
import matplotlib; matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reel_common import (W, H, FPS, FS, BG, GOLD, BLUE, WHITE, SOFT, MUTED, GREEN, RED, HANDLE, BGIMG, font, fit, ease,
                         fmt, dec, chip, paste_img, shade, faded, chart, Audio)

HERE = os.path.dirname(os.path.abspath(__file__))
JBM = os.path.join(os.path.dirname(HERE), "fonts", "JBM.ttf")
WPS = 2.4            # palabras por segundo estimadas (se sustituye por la voz real)
PAUSE = 0.45         # silencio tras cada segmento
SERIE = ["MLP", "CNN", "RNN", "Transf.", "AE", "GAN", "GNN", "DBN"]
PANEL = (60, 500, 1020, 1180); CARD = (60, 1200, 1020, 1312); SUB_Y = 1388

def norm(w): return unicodedata.normalize("NFC", re.sub(r"[^\wáéíóúüñÁÉÍÓÚÜÑ]", "", w.lower()))

# ---------------- línea de tiempo ----------------
class Segment:
    def __init__(self, d, start, dur, wtimes):
        self.id, self.scene, self.section, self.text = d["id"], d["escena"], d.get("seccion", ""), d["texto"]
        self.cues_def = d.get("cues", {}); self.start, self.dur = start, dur
        self.words = self.text.split(); self.wt = wtimes
    def cue(self, name):
        """Instante (s, relativo al segmento) en que se pronuncia la palabra asociada al cue."""
        target = norm(self.cues_def[name])
        for w, t in zip(self.words, self.wt):
            if norm(w) == target or norm(w).startswith(target): return t
        raise KeyError(f"cue '{name}' ('{target}') no está en {self.id}")
    def a(self, t, name, d=0.45, off=0.0): return ease((t - self.cue(name) - off) / d)

class Timeline:
    def __init__(self, guion_path, tail=2.6):
        self.G = yaml.safe_load(open(guion_path)); tim = {}
        self.vdir = os.path.join(HERE, "voz", os.path.splitext(os.path.basename(guion_path))[0])   # una carpeta de voz por episodio
        tp = os.path.join(self.vdir, "timings.json")
        if os.path.exists(tp): tim = json.load(open(tp)); print("tiempos reales de la voz:", tp)
        self.segs = []; t = 0.0
        for d in self.G["segmentos"]:
            n = len(d["texto"].split())
            if d["id"] in tim:
                dur = tim[d["id"]]["dur"] + PAUSE; wt = tim[d["id"]].get("words") or list(np.linspace(0.1, dur - PAUSE - 0.2, n))
            else:
                dur = n / WPS + PAUSE; wt = list(np.linspace(0.15, dur - PAUSE - 0.25, n))
            self.segs.append(Segment(d, t, dur, wt)); t += dur
        self.talk_end = t; self.T = t + tail; self.N = int(round(self.T * FPS))
        self.keys = {norm(k) for k in self.G.get("clave", [])}
    def at(self, t):
        for s in self.segs:
            if s.start <= t < s.start + s.dur: return s
        return None
    def subtitle(self, t):
        """Bloques de ≤ 7 palabras cortados en signos de puntuación; devuelve (palabras, alfa)."""
        s = self.at(t)
        if s is None: return None, 0
        rel = t - s.start; blocks, cur = [], []
        for i, w in enumerate(s.words):
            cur.append(i)
            if len(cur) >= 7 or w[-1] in ".:;?!" or (w[-1] == "," and len(cur) >= 4): blocks.append(cur); cur = []
        if cur: blocks.append(cur)
        for b in blocks:
            t0 = s.wt[b[0]] - 0.12; t1 = (s.wt[b[-1]] + 0.45)
            if t0 <= rel < t1: return [s.words[i] for i in b], min(1, (rel - t0) / 0.12, (t1 - rel) / 0.12)
        return None, 0

# ---------------- fórmulas (LaTeX vía matplotlib) ----------------
matplotlib.rcParams["mathtext.fontset"] = "cm"
class Formula:
    """parts: lista de (clave, '$...$' o texto). render(estado) -> imagen RGBA; estado: {clave: (color, alfa)}.
    Las partes ocultas (alfa 0) siguen ocupando su sitio, así la fórmula no «salta» al aparecer término a término."""
    def __init__(self, parts, px=64, gap=6):
        self.parts, self.px, self.gap, self.cache = parts, px, gap, {}
    def render(self, state):
        key = tuple((k, tuple(state.get(k, (WHITE, 1.0))[0]), round(state.get(k, (WHITE, 1.0))[1], 2)) for k, _ in self.parts)
        if key in self.cache: return self.cache[key]
        fig = Figure(figsize=(16, 3), dpi=100); fig.patch.set_alpha(0); cv = FigureCanvasAgg(fig); r = cv.get_renderer()
        x = 10; pt = self.px * 72 / 100; boxes = []
        for k, s in self.parts:
            col, al = state.get(k, (WHITE, 1.0))
            tx = fig.text(x / 1600, 0.5, s, fontsize=pt, color=tuple(c / 255 for c in col), alpha=al, va="baseline", ha="left")
            bb = tx.get_window_extent(r); boxes.append(bb); x = bb.x1 + self.gap
        cv.draw(); im = Image.fromarray(np.asarray(cv.buffer_rgba()).copy())
        x0 = min(b.x0 for b in boxes) - 4; x1 = max(b.x1 for b in boxes) + 4; y0 = min(b.y0 for b in boxes) - 6; y1 = max(b.y1 for b in boxes) + 6
        out = im.crop((int(x0), int(300 - y1), int(x1), int(300 - y0))); self.cache[key] = out; return out
    def draw(self, img, cx, cy, state, max_w=None):
        im = self.render(state)
        if max_w and im.width > max_w: im = im.resize((int(max_w), int(im.height * max_w / im.width)), Image.LANCZOS)
        img.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2))); return im.size

# ---------------- código (Pygments) ----------------
from pygments.lexers import PythonLexer
from pygments.token import Token
TOKCOL = {Token.Keyword: (240, 140, 120), Token.Name.Builtin: (110, 200, 230), Token.Literal.Number: (180, 150, 255),
          Token.Comment: (110, 128, 146), Token.Operator: (244, 190, 78), Token.Literal.String: (140, 214, 150), Token.Name.Function: (110, 200, 230)}
def tok_color(tt):
    while tt is not Token:
        if tt in TOKCOL: return TOKCOL[tt]
        tt = tt.parent
    return (226, 232, 238)
class CodeBlock:
    def __init__(self, code, size=29, lh=1.38):
        self.lines = code.rstrip("\n").split("\n"); self.f = ImageFont.truetype(JBM, size)
        try: self.f.set_variation_by_axes([500])
        except Exception: pass
        self.lh = int(size * lh); self.size = size; self.cw = self.f.getlength("M")
        self.bright = [self._line(i, 1.0) for i in range(len(self.lines))]; self.dim = [self._line(i, 0.28) for i in range(len(self.lines))]
    def _line(self, i, k):
        w = int(self.cw * (len(self.lines[i]) + 6)) + 10; im = Image.new("RGBA", (max(w, 10), self.lh), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        d.text((0, self.lh / 2), f"{i + 1:>2}", font=self.f, fill=tuple(int(c * k) for c in MUTED) + (255,), anchor="lm"); x = self.cw * 3.2
        for tt, v in PythonLexer().get_tokens(self.lines[i]):
            v = v.rstrip("\n")
            if not v: continue
            col = tok_color(tt); d.text((x, self.lh / 2), v, font=self.f, fill=tuple(int(c * k + 16 * (1 - k)) for c in col) + (255,), anchor="lm"); x += self.f.getlength(v)
        return im
    def draw(self, img, x, y, active=None, focus=1.0):
        """active: conjunto de índices de línea (0-based) resaltados; focus: 0..1 cuánto se atenúa el resto."""
        d = ImageDraw.Draw(img, "RGBA")
        for i in range(len(self.lines)):
            yy = y + i * self.lh; on = active is None or i in active
            if active is not None and on and focus > 0:
                d = shade(img, (x - 14, yy + 1, PANEL[2] - 24, yy + self.lh - 1), GOLD + (int(38 * focus),))
                ImageDraw.Draw(img).rectangle((x - 14, yy + 1, x - 9, yy + self.lh - 1), fill=GOLD)
            src = self.bright[i] if on or focus <= 0 else (self.dim[i] if focus >= 1 else Image.blend(self.bright[i], self.dim[i], focus))
            img.alpha_composite(src, (int(x), int(yy)))

# ---------------- marco fijo ----------------
_frame_cache = {}
def frame_base(tl, seg_section, prog):
    G = tl.G; key = (seg_section,)
    if key not in _frame_cache:
        img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
        d.text((W / 2, 262), f"REDES NEURONALES · {HANDLE}", font=font(FS["mono"], 24), fill=MUTED, anchor="mm")
        n = len(SERIE); x0, x1 = 70, 1010; cw = (x1 - x0) / n; act = SERIE.index(G["arquitectura"]) if G["arquitectura"] in SERIE else -1
        for i, nm in enumerate(SERIE):
            cx = x0 + cw * (i + .5); on = i == act
            d.rounded_rectangle((cx - cw / 2 + 4, 292, cx + cw / 2 - 4, 336), radius=12, fill=(40, 32, 14, 255) if on else (18, 25, 33, 255),
                                outline=GOLD if on else (44, 58, 74), width=3 if on else 2)
            d.text((cx, 314), nm, font=font(FS["monob"], 22), fill=GOLD if on else MUTED, anchor="mm")
        head = G.get("cabecera", f"{G['titulo'].upper()} ({G['arquitectura']})")
        d.text((W / 2, 400), head, font=fit(FS["head"], 54, [head], 940), fill=WHITE, anchor="mm")
        if seg_section: d.text((W / 2, 458), seg_section.upper(), font=font(FS["monob"], 26), fill=GOLD, anchor="mm")
        d.rounded_rectangle(PANEL, radius=26, fill=(14, 20, 30, 255), outline=(44, 58, 74, 255), width=2)
        d.rounded_rectangle(CARD, radius=20, fill=(16, 22, 32, 255), outline=(44, 58, 74, 255), width=2)
        c = G.get("tarjeta", {}); fm, fb = font(FS["monob"], 20), font(FS["body"], 26)
        d.text((92, 1236), "USOS", font=fm, fill=GOLD, anchor="lm"); d.text((330, 1236), c.get("usos", ""), font=fb, fill=SOFT, anchor="lm")
        d.line((90, 1256, 990, 1256), fill=(36, 48, 62), width=2)
        d.text((92, 1282), "OJO", font=fm, fill=GOLD, anchor="lm"); d.text((330, 1282), c.get("saber", ""), font=fb, fill=SOFT, anchor="lm")
        _frame_cache[key] = img
    img = _frame_cache[key].copy(); d = ImageDraw.Draw(img)
    d.rounded_rectangle((70, 346, 1010, 350), radius=2, fill=(36, 48, 62)); d.rounded_rectangle((70, 346, 70 + 940 * prog, 350), radius=2, fill=GOLD)
    return img

def draw_sub(img, tl, words, a):
    if not words or a <= 0: return
    f = font(FS["monob"], 44); lines, cur = [], []
    for w in words:
        if cur and f.getlength(" ".join(cur + [w])) > 880: lines.append(cur); cur = []
        cur.append(w)
    lines.append(cur); lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    for li, ln in enumerate(lines[:2]):
        y = SUB_Y + (li - (len(lines[:2]) - 1) / 2) * 56; tw = f.getlength(" ".join(ln)); x = W / 2 - tw / 2
        for w in ln:
            col = GOLD if norm(w) in tl.keys else WHITE
            d.text((x, y), w, font=f, fill=col, anchor="lm", stroke_width=4, stroke_fill=(8, 12, 18)); x += f.getlength(w + " ")
    img.alpha_composite(faded(lay, a))

def make_frame(tl, scenes, f):
    t = f / FPS; s = tl.at(t)
    if s is None:                                   # cola final: tarjeta de cierre
        return scenes["_end"](tl, t - tl.talk_end).convert("RGB")
    img = frame_base(tl, s.section, min(1, t / tl.talk_end)); rel = t - s.start
    a_in = 1.0 if s is tl.segs[0] else ease(rel / 0.35)   # fundido de entrada (salvo el gancho: visible desde el fotograma 0)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); scenes[s.scene](lay, s, rel)
    img.alpha_composite(faded(lay, a_in) if a_in < 1 else lay)
    words, a = tl.subtitle(t); draw_sub(img, tl, words, a)
    return img.convert("RGB")

# ---------------- audio y CLI ----------------
def make_audio(tl):
    A = Audio(tl.N); A.pad(g=0.022)
    for s in tl.segs[1:]: A.at(s.start * FPS - 4, A.whoosh(0.4), 0.10)
    A.chime(tl.talk_end * FPS, 0.07)
    vdir = tl.vdir
    for s in tl.segs:                                # voz real si existe (tts.py)
        p = os.path.join(vdir, f"{s.id}.wav")
        if os.path.exists(p):
            with wave.open(p) as wv:
                sr = wv.getframerate(); x = np.frombuffer(wv.readframes(wv.getnframes()), np.int16).astype(np.float32) / 32768
            if sr != A.sr: x = np.interp(np.arange(0, len(x), sr / A.sr), np.arange(len(x)), x)
            A.at(s.start * FPS, x, 1.6)
    return A

def cli(tl, scenes, name):
    mode = sys.argv[1] if len(sys.argv) > 1 else "chunk"
    if mode == "prev":
        for f in map(int, sys.argv[2:]): make_frame(tl, scenes, f).save(f"work/{name}_{f}.png")
    elif mode == "info":
        for s in tl.segs: print(f"{s.id} {s.scene:8s} {s.start:6.1f}s +{s.dur:5.1f}s  {len(s.words)} palabras")
        print(f"total {tl.T:.1f}s ({tl.N} fotogramas), {sum(len(s.words) for s in tl.segs)} palabras")
    elif mode == "chunk":
        out = f"work/frames_{name}"; os.makedirs(out, exist_ok=True); t0 = time.time(); budget = float(sys.argv[2]) if len(sys.argv) > 2 else 85
        for f in range(tl.N):
            p = f"{out}/{f:05d}.jpg"
            if os.path.exists(p): continue
            if time.time() - t0 > budget: print("pausa", f, tl.N); return
            make_frame(tl, scenes, f).save(p, quality=93)
        print("completo", tl.N)
    elif mode == "encode":
        wav = f"work/{name}.wav"; make_audio(tl).save(wav)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", f"work/frames_{name}/%05d.jpg", "-i", wav, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "19", "-c:a", "aac", "-b:a", "160k", "-shortest", f"out/{name}.mp4"])
        print("vídeo:", f"out/{name}.mp4")
