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
from reel_common import (W, H, FPS, FS, BG, GOLD, BLUE, WHITE, SOFT, MUTED, GREEN, RED, MARCA, HANDLE, BGIMG, font, fit, ease,
                         fmt, dec, chip, paste_img, shade, faded, chart, Audio)

HERE = os.path.dirname(os.path.abspath(__file__))
JBM = os.path.join(os.path.dirname(HERE), "fonts", "JBM.ttf")
WPS = 2.47           # palabras por segundo estimadas (se sustituye por la voz real).
                     # Medido sobre el episodio 01 completo con Qwen3-TTS a velocidad 0.92:
                     # 405 palabras en 163,7 s = 148 palabras/min. Los segmentos de episodio van
                     # más lentos que las frases cortas de un Reel (ahí salían 2,8).
PAUSE = 0.45         # silencio tras cada segmento
SERIE = ["MLP", "CNN", "RNN", "Transf.", "AE", "GAN", "GNN", "DBN"]
PANEL = (60, 500, 1020, 1180); CARD = (60, 1200, 1020, 1312); SUB_Y = 1388

def norm(w): return unicodedata.normalize("NFC", re.sub(r"[^\wáéíóúüñÁÉÍÓÚÜÑ]", "", w.lower()))

# ---------------- línea de tiempo ----------------
class Segment:
    def __init__(self, d, start, dur, wtimes):
        self.id, self.scene, self.section, self.text = d["id"], d["escena"], d.get("seccion", ""), d["texto"]
        self.cues_def = d.get("cues", {}); self.start, self.dur = start, dur
        self.words = self.text.split(); self.wt = wtimes; self.raw = d; self.tl = None
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
        for sg in self.segs: sg.tl = self
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

# ---------------- marco (formato v2: dos pisos) ----------------
# zonas (y): serie+progreso 150–210 · título 225–300 · lema ~338 · PISO 1 (intuición) 400–1100 · subtítulos ~1182 · PISO 2 (formal) 1252–1700
KICK, TITLE, TAG = 168, 262, 338
UP = (60, 400, 1020, 1100); SUBY = 1182; LOW = (60, 1252, 1020, 1700)
OLD_TOP = 480; SHIFT = 100                       # las escenas v1 dibujan en 500–1180: se desplazan 100 px hacia arriba
_frame_cache = {}
def frame_base(tl, si):
    G = tl.G; s = tl.segs[si]
    if si not in _frame_cache:
        img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
        sec = f" · {s.section.upper()}" if s.section else ""
        d.text((70, KICK), f"EP. {G['episodio']} DE 8{sec}", font=font(FS["monob"], 24), fill=MARCA, anchor="lm")
        d.text((1010, KICK), HANDLE, font=font(FS["mono"], 24), fill=MUTED, anchor="rm")
        n = len(tl.segs); x0, x1 = 70, 1010; w = (x1 - x0 - (n - 1) * 6) / n
        for i in range(n): d.rounded_rectangle((x0 + i * (w + 6), 198, x0 + i * (w + 6) + w, 204), radius=3, fill=MARCA if i <= si else (40, 52, 66))
        head = G.get("cabecera", f"{G['titulo'].upper()} ({G['arquitectura']})")
        d.text((W / 2, TITLE), head, font=fit(FS["head"], 74, [head], 940), fill=WHITE, anchor="mm")
        if G.get("lema"): d.text((W / 2, TAG), G["lema"], font=fit(FS["mono"], 30, [G["lema"]], 940), fill=SOFT, anchor="mm")
        d.rounded_rectangle(UP, radius=26, fill=(14, 20, 30, 255), outline=(44, 58, 74, 255), width=2)
        _frame_cache[si] = img
    return _frame_cache[si].copy()

def draw_sub(img, tl, words, a):
    if not words or a <= 0: return
    f = font(FS["monob"], 40); lines, cur = [], []
    for w in words:
        if cur and f.getlength(" ".join(cur + [w])) > 900: lines.append(cur); cur = []
        cur.append(w)
    lines.append(cur); lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    for li, ln in enumerate(lines[:2]):
        y = SUBY + (li - (len(lines[:2]) - 1) / 2) * 50; x = W / 2 - f.getlength(" ".join(ln)) / 2
        for w in ln:
            d.text((x, y), w, font=f, fill=GOLD if norm(w) in tl.keys else WHITE, anchor="lm", stroke_width=4, stroke_fill=(8, 12, 18)); x += f.getlength(w + " ")
    img.alpha_composite(faded(lay, a))

def draw_pill(img, txt, a, gold=False):
    if not txt or a <= 0: return
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay); f = font(FS["monob"], 32); tw = f.getlength(txt); col = GOLD if gold else BLUE
    d.rounded_rectangle((W / 2 - tw / 2 - 30, UP[3] - 30, W / 2 + tw / 2 + 30, UP[3] + 30), radius=30, fill=(40, 32, 14, 255) if gold else (14, 26, 40, 255), outline=col, width=3)
    d.text((W / 2, UP[3]), txt, font=f, fill=col, anchor="mm"); img.alpha_composite(faded(lay, a))

# --- piso 2: tipos genéricos descritos en el guion (piso2: {...} o lista de {..., cue: palabra}) ---
def _low_box(d, title=None, chrome=False):
    x0, y0, x1, y1 = LOW; d.rounded_rectangle(LOW, radius=22, fill=(12, 17, 25, 255), outline=(40, 52, 66, 255), width=2)
    if chrome:
        for k, c in enumerate((RED, GOLD, GREEN)): d.ellipse((x0 + 24 + k * 26, y0 + 22, x0 + 40 + k * 26, y0 + 38), fill=c)
        if title: d.text((x0 + 120, y0 + 30), title, font=font(FS["mono"], 22), fill=MUTED, anchor="lm")
        d.text((x1 - 24, y0 + 30), "python", font=font(FS["mono"], 20), fill=MUTED, anchor="rm")
    elif title: d.text((x0 + 30, y0 + 34), title, font=font(FS["monob"], 22), fill=GOLD, anchor="lm")
_cb_cache = {}
def low_code(img, spec, ex, s, t):
    code = ex["code"]; key = id(code)
    if key not in _cb_cache: _cb_cache[key] = CodeBlock(code, size=25, lh=1.42)
    cb = _cb_cache[key]; d = ImageDraw.Draw(img, "RGBA"); _low_box(d, ex.get("code_title", "codigo.py"), chrome=True)
    act = set(spec.get("lineas", [])); n = len(cb.lines); vis = 10
    first = 0 if not act else int(np.clip(min(act) - 2, 0, max(0, n - vis)))
    x, y0 = LOW[0] + 40, LOW[1] + 60
    for k, i in enumerate(range(first, min(n, first + vis))):
        yy = y0 + k * cb.lh; on = not act or i in act
        if act and on:
            d = shade(img, (x - 14, yy + 1, LOW[2] - 24, yy + cb.lh - 1), GOLD + (38,)); ImageDraw.Draw(img).rectangle((x - 14, yy + 1, x - 9, yy + cb.lh - 1), fill=GOLD)
        img.alpha_composite(cb.bright[i] if on else cb.dim[i], (int(x), int(yy)))
    ov = spec.get("ovalo")
    if ov and (not spec.get("ovalo_cue") or t >= s.cue(spec["ovalo_cue"])):
        for i in range(first, min(n, first + vis)):
            c0 = cb.lines[i].find(ov)
            if c0 >= 0:
                yy = y0 + (i - first) * cb.lh + cb.lh / 2; xa = x + cb.cw * (3.2 + c0) - 8; xb = x + cb.cw * (3.2 + c0 + len(ov)) + 8
                ImageDraw.Draw(img, "RGBA").ellipse((xa, yy - cb.lh * 0.62, xb, yy + cb.lh * 0.62), outline=BLUE, width=4); break
def low_banner(img, spec, ex, s, t):
    items = spec.get("items") or s.tl.G.get("promesas", []); a = ease(t / 0.4)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay); x0, y0, x1, y1 = LOW
    d.rounded_rectangle((x0 - 10, y0, x1 + 10, y1 - 40), radius=26, fill=(14, 10, 8, 240), outline=GOLD, width=6)
    d.line((150, y0 + 70, 350, y0 + 70), fill=GOLD, width=4); d.line((730, y0 + 70, 930, y0 + 70), fill=GOLD, width=4)
    d.text((W / 2, y0 + 70), "EN ESTE VÍDEO", font=font(FS["monob"], 34), fill=GOLD, anchor="mm")
    up = [x.upper() for x in items]; f = fit(FS["head"], 50, up, 860)
    for k, txt in enumerate(up):
        ak = ease((t - 0.3 - 0.25 * k) / 0.3); d.text((W / 2, y0 + 160 + k * 76), txt, font=f, fill=WHITE + (int(255 * ak),), anchor="mm")
    img.alpha_composite(faded(lay, a))
def low_formula(img, spec, ex, s, t):
    d = ImageDraw.Draw(img, "RGBA"); _low_box(d, spec.get("titulo"))
    F = ex["formulas"][spec["nombre"]]; st = {k: (WHITE, 1.0) for k, _ in F.parts}
    for k in spec.get("resalta", []): st[k] = (GOLD, 1.0)
    rows = spec.get("leyenda", []); cy = LOW[1] + (150 if rows else 220)
    F.draw(img, W / 2, cy, st, max_w=880); d = ImageDraw.Draw(img, "RGBA")
    for k, (sym, txt) in enumerate(rows):
        y = LOW[1] + 270 + k * 52; d.text((140, y), sym, font=font(FS["monob"], 30), fill=GOLD, anchor="lm"); d.text((330, y), txt, font=fit(FS["body"], 30, [txt], 620), fill=SOFT, anchor="lm")
def low_list(img, spec, ex, s, t):
    d = ImageDraw.Draw(img, "RGBA"); _low_box(d, spec.get("titulo"))
    for k, (a_, b_) in enumerate(spec.get("filas", [])):
        y = LOW[1] + 100 + k * 64; d.text((100, y), a_, font=font(FS["monob"], 28), fill=GOLD, anchor="lm")
        d.text((980, y), b_, font=fit(FS["body"], 28, [b_], 520), fill=SOFT, anchor="rm")
def low_ficha(img, spec, ex, s, t):
    G = s.tl.G; d = ImageDraw.Draw(img, "RGBA"); _low_box(d, f"FICHA · {G['arquitectura']}")
    cols = {"verde": GREEN, "oro": GOLD, "rojo": RED, "azul": BLUE, "gris": SOFT}
    for k, row in enumerate(G.get("ficha", [])):
        lab, txt = row[0], row[1]; col = cols.get(row[2] if len(row) > 2 else "gris", SOFT); y = LOW[1] + 84 + k * 70
        if k: d.line((90, y - 34, 990, y - 34), fill=(36, 48, 62), width=2)
        d.text((92, y - 12), lab, font=font(FS["monob"], 20), fill=col, anchor="lm"); d.text((92, y + 18), txt, font=fit(FS["body"], 30, [txt], 880), fill=WHITE, anchor="lm")
    d.text((LOW[2] - 30, LOW[1] + 34), "guárdalo ↗", font=font(FS["mono"], 20), fill=MUTED, anchor="rm")
def low_curve(img, spec, ex, s, t): ex["curves"][spec["nombre"]](img, LOW, s, t)
LOW_TYPES = dict(codigo=low_code, banner=low_banner, formula=low_formula, lista=low_list, ficha=low_ficha, curva=low_curve)
def draw_low(img, s, t, ex):
    spec = s.raw.get("piso2")
    if not spec: return
    specs = spec if isinstance(spec, list) else [spec]; cur = specs[0]
    for sp in specs:
        if not sp.get("cue") or t >= s.cue(sp["cue"]) - 0.1: cur = sp
    LOW_TYPES[cur["tipo"]](img, cur, ex, s, t)

def make_frame(tl, scenes, f):
    if tl.G.get("formato", 1) < 2:                       # guiones sin «formato: 2»: diseño anterior (un panel + tarjeta fija)
        import explainer_v1; return explainer_v1.make_frame(tl, scenes, f)
    t = f / FPS; s = tl.at(t)
    if s is None: return scenes["_end"](tl, t - tl.talk_end).convert("RGB")
    si = tl.segs.index(s); img = frame_base(tl, si); rel = t - s.start
    a_in = 1.0 if si == 0 else ease(rel / 0.35)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); scenes[s.scene](lay, s, rel)
    part = lay.crop((0, OLD_TOP, W, OLD_TOP + (UP[3] - UP[1]) + 20))       # solo la zona del antiguo panel
    up = Image.new("RGBA", (W, H), (0, 0, 0, 0)); up.alpha_composite(part, (0, OLD_TOP - SHIFT))
    img.alpha_composite(faded(up, a_in) if a_in < 1 else up)
    draw_pill(img, s.raw.get("pildora"), 1.0 if si == 0 else ease((rel - 0.5) / 0.4), gold=s.raw.get("pildora_oro", False))
    low = Image.new("RGBA", (W, H), (0, 0, 0, 0)); draw_low(low, s, rel, scenes.get("_extras", {})); img.alpha_composite(faded(low, a_in) if a_in < 1 else low)
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
