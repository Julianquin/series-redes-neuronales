"""Episodio 03 · Redes recurrentes (RNN y LSTM).  Uso: python3 ep03_rnn.py info | prev f… | chunk 85 | encode"""
import os, sys, pickle, json, math
import numpy as np
from PIL import Image, ImageDraw
from explainer import *
from reel_common import T_
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_pueblos as TP
from reel_pueblos import sign, put

HERE = os.path.dirname(os.path.abspath(__file__))
tl = Timeline(os.path.join(HERE, "guion_ep03_rnn.yaml"))
CX = (PANEL[0] + PANEL[2]) / 2

# ---------- datos reales (Ep. 3b) ----------
RUN = pickle.load(open("work/run_pueblos.pkl", "rb")); SEL = json.load(open("data/pueblos_sel.json"))
L, CK = RUN["L"], RUN["CK"]; Ls = np.convolve(np.pad(L, 12, mode="edge"), np.ones(25) / 25, "valid")
EVO = SEL["evo"]; NAME = EVO[-1]; NAMES = [x["n"] for x in RUN["recs"]]
SHORT = min(NAMES, key=len); LONG = max(NAMES, key=len); MID = "Villanueva de la Serena" if "Villanueva de la Serena" in NAMES else sorted(NAMES, key=len)[len(NAMES) // 2]
def trace():
    """Repite la generación del nombre final (misma secuencia aleatoria que en select_pueblos.py) guardando h y p en cada paso."""
    rng = np.random.default_rng(5)
    for _ in range(7): u = rng.random(24)
    P = RUN["snaps"][CK[-1]]; ci = TP.COMM.index(SEL["comm1"]); h = np.zeros((1, TP.Hd)); c = np.zeros((1, TP.Hd))
    tok = np.array([TP.ix[TP.START]]); cm = np.array([ci]); out, H_, PR = "", [], []
    for s in range(24):
        h, c, p, _ = TP.cell(P, tok, cm, h, c); q = p[0] ** (1 / 0.8); q /= q.sum()
        k = min(int(np.searchsorted(np.cumsum(q), u[s])), TP.V - 1); H_.append(h[0].copy()); PR.append(p[0].copy())
        if TP.vocab[k] == TP.END: break
        out += TP.vocab[k]; tok = np.array([k])
    return out, np.array(H_), np.array(PR)
TNAME, HS, PRS = trace(); assert TNAME == NAME, (TNAME, NAME)

def label(d, x, y, txt, col=SOFT, size=26, anchor="mm", bold=False):
    d.text((x, y), txt, font=font(FS["monob" if bold else "mono"], size), fill=col, anchor=anchor)
def hl(s, t, name, base=WHITE, win=1.2):
    return (GOLD if 0 <= t - s.cue(name) < win else base, s.a(t, name, 0.35))
def box(d, x, y, w, h, txt, col=WHITE, fill=(26, 34, 44), a=1.0, size=30, bold=True, outline=None):
    d.rounded_rectangle((x - w / 2, y - h / 2, x + w / 2, y + h / 2), radius=12, fill=fill + (int(255 * a),), outline=(outline or (70, 86, 104)) + (int(255 * a),), width=3)
    if txt: d.text((x, y), txt, font=font(FS["monob" if bold else "mono"], size), fill=col + (int(255 * a),) if len(col) == 3 else col, anchor="mm")
def arrow(d, x0, y0, x1, y1, col, a=1.0, w=4):
    d.line((x0, y0, x1, y1), fill=col + (int(255 * a),), width=w); ang = math.atan2(y1 - y0, x1 - x0)
    d.polygon([(x1, y1), (x1 - 14 * math.cos(ang - .45), y1 - 14 * math.sin(ang - .45)), (x1 - 14 * math.cos(ang + .45), y1 - 14 * math.sin(ang + .45))], fill=col + (int(255 * a),))

# ---------- escenas ----------
def hook(lay, s, t):
    put(lay, sign(860, 240, NAME, edge=GOLD, glow=GOLD), CX, 700); d = ImageDraw.Draw(lay, "RGBA")
    label(d, CX, 880, f"no existe · se parece a {SEL['near_name']} ({SEL['near_prov']}), que sí", SOFT, 26)
    a = s.a(t, "inventado"); label(d, CX, 960, "inventado letra a letra por una red LSTM", GOLD + (int(255 * a),), 28, bold=True)
    for k, (nm, txt) in enumerate((("recuerda", "cómo recuerda"), ("formulas", "fórmulas"), ("codigo", "código"), ("olvida", "por qué olvida"))):
        a = s.a(t, nm); chip(d, CX - 345 + k * 230, 1124, txt.upper(), col=GOLD + (int(255 * a),), size=22, anchor="mm", bg=(40, 32, 14, int(255 * a)))

def letters(d, word, x0, y, size=60, gap=8, a=1.0, hi=None, col=WHITE):
    f = font(FS["monob"], int(size * 0.62)); x = x0
    for i, ch in enumerate(word):
        on = hi is not None and i in hi
        box(d, x + size / 2, y, size, size, "_" if ch == " " else ch, col=GOLD if on else col, a=a, size=int(size * .55), outline=GOLD if on else None)
        x += size + gap
    return x
def problem(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); w = "Villanueva de la"
    x_end = letters(d, w, 110, 600, size=48, gap=6, a=1.0)
    a_q = s.a(t, "siguiente"); pq = 1 + 0.08 * math.sin(t * 6)
    box(d, x_end + 24, 600, 48 * pq, 48 * pq, "?", col=GOLD, a=a_q, size=30, outline=GOLD)
    if s.a(t, "orden") > 0: label(d, CX, 680, "depende de todas las letras anteriores… y de su orden", SOFT + (int(255 * s.a(t, "orden")),), 26)
    a_c = s.a(t, "convolucional")
    if a_c > 0:
        k = int((t - s.cue("convolucional")) * 3) % (len(w) - 2); d.rounded_rectangle((110 + k * 54 - 6, 570, 110 + (k + 3) * 54, 630), radius=10, outline=BLUE + (int(255 * a_c),), width=5)
        label(d, 290, 760, "CNN: ventana fija de 3", BLUE + (int(255 * a_c),), 26, bold=True)
    a_p = s.a(t, "perceptron")
    if a_p > 0:
        for i in range(8): d.rectangle((560 + i * 50, 735, 600 + i * 50, 785), outline=WHITE + (int(255 * a_p),), width=3)
        label(d, 760, 810, "MLP: siempre 8 casillas", WHITE + (int(255 * a_p),), 24, bold=True)
    a_l = s.a(t, "longitud")
    if a_l > 0:
        for k, nm in enumerate((SHORT, MID, LONG)):
            aa = ease((t - s.cue("longitud") - 0.3 * k) / 0.3); y = 900 + k * 80; tw = 22 * len(nm)
            d.rounded_rectangle((110, y - 26, 110 + tw + 30, y + 26), radius=12, fill=(26, 34, 44, int(255 * aa)), outline=GOLD + (int(255 * aa),), width=2)
            d.text((125, y), nm, font=font(FS["monob"], 30), fill=WHITE + (int(255 * aa),), anchor="lm"); label(d, 110 + tw + 60, y, f"{len(nm)} letras", SOFT + (int(255 * aa),), 22, anchor="lm")

F_RNN = Formula([("h", r"$h_t$"), ("eq", "$=$"), ("tanh", r"$\tanh($"), ("wx", r"$W x_t$"), ("p1", "$+$"), ("uh", r"$U h_{t-1}$"), ("p2", "$+$"), ("b", r"$b)$")], px=60)
STEPS = "Chipe"
def unrolled(d, n, xs, a_cells=1.0, hot=None, label_wu=False, a_h=1.0, letters_vis=None, out=True, t=0.0):
    for i in range(n):
        x = xs[i]; av = 1.0 if letters_vis is None else letters_vis[i]
        box(d, x, 960, 72, 60, STEPS[i] if i < len(STEPS) else "", col=WHITE, a=av, size=34)
        arrow(d, x, 928, x, 862, MUTED, a=av * a_cells, w=3)
        box(d, x, 820, 120, 76, "RNN", col=GOLD if (hot is not None and i in hot) else WHITE, a=av * a_cells, size=26, outline=GOLD if (hot is not None and i in hot) else None)
        if label_wu: label(d, x, 878, "W, U", GOLD + (int(255 * av),), 18, bold=True)
        if out:
            arrow(d, x, 780, x, 718, MUTED, a=av * a_cells, w=3); nxt = (STEPS + "r")[i + 1]
            box(d, x, 690, 60, 50, nxt, col=SOFT, a=av * a_cells, size=28)
        if i < n - 1:
            arrow(d, x + 62, 820, xs[i + 1] - 62, 820, GOLD, a=av * a_h * a_cells, w=5); label(d, (x + xs[i + 1]) / 2, 795, "h", GOLD + (int(255 * av * a_h),), 22, bold=True)
def rnn(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); xs = [150 + i * 190 for i in range(5)]
    vis = [ease((t - s.cue("lee") - 0.45 * i) / 0.3) for i in range(5)] if t >= s.cue("lee") else [0] * 5
    hot = {int(((t - s.cue("pasa")) * 1.5) % 5)} if s.a(t, "pasa") > 0 else None
    unrolled(d, 5, xs, a_cells=1.0, hot=hot, label_wu=s.a(t, "mismos") > 0, a_h=s.a(t, "estado"), letters_vis=vis)
    if s.a(t, "estado") > 0: label(d, CX, 600, "estado oculto h: un resumen de lo leído hasta ahora", GOLD + (int(255 * s.a(t, "estado")),), 26, bold=True)
    F_RNN.draw(lay, CX, 1090, {"h": hl(s, t, "estado"), "eq": (WHITE, s.a(t, "combina")), "tanh": (WHITE, s.a(t, "combina")), "wx": hl(s, t, "combina"), "p1": (WHITE, s.a(t, "combina")),
                               "uh": hl(s, t, "combina", win=2.0), "p2": (WHITE, s.a(t, "combina")), "b": (WHITE, s.a(t, "combina"))}, max_w=900)

F_V = Formula([("l", r"$\frac{\partial h_t}{\partial h_{t-k}}$"), ("ap", r"$\approx$"), ("u", r"$U^{\,k}$"), ("ar", r"$\rightarrow\ 0$")], px=62)
def vanish(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); n = 7; xs = [120 + i * 140 for i in range(n)]
    for i in range(n):
        box(d, xs[i], 760, 96, 66, "RNN", col=WHITE, size=22)
        if i < n - 1: arrow(d, xs[i] + 50, 760, xs[i + 1] - 50, 760, (90, 104, 122), w=3)
    a_b = s.a(t, "atras")
    if a_b > 0:
        ph = ((t - s.cue("atras")) * 0.7) % 1; pos = xs[-1] - ph * (xs[-1] - xs[0])
        fade = s.a(t, "desvanece")
        for i in range(n):
            k = n - 1 - i; inten = 0.5 ** k if fade > 0 else 1.0; aa = ease((t - s.cue("multiplica") - 0.2 * k) / 0.3) if t > s.cue("multiplica") else 0
            col = RED if s.a(t, "pequenos") > 0 and inten < 0.2 else GOLD
            d.line((xs[i] - 40, 840, xs[i] + 40, 840), fill=col + (int(255 * max(0.08, inten) * a_b),), width=10)
            if aa > 0: label(d, xs[i], 880, dec(0.5 ** k, 2) if k < 5 else dec(0.5 ** k, 3), col + (int(255 * aa),), 22, bold=True)
        d.ellipse((pos - 12, 828, pos + 12, 852), fill=RED + (230,))
        label(d, CX, 650, "el error viaja hacia atrás · ×0,5 en cada paso", SOFT + (int(255 * a_b),), 26)
    F_V.draw(lay, CX, 1010, {"l": (WHITE, s.a(t, "multiplica")), "ap": (WHITE, s.a(t, "multiplica")), "u": hl(s, t, "multiplica", win=1.6), "ar": hl(s, t, "desvanece", win=1.6)})
    if s.a(t, "olvida") > 0: chip(d, CX, 1120, "olvida lo que leyó al principio", col=RED + (int(255 * s.a(t, "olvida")),), size=28, anchor="mm", bg=(40, 14, 18, int(240 * s.a(t, "olvida"))))

F_C = Formula([("c", r"$c_t$"), ("eq", "$=$"), ("f", r"$f_t\odot c_{t-1}$"), ("p", "$+$"), ("ig", r"$i_t\odot g_t$")], px=58)
F_H2 = Formula([("h", r"$h_t$"), ("eq", "$=$"), ("o", r"$o_t\odot\tanh(c_t)$")], px=58)
def lstm(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); a_c = s.a(t, "celda"); a_g = s.a(t, "puertas")
    y = 620; d.line((110, y, 960, y), fill=GOLD + (int(255 * max(0.25, a_c)),), width=8); label(d, 130, y - 32, "c (memoria)", GOLD + (int(255 * max(0.3, a_c)),), 24, anchor="lm", bold=True)
    gates = [("olvido", "f", 330, "×", "borra"), ("entrada", "i", 560, "+", "guarda"), ("salida", "o", 800, "×", "muestra")]
    for cue, g, x, op, verb in gates:
        hot = 0 <= t - s.cue(cue) < 1.6; a = max(a_g, s.a(t, cue)); col = GOLD if hot else WHITE
        if a < 0.02: continue                   # invisible: no dibujar (dibujar con alfa 0 «borraría» la línea de memoria)
        if g != "o":
            d.ellipse((x - 30, y - 30, x + 30, y + 30), fill=(26, 34, 44, int(255 * a)), outline=col + (int(255 * a),), width=4); label(d, x, y - 2, op, col + (int(255 * a),), 34, bold=True)
            arrow(d, x, 760, x, y + 34, col, a=a, w=4)
        else:
            d.line((x, y, x, 700), fill=WHITE + (int(255 * a),), width=4); box(d, x, 720, 90, 40, "tanh", col=WHITE, a=a, size=20); arrow(d, x + 50, 720, x + 120, 720, col, a=a, w=4)
            d.ellipse((x + 120, 690, x + 180, 750), fill=(26, 34, 44, int(255 * a)), outline=col + (int(255 * a),), width=4); label(d, x + 150, 718, "×", col + (int(255 * a),), 34, bold=True)
            arrow(d, x + 150, 760, x + 150, 754, col, a=a, w=4); label(d, x + 150, 650, "h", GOLD + (int(255 * a),), 26, bold=True)
        cx_ = x if g != "o" else x + 150
        d.ellipse((cx_ - 34, 770, cx_ + 34, 838), fill=(40, 32, 14, int(255 * a)) if hot else (26, 34, 44, int(255 * a)), outline=col + (int(255 * a),), width=4)
        label(d, cx_, 803, f"{g} σ", col + (int(255 * a),), 26, bold=True); label(d, cx_, 868, f"{cue}: {verb}", (GOLD if hot else SOFT) + (int(255 * max(a, s.a(t, cue))),), 20)
    if s.a(t, "sumando") > 0:
        pr = 36 + 8 * math.sin((t - s.cue("sumando")) * 6); d.ellipse((560 - pr, y - pr, 560 + pr, y + pr), outline=GOLD, width=4)
        label(d, CX, 930, "la memoria se actualiza sumando: el recuerdo no se multiplica hasta desaparecer", GOLD, 22, bold=True)
    F_C.draw(lay, CX, 1020, {"c": hl(s, t, "celda"), "eq": (WHITE, s.a(t, "celda")), "f": hl(s, t, "olvido"), "p": hl(s, t, "sumando", win=1.6), "ig": hl(s, t, "entrada")})
    F_H2.draw(lay, CX, 1110, {"h": hl(s, t, "salida"), "eq": (WHITE, s.a(t, "salida")), "o": hl(s, t, "salida")})

CODE = """def paso_lstm(x, h, c, W, U, b):
    z = x @ W + h @ U + b      # 4 puertas a la vez
    f, i, o, g = np.split(z, 4, axis=1)
    f, i, o = sigmoid(f), sigmoid(i), sigmoid(o)
    g = np.tanh(g)
    c = f * c + i * g          # memoria: se suma
    h = o * np.tanh(c)         # lo que se muestra
    return h, c

h = c = np.zeros((1, 128))
for letra in palabra:          # letra a letra
    x = one_hot(letra)
    h, c = paso_lstm(x, h, c, W, U, b)
    p = softmax(h @ Wy + by)   # siguiente letra"""
CB = CodeBlock(CODE, size=26, lh=1.4)
GROUPS = [("multiplicacion", range(1, 5)), ("celda", {5}), ("estado", {6, 7}), ("repite", range(9, 14))]
def code(lay, s, t):
    act, foc = None, 0.0
    for name, rng in GROUPS:
        if t >= s.cue(name) - 0.1: act, foc = set(rng), ease((t - s.cue(GROUPS[0][0])) / 0.4)
    CB.draw(lay, 96, 545, act, foc)
    label(ImageDraw.Draw(lay, "RGBA"), CX, 1150, "la red real además recibe la comunidad autónoma en cada paso", MUTED, 20)

CAPS = ["ruido puro", "empieza a separar palabras", "sílabas casi reales", "ya suena a pueblo", "un pueblo que no existe"]
def demo(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); t0 = s.cue("principio") - 0.3; t1 = s.dur - 0.4
    wts = np.array([1, 1, 1, 1, 2.2]); bnd = t0 + np.r_[0, np.cumsum(wts)] / wts.sum() * (t1 - t0)
    ci = int(np.clip(np.searchsorted(bnd, t, side="right") - 1, 0, 4)) if t >= t0 else 0; seg = (bnd[ci + 1] - bnd[ci]) if ci < 4 else (bnd[1] - bnd[0])
    lf = (t - bnd[ci]) if t >= t0 else 0
    full = EVO[ci]; nshow = int(np.clip((lf - 0.1) * len(full) / (seg * 0.55), 0, len(full))) if t >= t0 else 0
    chip(d, CX, 560, SEL["comm1"], col=WHITE, size=26, anchor="mm", bg=(30, 40, 54, 255))
    put(lay, sign(820, 190, full, full[:nshow], caret=nshow < len(full) or int(t * 4) % 2 == 0, edge=GOLD if ci == 4 else None, glow=GOLD if ci == 4 and nshow == len(full) else None), CX, 690)
    d = ImageDraw.Draw(lay, "RGBA"); label(d, CX, 832, f"{T_(CAPS[ci])} · iteración {fmt(CK[ci])}", GOLD if ci == 4 else WHITE, 26, bold=True)
    prev = CK[ci - 1] if ci else 1; upto = prev + (CK[ci] - prev) * ease(lf / (seg * 0.4)) if t >= t0 else 1
    xs = np.arange(1, len(Ls) + 1)
    chart(lay, (90, 880, 990, 1170), [dict(xs=xs[::5], ys=Ls[::5], upto=upto, col=GOLD, name="error")], "error al adivinar la siguiente letra",
          xmax=CK[-1], ylim=(1.2, 4.4), yticks=[1.5, 2.5, 3.5], logy=False, yfmt=lambda v: dec(v, 1))

def memory(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); n = len(NAME); cw = min(76, 820 // n); x0 = CX - n * cw / 2; y0 = 570; hh = 384
    a_c = s.a(t, "memoria"); k_vis = int(np.clip((t - s.cue("columna")) * 3 + 1, 0, n)) if t >= s.cue("columna") else 0
    hmax = np.abs(HS).max()
    if a_c > 0:
        for j in range(n):
            if j >= k_vis: break
            col = HS[j] / hmax; arr = np.zeros((128, 1, 3), np.uint8)
            arr[:, 0, 0] = np.where(col > 0, 244 * col, 0) + 14; arr[:, 0, 1] = np.where(col > 0, 190 * col, 168 * -col) + 20; arr[:, 0, 2] = np.where(col > 0, 78 * col, 255 * -col) + 30
            lay.paste(Image.fromarray(arr.clip(0, 255).astype(np.uint8)).resize((cw - 6, hh), Image.NEAREST), (int(x0 + j * cw), y0))
        d = ImageDraw.Draw(lay, "RGBA")
        for j in range(n):
            box(d, x0 + j * cw + (cw - 6) / 2, y0 + hh + 40, cw - 10, 54, NAME[j], col=GOLD if j == k_vis - 1 else WHITE, a=1.0 if j < k_vis else 0.25, size=30)
        label(d, x0 - 20, y0 + hh / 2, "128 neuronas", SOFT + (int(255 * s.a(t, "fila")),), 20, anchor="rm") if x0 > 200 else None
    if 0 < k_vis <= n:
        j = k_vis - 1; top = np.argsort(PRS[j])[::-1][:3]; prefix = NAME[:j]
        txt = "  ·  ".join(f"{'_' if TP.vocab[q] == ' ' else ('fin' if TP.vocab[q] == TP.END else TP.vocab[q])} {dec(PRS[j][q] * 100, 0)} %" for q in top)
        label(d, CX, 1080, f"tras «{prefix}», predice:" if prefix else "al empezar, predice:", SOFT, 24); label(d, CX, 1122, txt, GOLD, 30, bold=True)
    if s.a(t, "izquierda") > 0:
        a = s.a(t, "izquierda"); d.line((x0, y0 - 18, x0 + max(1, k_vis - 1) * cw, y0 - 18), fill=GOLD + (int(255 * a),), width=4)
        label(d, x0, y0 - 40, "todo esto influye en la letra siguiente", GOLD + (int(255 * a),), 20, anchor="lm", bold=True)

def when(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    rows = [("estandar", "✓", GREEN, "traducción, voz y series temporales"), ("paralelizarlas", "✗", RED, "secuencial: un paso tras otro"), ("largas", "✗", RED, "con secuencias muy largas, sigue olvidando")]
    for i, (cue, sym, col, txt) in enumerate(rows):
        a = s.a(t, cue); y = 580 + i * 80
        if a <= 0: continue
        d.ellipse((96, y - 24, 144, y + 24), fill=col + (int(60 * a),), outline=col + (int(255 * a),), width=3)
        d.text((120, y), sym, font=font(FS["monob"], 30), fill=col + (int(255 * a),), anchor="mm"); d.text((166, y), txt, font=font(FS["body"], 34), fill=WHITE + (int(255 * a),), anchor="lm")
    a = s.a(t, "orden")
    if a > 0:
        n = 10; prog = ((t - s.cue("orden")) * 2.0) % (n + 2)
        for i in range(n):
            x = 120 + i * 86; on = i < prog
            d.rounded_rectangle((x, 900, x + 70, 960), radius=10, fill=((40, 32, 14) if on else (26, 34, 44)) + (int(255 * a),), outline=(GOLD if on else (70, 86, 104)) + (int(255 * a),), width=3)
            if i < n - 1: arrow(d, x + 72, 930, x + 84, 930, MUTED, a=a, w=3)
        label(d, CX, 1000, "cada paso espera al anterior", SOFT + (int(255 * a),), 26)

def next_(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    d.text((CX, 590), "PRÓXIMO EPISODIO", font=font(FS["monob"], 30), fill=SOFT, anchor="mm")
    d.text((CX, 670), T_("Transformers"), font=font(FS["head"], 70), fill=GOLD, anchor="mm")
    words = ["el", "gato", "se", "sentó", "en", "la", "alfombra"]; f = font(FS["monob"], 30); xs = []; x = 120
    for w_ in words:
        tw = f.getlength(w_) + 28; xs.append(x + tw / 2); x += tw + 12
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            a = 0.25 + 0.75 * (0.5 + 0.5 * math.sin(t * 3 + i + j)); mid = (xs[i] + xs[j]) / 2; hgt = 40 + 22 * (j - i)
            pts = [(xs[i] + (xs[j] - xs[i]) * u, 940 - hgt * math.sin(math.pi * u)) for u in np.linspace(0, 1, 24)]
            d.line(pts, fill=GOLD + (int(150 * a),), width=2)
    x = 120
    for w_ in words:
        tw = f.getlength(w_) + 28; d.rounded_rectangle((x, 940, x + tw, 996), radius=12, fill=(26, 34, 44), outline=(70, 86, 104), width=2)
        d.text((x + tw / 2, 968), w_, font=f, fill=WHITE, anchor="mm"); x += tw + 12
    label(d, CX, 1060, "todas las palabras a la vez · atención entre todas", SOFT, 28)
    a = s.a(t, "bucle"); d.line((404, 360, 440, 360), fill=GOLD + (int(255 * a),), width=4); d.polygon([(440, 352), (440, 368), (454, 360)], fill=GOLD + (int(255 * a),))

def end(tl_, t):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    d.text((W / 2, 720), "REDES NEURONALES", font=font(FS["head"], 84), fill=GOLD, anchor="mm")
    d.text((W / 2, 810), "Ep. 3 · Redes recurrentes", font=font(FS["body"], 46), fill=WHITE, anchor="mm")
    d.rounded_rectangle((200, 900, 880, 1140), radius=24, fill=(16, 22, 32, 255), outline=(44, 58, 74, 255), width=2)
    d.text((W / 2, 960), "PRÓXIMO: TRANSFORMERS", font=font(FS["monob"], 28), fill=GOLD, anchor="mm")
    d.text((W / 2, 1020), "código completo en la descripción", font=font(FS["body"], 30), fill=SOFT, anchor="mm")
    d.text((W / 2, 1090), HANDLE, font=font(FS["head"], 52), fill=WHITE, anchor="mm")
    return img

SCENES = dict(hook=hook, problem=problem, rnn=rnn, vanish=vanish, lstm=lstm, code=code, demo=demo, memory=memory, when=when, next=next_, _end=end)
if __name__ == "__main__":
    cli(tl, SCENES, "ep03_rnn_piloto")
