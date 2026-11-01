"""Episodio 04 · Transformers y atención (la red que pone tildes).  Uso: python3 ep04_transformer.py info | prev f… | chunk 85 | encode"""
import os, sys, pickle, math
import numpy as np
from PIL import Image, ImageDraw
from explainer import *
from reel_common import T_
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_tildes as TT

HERE = os.path.dirname(os.path.abspath(__file__))
tl = Timeline(os.path.join(HERE, "guion_ep04_transformer.yaml"))
CX = (PANEL[0] + PANEL[2]) / 2

# ---------- red real ----------
R = pickle.load(open("work/run_tildes.pkl", "rb")); SN = R["snaps"]; KEYS = sorted(SN); PF = SN[KEYS[-1]]
LH = R["L"]; ACC = R["acc"]; D = TT.D; KIX = {k: i for i, k in enumerate(D["keys"])}
def run(word, P=PF):
    i = KIX[word]; p, As = TT.forward(P, TT.X[[i]], TT.POSE[[i]], TT.M[[i]]); n = len(word)
    return p[0, :n, 1], [A[0, :n, :n] for A in As]
def accent(word, pt, thr=0.5):
    return "".join("áéíóú"["aeiou".index(c)] if (pt[i] > thr and c in "aeiou") else c for i, c in enumerate(word))
HAS = TT.Y[TT.TE].any(1); OK_FINAL = None
TE = TT.TE; _, predF = TT.evaluate(PF, TE); okF = ((predF == TT.Y[TE]) | ~TT.M[TE]).all(1)
ACC_ALL, ACC_TILDE, BASE = okF.mean(), okF[HAS].mean(), (~HAS).mean()
# palabras de examen para la demo: con tilde y acertadas al final (más una sin tilde)
cand = [D["keys"][i] for r, i in enumerate(TE) if HAS[r] and okF[r] and 6 <= len(D["keys"][i]) <= 10]
DEMO = (["pelicula"] if "pelicula" in cand else []) + [w for w in cand if w != "pelicula"][3:8]
DEMO += [D["keys"][i] for r, i in enumerate(TE) if (not HAS[r]) and okF[r] and 6 <= len(D["keys"][i]) <= 9][:1]
# atención media desde la vocal con tilde, por distancia al final (capa 1)
acc_d, cnt = np.zeros(16), np.zeros(16)
for r in TE[:3000]:
    if not TT.Y[r].any(): continue
    n = TT.M[r].sum(); q = int(TT.Y[r].argmax()); _, As = TT.forward(PF, TT.X[[r]], TT.POSE[[r]], TT.M[[r]])
    for j in range(n): acc_d[n - 1 - j] += As[0][0, q, j]; cnt[n - 1 - j] += 1
AVG_D = acc_d[:8] / cnt[:8]; LAST3 = AVG_D[:3].sum()
UNIF = float(np.mean([1 / TT.M[r].sum() for r in TE[:3000] if TT.Y[r].any()]))

def label(d, x, y, txt, col=SOFT, size=26, anchor="mm", bold=False):
    d.text((x, y), txt, font=font(FS["monob" if bold else "mono"], size), fill=col, anchor=anchor)
def hl(s, t, name, base=WHITE, win=1.2): return (GOLD if 0 <= t - s.cue(name) < win else base, s.a(t, name, 0.35))
def word_boxes(d, word, cx, y, size=78, gap=8, hot=None, cols=None, a=1.0):
    n = len(word); x0 = cx - (n * size + (n - 1) * gap) / 2; xs = []
    for i, ch in enumerate(word):
        x = x0 + i * (size + gap); xs.append(x + size / 2); on = hot is not None and i in hot; col = (cols or {}).get(i, GOLD if on else WHITE)
        d.rounded_rectangle((x, y - size / 2, x + size, y + size / 2), radius=14, fill=((40, 32, 14) if on else (26, 34, 44)) + (int(255 * a),), outline=(GOLD if on else (70, 86, 104)) + (int(255 * a),), width=3)
        d.text((x + size / 2, y - 2), ch, font=font(FS["monob"], int(size * .56)), fill=col + (int(255 * a),), anchor="mm")
    return xs
def arcs(d, xs, y, q, w, a=1.0, col=GOLD, up=True, scale=1.0):
    for j, wt in enumerate(w):
        if j == q or wt < 0.02: continue
        x0, x1 = xs[q], xs[j]; hgt = (30 + 0.28 * abs(x1 - x0)) * scale
        pts = [(x0 + (x1 - x0) * u, y - (hgt * math.sin(math.pi * u)) * (1 if up else -1)) for u in np.linspace(0, 1, 28)]
        d.line(pts, fill=col + (int(255 * a * min(1, 0.25 + wt * 1.4)),), width=max(2, int(2 + 16 * wt)), joint="curve")

# ---------- escenas ----------
HW = "murcielago"; HP, HA = run(HW)
def hook(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); a_s = s.a(t, "sabe"); q = int(HP.argmax())
    word = accent(HW, HP) if a_s > 0.5 else HW
    xs = word_boxes(d, word, CX, 820, size=80, gap=8, hot={q} if a_s > 0 else None)
    if a_s > 0: chip(d, xs[q], 720, f"tilde · {dec(HP[q] * 100, 0)} %", col=GOLD + (int(255 * a_s),), size=26, anchor="mm", bg=(40, 32, 14, int(255 * a_s)))
    a_m = s.a(t, "mira")
    if a_m > 0: arcs(d, xs, 870, q, HA[1][q], a=a_m, up=False); label(d, CX, 1010, "dónde mira la «e» antes de decidir", GOLD + (int(255 * a_m),), 26, bold=True)
    label(d, CX, 600, "la red recibe la palabra sin tildes", SOFT, 26)
    for k, (nm, txt) in enumerate((("atencion", "la atención"), ("formulas", "fórmulas"), ("codigo", "código"), ("ninguna", "lo imposible"))):
        a = s.a(t, nm); chip(d, CX - 345 + k * 230, 1124, txt.upper(), col=GOLD + (int(255 * a),), size=22, anchor="mm", bg=(40, 32, 14, int(255 * a)))

CP, CA = run("camion")
def problem(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); a_c = s.a(t, "camion"); word = "camión" if a_c > 0.5 else "camion"; q = 4
    xs = word_boxes(d, word, CX, 640, size=90, hot={q} if s.a(t, "demas") > 0 else None, cols={5: GOLD} if s.a(t, "ene") > 0 else None)
    if s.a(t, "demas") > 0 and s.a(t, "final") < 1: arcs(d, xs, 590, q, [1 / 6] * 6, a=s.a(t, "demas") * (1 - s.a(t, "final")), col=WHITE)
    if s.a(t, "final") > 0: arcs(d, xs, 590, q, [0, 0, 0, 0.1, 0, 0.8], a=s.a(t, "final"))
    if s.a(t, "ene") > 0: label(d, CX, 720, "termina en n → aguda con tilde", GOLD + (int(255 * s.a(t, "ene")),), 28, bold=True)
    a_r = s.a(t, "recurrente")
    if a_r > 0:
        label(d, 260, 820, "recurrente: paso a paso", SOFT + (int(255 * a_r),), 24, bold=True)
        for i, ch in enumerate("camion"):
            x = 110 + i * 70; d.rounded_rectangle((x, 850, x + 56, 900), radius=10, fill=(26, 34, 44, int(255 * a_r)), outline=(70, 86, 104, int(255 * a_r)), width=2)
            d.text((x + 28, 875), ch, font=font(FS["monob"], 28), fill=WHITE + (int(255 * a_r),), anchor="mm")
        ph = ((t - s.cue("recurrente")) * 0.6) % 1; px = 138 + ph * 350; d.ellipse((px - 9, 916, px + 9, 934), fill=BLUE + (int(255 * a_r),))
    a_t = s.a(t, "directamente")
    if a_t > 0:
        label(d, 790, 820, "transformer: directo", GOLD + (int(255 * a_t),), 24, bold=True)
        for i, ch in enumerate("camion"):
            x = 600 + i * 66; d.rounded_rectangle((x, 850, x + 54, 900), radius=10, fill=(26, 34, 44, int(255 * a_t)), outline=((GOLD if i in (4, 5) else (70, 86, 104))) + (int(255 * a_t),), width=2)
            d.text((x + 27, 875), ch, font=font(FS["monob"], 28), fill=WHITE + (int(255 * a_t),), anchor="mm")
        arcs(d, [627 + i * 66 for i in range(6)], 905, 4, [0, 0, 0, 0, 0, 0.9], a=a_t, up=False, scale=1.2)

def attention(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); w = "camion"; A = CA[0]; q = 4; xs = word_boxes(d, w, CX, 600, size=84)
    a_q = s.a(t, "pregunta"); a_k = s.a(t, "etiqueta")
    for i, x in enumerate(xs):
        if a_q > 0: chip(d, x, 530, "Q", col=GOLD + (int(255 * a_q),), size=22, anchor="mm", bg=(40, 32, 14, int(255 * a_q)))
        if a_k > 0: chip(d, x, 672, "K", col=BLUE + (int(255 * a_k),), size=22, anchor="mm", bg=(18, 30, 52, int(255 * a_k)))
    a_t = s.a(t, "todas")
    if a_t > 0:                               # matriz real de atención (capa 1)
        cs = 52; x0, y0 = CX - 3 * cs + 40, 730
        for i in range(6):
            label(d, x0 - 26, y0 + i * cs + cs / 2, w[i], (GOLD if (i == q and s.a(t, "encajan") > 0) else SOFT) + (int(255 * a_t),), 26, bold=True)
            label(d, x0 + i * cs + cs / 2, y0 - 20, w[i], SOFT + (int(255 * a_t),), 22)
            for j in range(6):
                v = A[i, j] ** 0.7; col = tuple(int(18 + (c - 18) * v) for c in GOLD)
                d.rectangle((x0 + j * cs + 2, y0 + i * cs + 2, x0 + (j + 1) * cs - 2, y0 + (i + 1) * cs - 2), fill=col + (int(255 * a_t),))
        if s.a(t, "encajan") > 0:
            d.rectangle((x0, y0 + q * cs, x0 + 6 * cs, y0 + (q + 1) * cs), outline=GOLD, width=4)
            label(d, x0 + 6 * cs + 20, y0 + q * cs + cs / 2, f"la «o» mira a la «n»: {dec(A[q, 5] * 100, 0)} %", GOLD, 22, anchor="lm", bold=True)
        label(d, CX, y0 + 6 * cs + 26, "pesos reales de la red · capa 1", MUTED + (int(255 * a_t),), 20)
    if s.a(t, "recoge") > 0: arcs(d, xs, 650, q, A[q], a=s.a(t, "recoge"), up=False, scale=0.5)

F_AT = Formula([("at", r"$\mathrm{Atención}(Q,K,V)$"), ("eq", "$=$"), ("sm", r"$\mathrm{softmax}($"), ("qk", r"$QK^{\top}$"), ("rd", r"$/\sqrt{d}$"), ("cl", r"$)$"), ("v", r"$V$")], px=56)
F_QKV = Formula([("q", r"$Q = XW_Q$"), ("c1", r"$,$"), ("k", r"$K = XW_K$"), ("c2", r"$,$"), ("vv", r"$V = XW_V$")], px=52)
def formulas(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    F_QKV.draw(lay, CX, 590, {k: hl(s, t, "vectores", win=1.8) for k in ("q", "c1", "k", "c2", "vv")})
    st = {"at": (WHITE, s.a(t, "producto")), "eq": (WHITE, s.a(t, "producto")), "sm": hl(s, t, "softmax"), "qk": hl(s, t, "producto", win=1.6),
          "rd": hl(s, t, "raiz"), "cl": hl(s, t, "softmax"), "v": hl(s, t, "mezclan", win=2.0)}
    F_AT.draw(lay, CX, 700, st, max_w=920)
    i = KIX["camion"]; h = PF["E"][TT.X[[i]]] + PF["Ps"][None] + PF["Pe"][TT.POSE[[i]]]; Q, K = h @ PF["Wq0"], h @ PF["Wk0"]
    S = (Q @ K.transpose(0, 2, 1))[0, :6, :6] / math.sqrt(TT.d); A = np.exp(S - S.max(1, keepdims=True)); A /= A.sum(1, keepdims=True)
    for k, (mat, ttl, cue) in enumerate(((S, "QKᵀ/√d  (parecidos)", "raiz"), (A, "softmax  (pesos)", "softmax"))):
        a = s.a(t, cue)
        if a <= 0: continue
        cs = 46; x0 = 120 + k * 470; y0 = 820; m = (mat - mat.min()) / (mat.max() - mat.min() + 1e-9)
        for r in range(6):
            for c in range(6):
                v = m[r, c] ** 0.8; col = tuple(int(18 + (cc - 18) * v) for cc in (GOLD if k else BLUE))
                d.rectangle((x0 + c * cs + 2, y0 + r * cs + 2, x0 + (c + 1) * cs - 2, y0 + (r + 1) * cs - 2), fill=col + (int(255 * a),))
        label(d, x0 + 3 * cs, y0 - 22, ttl, SOFT + (int(255 * a),), 22, bold=True)
        if k: label(d, x0 + 6 * cs + 30, y0 + 3 * cs, "cada fila suma 1", MUTED + (int(255 * a),), 20, anchor="lm")
    if s.a(t, "softmax") > 0: d.text((560, 820 + 3 * 46), "→", font=font(FS["head"], 46), fill=GOLD + (int(255 * s.a(t, "softmax")),), anchor="mm")

def block(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); w = "camion"
    rows = [("letra", list(w), WHITE, None), ("posición", [str(i) for i in range(6)], BLUE, "posicion"), ("distancia al final", [str(5 - i) for i in range(6)], GOLD, "distancia")]
    for r, (lab, vals, col, cue) in enumerate(rows):
        a = 1.0 if cue is None else s.a(t, cue); y = 560 + r * 70
        if a <= 0: continue
        label(d, 110, y, lab, col + (int(255 * a),), 24, anchor="lm", bold=True)
        for i, v in enumerate(vals):
            x = 400 + i * 66; d.rounded_rectangle((x, y - 26, x + 56, y + 26), radius=10, fill=(26, 34, 44, int(255 * a)), outline=col + (int(255 * a),), width=2)
            d.text((x + 28, y), v, font=font(FS["monob"], 26), fill=col + (int(255 * a),), anchor="mm")
        if r: d.text((380, y - 35), "+", font=font(FS["head"], 30), fill=SOFT + (int(255 * a),), anchor="mm")
    if s.a(t, "orden") > 0 and s.a(t, "posicion") < 1:
        label(d, 790, 560, "sin posición, «camion» y «nocami» serían iguales", RED + (int(255 * s.a(t, "orden") * (1 - s.a(t, "posicion"))),), 20, anchor="mm")
    a_d = s.a(t, "densa"); a_r = s.a(t, "atajos"); bx = CX
    if a_d > 0:
        for k, (lab, y) in enumerate((("atención", 840), ("red densa", 990))):
            d.rounded_rectangle((bx - 170, y - 40, bx + 170, y + 40), radius=16, fill=(26, 34, 44, int(255 * a_d)), outline=(GOLD if k == 0 else BLUE) + (int(255 * a_d),), width=4)
            label(d, bx, y, lab, WHITE + (int(255 * a_d),), 30, bold=True)
            d.ellipse((bx - 22, y + 50, bx + 22, y + 94), fill=(26, 34, 44, int(255 * a_d)), outline=WHITE + (int(255 * a_d),), width=3); label(d, bx, y + 71, "+", WHITE + (int(255 * a_d),), 30, bold=True)
            d.line((bx, y - 72, bx, y - 42), fill=WHITE + (int(255 * a_d),), width=3); d.line((bx, y + 42, bx, y + 50), fill=WHITE + (int(255 * a_d),), width=3)
            if a_r > 0:
                hot = 0.5 + 0.5 * math.sin((t - s.cue("atajos")) * 5)
                d.line((bx - 170, y - 60, bx - 230, y - 60, bx - 230, y + 72, bx - 24, y + 72), fill=GOLD + (int(255 * a_r * (0.5 + 0.5 * hot)),), width=5)
        if a_r > 0: label(d, bx + 320, 905, "atajo: suma la entrada", GOLD + (int(255 * a_r),), 20, bold=True); label(d, bx + 320, 932, "a la salida", GOLD + (int(255 * a_r),), 20, bold=True)
        label(d, bx + 290, 1130, "× 2 bloques", MUTED + (int(255 * a_d),), 22, bold=True)

CODE = """def atencion(X, Wq, Wk, Wv):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    S = Q @ K.T / np.sqrt(K.shape[1])  # todas con todas
    A = softmax(S, axis=-1)            # pesos de atención
    return A @ V, A                    # mezcla de valores

def bloque(X, P):
    M, A = atencion(X, P.Wq, P.Wk, P.Wv)
    X = X + M @ P.Wo                   # atajo
    X = X + relu(X @ P.W1 + P.b1) @ P.W2 + P.b2
    return X, A

X = E[letras] + Ppos[pos] + Pfin[dist_final]
for P in bloques:                      # apilamos 2
    X, A = bloque(X, P)
p_tilde = softmax(X @ Wc + bc)         # ¿tilde en cada letra?"""
CB = CodeBlock(CODE, size=25, lh=1.4)
GROUPS = [("multiplicaciones", {1}), ("compara", {2, 3}), ("mezcla", {4}), ("bloque", range(6, 11)), ("apilamos", range(12, 16))]
def code(lay, s, t):
    act, foc = None, 0.0
    for name, rng in GROUPS:
        if t >= s.cue(name) - 0.1: act, foc = set(rng), ease((t - s.cue(GROUPS[0][0])) / 0.4)
    CB.draw(lay, 92, 548, act, foc)

ITS = [it for it, _, _ in LH]
def demo(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); t0, t1 = s.cue("entrenamos"), s.cue("noventa")
    u = np.clip((t - t0) / max(0.5, t1 - t0), 0, 1) if t >= t0 else 0; k = KEYS[int(round(u ** 0.8 * (len(KEYS) - 1)))]
    label(d, 110, 548, "palabras que nunca vio", SOFT, 22, anchor="lm"); label(d, 970, 548, f"iteración {fmt(k)}", SOFT, 22, anchor="rm")
    for r, w in enumerate(DEMO[:6]):
        pt, _ = run(w, SN[k]); guess = accent(w, pt); ok = guess == D["targets"][KIX[w]]; y = 590 + r * 52; c_ = r % 2; yy = 590 + (r // 2) * 62; x = 120 + c_ * 440
        d.text((x, yy), w, font=font(FS["mono"], 26), fill=MUTED, anchor="lm"); d.text((x + 175, yy), "→", font=font(FS["mono"], 26), fill=MUTED, anchor="lm")
        d.text((x + 215, yy), guess, font=font(FS["monob"], 32), fill=GREEN if ok else RED, anchor="lm")
    if s.a(t, "noventa") > 0:
        a = s.a(t, "noventa"); d.text((280, 820), f"{dec(ACC_ALL * 100, 0)} %", font=font(FS["head"], 64), fill=GOLD + (int(255 * a),), anchor="mm"); label(d, 280, 870, "acierto total", SOFT + (int(255 * a),), 22)
    if s.a(t, "ochenta") > 0:
        a = s.a(t, "ochenta"); d.text((540, 820), f"{dec(BASE * 100, 0)} %", font=font(FS["head"], 64), fill=MUTED + (int(255 * a),), anchor="mm"); label(d, 540, 870, "sin poner ninguna", SOFT + (int(255 * a),), 22)
    if s.a(t, "dos") > 0:
        a = s.a(t, "dos"); d.text((800, 820), f"{dec(ACC_TILDE * 100, 0)} %", font=font(FS["head"], 64), fill=WHITE + (int(255 * a),), anchor="mm"); label(d, 800, 870, "las que llevan tilde", SOFT + (int(255 * a),), 22)
    xs = np.array(ITS); up = k
    chart(lay, (90, 905, 990, 1170), [dict(xs=xs, ys=LH[:, 1], upto=up, col=BLUE, name="aprende", val_fmt=lambda v: dec(v, 3)),
                                      dict(xs=xs, ys=LH[:, 2], upto=up, col=GOLD, name="examen", val_fmt=lambda v: dec(v, 3))],
          "error por letra", xmax=R["ITERS"], ylim=(0.012, 0.8), yticks=[0.02, 0.1, 0.5], yfmt=lambda v: dec(v, 2))

def where(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); w = "camion"; q = 4; a_c = s.a(t, "camion")
    xs = word_boxes(d, "camión", CX, 590, size=84, hot={q}, cols={5: GOLD} if s.a(t, "ene") > 0 else None, a=max(a_c, 0.3))
    if a_c > 0: arcs(d, xs, 640, q, CA[0][q], a=a_c, up=False, scale=0.9)
    if s.a(t, "ene") > 0: chip(d, xs[5], 520, f"{dec(CA[0][q, 5] * 100, 0)} %", col=GOLD, size=26, anchor="mm", bg=(40, 32, 14, 255))
    a_p = s.a(t, "primera")
    if a_p > 0:
        x0, y0, bw, bh = 150, 1100, 90, 300; label(d, CX, 790, "a qué distancia del final mira la vocal con tilde (media, capa 1)", SOFT + (int(255 * a_p),), 21)
        top = AVG_D.max()
        for k_, v in enumerate(AVG_D):
            x = x0 + (7 - k_) * (bw + 12); hgt = bh * v / top * 0.95; hot = k_ < 3 and s.a(t, "tres") > 0
            d.rectangle((x, y0 - hgt * a_p, x + bw, y0), fill=(GOLD if hot else BLUE) + (int(255 * a_p),))
            label(d, x + bw / 2, y0 + 22, "última" if k_ == 0 else f"−{k_}", MUTED + (int(255 * a_p),), 18)
            label(d, x + bw / 2, y0 - hgt * a_p - 16, f"{dec(v * 100, 0)}", SOFT + (int(255 * a_p),), 18)
        yu = y0 - bh * UNIF / top * 0.95
        for xx in range(int(x0), int(x0 + 8 * (bw + 12)), 18): d.line((xx, yu, xx + 9, yu), fill=WHITE + (int(160 * a_p),), width=2)
        label(d, x0 + 4, yu - 14, "si mirara al azar", WHITE + (int(200 * a_p),), 18, anchor="lm")
        if s.a(t, "tres") > 0: chip(d, 340, 860, f"{dec(LAST3 * 100, 0)} % en las 3 últimas", col=GOLD, size=26, anchor="mm", bg=(40, 32, 14, 255))
    if s.a(t, "sola") > 0: chip(d, 340, 925, "nadie le dio la regla", col=WHITE, size=24, anchor="mm", bg=(30, 40, 54, 255))

def limits(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    a_t = s.a(t, "termino")
    if a_t > 0:
        pt, _ = run("termino"); label(d, CX, 560, f"«termino» → la red dice: {accent('termino', pt)}", SOFT + (int(255 * a_t),), 24)
        for k, (wd, ex) in enumerate((("término", "el término exacto"), ("termino", "yo termino ya"), ("terminó", "ayer terminó"))):
            a = ease((t - s.cue("termino") - 0.25 * k) / 0.3); x = 220 + k * 320
            d.rounded_rectangle((x - 140, 600, x + 140, 700), radius=16, fill=(26, 34, 44, int(255 * a)), outline=GOLD + (int(255 * a),), width=3)
            d.text((x, 635), wd, font=font(FS["monob"], 36), fill=WHITE + (int(255 * a),), anchor="mm"); label(d, x, 678, ex, SOFT + (int(255 * a),), 20)
    a_e = s.a(t, "esta")
    if a_e > 0:
        for k, (wd, ex) in enumerate((("esta", "esta casa"), ("está", "está en casa"))):
            x = 380 + k * 320
            d.rounded_rectangle((x - 140, 740, x + 140, 840), radius=16, fill=(26, 34, 44, int(255 * a_e)), outline=BLUE + (int(255 * a_e),), width=3)
            d.text((x, 775), wd, font=font(FS["monob"], 36), fill=WHITE + (int(255 * a_e),), anchor="mm"); label(d, x, 818, ex, SOFT + (int(255 * a_e),), 20)
    a_f = s.a(t, "frase")
    if a_f > 0:
        label(d, CX, 910, "la pista está en las palabras de alrededor", GOLD + (int(255 * a_f),), 28, bold=True)
        label(d, CX, 960, "(y en «teléfono», nada en las letras dice dónde cae el acento)", MUTED + (int(255 * a_f),), 21)
    if s.a(t, "modelos") > 0: chip(d, CX, 1060, "los modelos de lenguaje miran frases enteras", col=WHITE, size=28, anchor="mm", bg=(30, 40, 54, 255))

def next_(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); a = s.a(t, "cuadrado")
    n = 4 + int(12 * ease((t - s.cue("cuadrado")) / 2.0)) if t > s.cue("cuadrado") else 4; cs = min(26, 360 // n); x0 = 160; y0 = 560
    for i in range(n):
        for j in range(n): d.rectangle((x0 + j * cs, y0 + i * cs, x0 + (j + 1) * cs - 2, y0 + (i + 1) * cs - 2), fill=(GOLD if i == j else (60, 74, 88)) + (255,))
    label(d, x0 + n * cs / 2, y0 + n * cs + 30, f"{n} letras → {n * n} comparaciones", SOFT, 24, bold=True)
    d.text((760, 650), "PRÓXIMO", font=font(FS["monob"], 28), fill=SOFT, anchor="mm")
    d.text((760, 720), T_("Autoencoders"), font=font(FS["head"], 56), fill=GOLD, anchor="mm")
    label(d, 760, 790, "aprender a comprimir", SOFT, 26)
    a2 = s.a(t, "autoencoders"); d.line((522, 360, 558, 360), fill=GOLD + (int(255 * a2),), width=4); d.polygon([(558, 352), (558, 368), (572, 360)], fill=GOLD + (int(255 * a2),))

def end(tl_, t):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    d.text((W / 2, 720), "REDES NEURONALES", font=font(FS["head"], 84), fill=GOLD, anchor="mm")
    d.text((W / 2, 810), "Ep. 4 · Transformers y atención", font=font(FS["body"], 46), fill=WHITE, anchor="mm")
    d.rounded_rectangle((200, 900, 880, 1140), radius=24, fill=(16, 22, 32, 255), outline=(44, 58, 74, 255), width=2)
    d.text((W / 2, 960), "PRÓXIMO: AUTOENCODERS", font=font(FS["monob"], 28), fill=GOLD, anchor="mm")
    d.text((W / 2, 1020), "código completo en la descripción", font=font(FS["body"], 30), fill=SOFT, anchor="mm")
    d.text((W / 2, 1090), HANDLE, font=font(FS["head"], 52), fill=WHITE, anchor="mm")
    return img

SCENES = dict(hook=hook, problem=problem, attention=attention, formulas=formulas, block=block, code=code, demo=demo, where=where, limits=limits, next=next_, _end=end)
if __name__ == "__main__":
    cli(tl, SCENES, "ep04_transformer_piloto")
