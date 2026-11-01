"""Episodio 07 · Redes de grafos.  Uso: python3 ep07_gnn.py info | prev f… | chunk 85 | encode"""
import os, sys, math, collections
import numpy as np
from PIL import Image, ImageDraw
from explainer import *
from reel_common import T_
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reel_gnn as RN                      # piezas del Reel 7: mapa, colores por comunidad, datos de la GNN

HERE = os.path.dirname(os.path.abspath(__file__))
tl = Timeline(os.path.join(HERE, "guion_ep07_gnn.yaml"))
CX = (PANEL[0] + PANEL[2]) / 2
R, NAMES, PROV, COMM, Y, TEset, KEYS, N = RN.R, RN.NAMES, RN.PROV, RN.COMM, RN.Y, RN.TEset, RN.KEYS, RN.N
CCOL, FINAL, ACC = RN.CCOL, RN.FINAL, RN.ACC
NB = collections.defaultdict(list)
for a_, b_ in RN.EDGES: NB[a_].append(b_); NB[b_].append(a_)
LAT, LON = R["lat"], R["lon"]
EX = NAMES.index("Nabarniz"); TRE = NAMES.index("Condado de Treviño")
MLPF, GNNF = FINAL["mlp"], FINAL["gnn"]
col_of = lambda c: CCOL[COMM[c]]

def label(d, x, y, txt, col=SOFT, size=26, anchor="mm", bold=False):
    d.text((x, y), txt, font=font(FS["monob" if bold else "mono"], size), fill=col, anchor=anchor)
def hl(s, t, name, base=WHITE, win=1.2):
    return (GOLD if 0 <= t - s.cue(name) < win else base, s.a(t, name, 0.35))
def comm_cols():
    return [CCOL[COMM[Y[i] if not TEset[i] else GNNF[i]]] for i in range(N)]

# ---------- escenas ----------
def hook(lay, s, t):
    RN.draw_map(lay, (160, 540, 760), comm_cols(), r=2.4); d = ImageDraw.Draw(lay, "RGBA")
    items = [("grafo", "QUÉ ES UN GRAFO"), ("mensajes", "MENSAJES"), ("formulas", "FÓRMULAS"), ("codigo", "CÓDIGO"), ("trevino", "TREVIÑO")]
    f = font(FS["monob"], 19); ws = [f.getlength(x) + 28 for _, x in items]; x = CX - (sum(ws) + 12 * 4) / 2
    for (cue, txt), w in zip(items, ws):
        a = s.a(t, cue)
        if a > 0: chip(d, x + w / 2, 1140, txt, col=GOLD + (int(255 * a),), size=19, anchor="mm", bg=(40, 32, 14, int(255 * a)))
        x += w + 12

def problem(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    a = s.a(t, "tablas")
    for i in range(4):
        for j in range(3): d.rectangle((110 + j * 54, 580 + i * 34, 160 + j * 54, 610 + i * 34), outline=SOFT + (int(255 * a),), width=2)
    label(d, 190, 735, "tablas", SOFT + (int(255 * a),), 22)
    a = s.a(t, "cuadriculas"); rng = np.random.default_rng(1)
    for i in range(6):
        for j in range(6): v = int(60 + 150 * rng.random()); d.rectangle((420 + j * 26, 572 + i * 26, 444 + j * 26, 596 + i * 26), fill=(v, v, v, int(255 * a)))
    label(d, 498, 735, "píxeles", SOFT + (int(255 * a),), 22)
    a = s.a(t, "secuencias")
    for j, ch in enumerate("hola"): d.rounded_rectangle((720 + j * 62, 610, 772 + j * 62, 662), radius=10, outline=SOFT + (int(255 * a),), width=2); label(d, 746 + j * 62, 636, ch, WHITE + (int(255 * a),), 26, bold=True)
    label(d, 840, 735, "secuencias", SOFT + (int(255 * a),), 22)
    a = s.a(t, "conexiones")
    if a > 0:
        rng = np.random.default_rng(5); P_ = rng.uniform([180, 820], [900, 1100], (13, 2))
        E_ = [(i, j) for i in range(13) for j in range(i + 1, 13) if np.hypot(*(P_[i] - P_[j])) < 230]
        deg = collections.Counter([i for e in E_ for i in e])
        for i, j in E_: d.line((tuple(P_[i]), tuple(P_[j])), fill=(110, 124, 140, int(255 * a)), width=3)
        for k, (x, y) in enumerate(P_):
            d.ellipse((x - 20, y - 20, x + 20, y + 20), fill=(26, 34, 44, int(255 * a)), outline=GOLD + (int(255 * a),), width=3)
            if s.a(t, "vecinos") > 0: label(d, x, y, str(deg[k]), GOLD + (int(255 * s.a(t, "vecinos")),), 20, bold=True)
        label(d, CX, 790, "conexiones: sin orden ni tamaño fijo", GOLD + (int(255 * a),), 26, bold=True)

def subgraph(center, hops=2, cap=40):
    nodes = [center]; seen = {center}; frontier = [center]
    for _ in range(hops):
        nxt = []
        for u in frontier:
            for v in NB[u]:
                if v not in seen and len(nodes) < cap: seen.add(v); nodes.append(v); nxt.append(v)
        frontier = nxt
    return nodes
SG = subgraph(EX)
def graph(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); lon = LON[SG]; lat = LAT[SG]; cs = np.cos(np.radians(lat.mean()))
    gx = (lon - lon.min()) * cs; gy = lat.max() - lat; sc = min(760 / (gx.max() + 1e-9), 500 / (gy.max() + 1e-9))
    geo = np.c_[160 + gx * sc, 560 + gy * sc]; rnd = np.random.default_rng(4).uniform([160, 560], [920, 1060], (len(SG), 2))
    m = s.a(t, "mapa", 1.0); pos = geo * (1 - m) + rnd * m; idx = {n: k for k, n in enumerate(SG)}
    a_e = max(0.25, s.a(t, "municipio"))
    for u in SG:
        for v in NB[u]:
            if v in idx and u < v: d.line((tuple(pos[idx[u]]), tuple(pos[idx[v]])), fill=(110, 124, 140, int(220 * a_e)), width=2)
    hot = s.a(t, "ocho") > 0
    for k, u in enumerate(SG):
        x, y = pos[k]; c = GOLD if u == EX else (BLUE if hot and u in NB[EX] else WHITE); r_ = 14 if u == EX else 9
        d.ellipse((x - r_, y - r_, x + r_, y + r_), fill=(26, 34, 44), outline=c, width=3)
    if hot: label(d, CX, 1100, "cada pueblo, unido a sus 8 vecinos más cercanos", BLUE, 24, bold=True)
    if s.a(t, "mapa") > 0: chip(d, CX, 540, "la red no ve posiciones: solo las conexiones", col=GOLD, size=24, anchor="mm", bg=(40, 32, 14, int(240 * s.a(t, "mapa"))))
    if s.a(t, "nombre") > 0:
        a = s.a(t, "nombre")
        for u in [EX] + list(NB[EX])[:8]:
            x, y = pos[idx[u]]; chip(d, x, y - 26, NAMES[u], col=(GOLD if u == EX else WHITE) + (int(255 * a),), size=18, anchor="mm", bg=(8, 12, 18, int(220 * a)))

RING1 = list(NB[EX])[:8]; RING2 = []
for u in RING1:
    for v in NB[u]:
        if v != EX and v not in RING1 and v not in RING2 and len([w for w in RING2 if w in NB[u]]) < 2: RING2.append(v)
RING2 = RING2[:16]
def messages(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); cx, cy = CX, 830
    p1 = {u: (cx + 180 * math.cos(2 * math.pi * k / len(RING1) - math.pi / 2), cy + 180 * math.sin(2 * math.pi * k / len(RING1) - math.pi / 2)) for k, u in enumerate(RING1)}
    p2 = {}
    for k, v in enumerate(RING2):
        ang = 2 * math.pi * k / len(RING2) - math.pi / 2; p2[v] = (cx + 310 * math.cos(ang), cy + 250 * math.sin(ang))
    for u in RING1: d.line((cx, cy, *p1[u]), fill=(90, 104, 122), width=2)
    for v in RING2:
        u = min(RING1, key=lambda q: np.hypot(p1[q][0] - p2[v][0], p1[q][1] - p2[v][1])); d.line((*p1[u], *p2[v]), fill=(70, 84, 100), width=2)
    a_e, a_m, a_2 = s.a(t, "envia"), s.a(t, "mezcla"), s.a(t, "dos")
    if a_e > 0:
        ph = ((t - s.cue("envia")) * 0.9) % 1
        for u in RING1: x, y = p1[u]; d.ellipse((x + (cx - x) * ph - 6, y + (cy - y) * ph - 6, x + (cx - x) * ph + 6, y + (cy - y) * ph + 6), fill=col_of(MLPF[u]))
    if a_2 > 0:
        ph = ((t - s.cue("dos")) * 0.9) % 1
        for v in RING2:
            u = min(RING1, key=lambda q: np.hypot(p1[q][0] - p2[v][0], p1[q][1] - p2[v][1])); x, y = p2[v]; tx, ty = p1[u]
            d.ellipse((x + (tx - x) * ph - 5, y + (ty - y) * ph - 5, x + (tx - x) * ph + 5, y + (ty - y) * ph + 5), fill=col_of(MLPF[v]))
        d.ellipse((cx - 245, cy - 205, cx + 245, cy + 205), outline=GOLD + (int(120 * a_2),), width=2); label(d, cx + 250, cy - 240, "2 saltos", GOLD + (int(255 * a_2),), 22, bold=True)
    for v in RING2: x, y = p2[v]; d.ellipse((x - 11, y - 11, x + 11, y + 11), fill=col_of(MLPF[v]), outline=(8, 12, 18), width=2)
    for u in RING1: x, y = p1[u]; d.ellipse((x - 15, y - 15, x + 15, y + 15), fill=col_of(MLPF[u]), outline=(8, 12, 18), width=2)
    c0 = np.array(col_of(MLPF[EX]), float); c1 = np.array(col_of(GNNF[EX]), float); cc = tuple(int(v) for v in c0 + (c1 - c0) * a_m)
    d.ellipse((cx - 30, cy - 30, cx + 30, cy + 30), fill=cc, outline=WHITE, width=4)
    chip(d, cx, cy + 58, NAMES[EX], col=WHITE, size=22, anchor="mm", bg=(8, 12, 18, 230))
    label(d, CX, 545, "color = lo que cada pueblo «cree» por su nombre", SOFT, 22)
    txt = f"por su nombre: {COMM[MLPF[EX]]}" if a_m < 0.5 else f"con sus vecinos: {COMM[GNNF[EX]]} ✓"
    chip(d, CX, 1125, txt.replace(" ✓", ""), col=GREEN if a_m >= 0.5 else RED, size=26, anchor="mm", bg=(14, 22, 30, 240))

F_G = Formula([("h", r"$H'$"), ("eq", "$=$"), ("s", r"$\sigma($"), ("a", r"$\hat{A}$"), ("H", r"$H$"), ("W", r"$W$"), ("c", r"$)$")], px=74)
F_A = Formula([("a", r"$\hat{A}$"), ("eq", "$=$"), ("n", r"$D^{-1/2}\,(A + I)\,D^{-1/2}$")], px=56)
TINY = [EX] + RING1[:5]
def formula(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    F_G.draw(lay, CX, 600, {"h": hl(s, t, "capa"), "eq": (WHITE, s.a(t, "capa")), "s": (WHITE, s.a(t, "capa")), "c": (WHITE, s.a(t, "capa")),
                            "a": hl(s, t, "matriz", win=1.6), "H": hl(s, t, "caracteristicas", win=1.6), "W": hl(s, t, "pesos", win=1.6)})
    for k, (cue, txt) in enumerate((("caracteristicas", "H: lo que sabe cada nodo"), ("pesos", "W: pesos que se aprenden"), ("matriz", "Â: quién es vecino de quién"))):
        a = s.a(t, cue); label(d, CX, 690 + k * 38, txt, (GOLD if 0 <= t - s.cue(cue) < 1.6 else SOFT) + (int(255 * a),), 24, bold=True)
    a_m = s.a(t, "matriz")
    if a_m > 0:
        n = len(TINY); A = np.array([[1.0 if (i == j or TINY[j] in NB[TINY[i]]) else 0 for j in range(n)] for i in range(n)])
        dg = A.sum(1); An = A / np.sqrt(np.outer(dg, dg)); nm = s.a(t, "normalizada"); M = A * (1 - nm) + An * nm; cs = 48; x0, y0 = 110, 840
        for i in range(n):
            for j in range(n):
                v = M[i, j]; col = tuple(int(18 + (g - 18) * min(1, v)) for g in GOLD)
                d.rectangle((x0 + j * cs, y0 + i * cs, x0 + (j + 1) * cs - 4, y0 + (i + 1) * cs - 4), fill=col + (int(255 * a_m),))
                if v > 0: label(d, x0 + j * cs + 22, y0 + i * cs + 22, dec(v, 1) if nm > 0.5 else "1", (8, 12, 18, int(255 * a_m)), 16, bold=True)
        label(d, x0 + n * cs / 2, y0 - 22, "Â de 6 pueblos", SOFT + (int(255 * a_m),), 20)
        F_A.draw(lay, 730, 960, {"a": hl(s, t, "normalizada"), "eq": (WHITE, nm), "n": hl(s, t, "normalizada", win=1.8)})
        if nm > 0: label(d, 730, 1030, "nadie pesa más por tener muchos vecinos", GOLD + (int(255 * nm),), 20, bold=True)

CODE = """import scipy.sparse as sp

A = sp.csr_matrix(vecinos) + sp.eye(N)
d = np.asarray(A.sum(1)).ravel()
D = sp.diags(1 / np.sqrt(d))
A_hat = D @ A @ D             # normalizada

AX = A_hat @ (A_hat @ X)      # 2 saltos
H  = relu(AX @ W1 + b1)       # capa densa
Z  = A_hat @ (H @ W2 + b2)    # 17 comunidades
p  = softmax(Z)"""
CB = CodeBlock(CODE, size=27, lh=1.45)
GROUPS = [("matriz", range(2, 6)), ("dos", {7}), ("densa", {8}), ("diecisiete", {9, 10})]
def code(lay, s, t):
    act, foc = None, 0.0
    for name, rng_ in GROUPS:
        if t >= s.cue(name) - 0.1: act, foc = set(rng_), ease((t - s.cue(GROUPS[0][0])) / 0.4)
    if act is not None and t >= s.cue("dispersa") - 0.1 and t < s.cue("dos") - 0.1: act = {0, 2}
    CB.draw(lay, 104, 570, act, foc)
    label(ImageDraw.Draw(lay, "RGBA"), CX, 1140, "X: letras del nombre de cada pueblo (pares y tríos de letras)", MUTED, 20)

def kidx(s, t):
    t0 = s.cue("diez"); u = np.clip((t - t0) / max(1.0, s.cue("noventa") + 0.8 - t0), 0, 1) if t >= t0 else 0
    return KEYS[int(round(u ** 0.9 * (len(KEYS) - 1)))]
def demo(lay, s, t):
    e = kidx(s, t); d = ImageDraw.Draw(lay, "RGBA")
    for k, (kind, lab, col) in enumerate((("mlp", "solo el nombre", WHITE), ("gnn", "nombre + vecinos", GOLD))):
        x0 = 90 + k * 470; p = R[kind]["preds"][e]; cols = [GREEN if p[i] == Y[i] else RED for i in range(N)]
        RN.draw_map(lay, (x0, 560, 430), cols, mask=TEset, r=1.4); d = ImageDraw.Draw(lay, "RGBA")
        chip(d, x0, 548, T_(lab), col=col, size=22); chip(d, x0 + 430, 600, f"{dec(ACC[kind][e - 1] * 100)} %", col=col, size=32, anchor="rt", bg=(8, 12, 18, 235))
    label(d, CX, 915, f"época {e} · verde acierta, rojo falla · examen: el 90 % que no conoce", SOFT, 20)
    E_ = len(ACC["gnn"]); xs = np.arange(1, E_ + 1)
    chart(lay, (90, 940, 990, 1170), [dict(xs=xs, ys=ACC["mlp"] * 100, upto=e, col=WHITE, name="nombre", val_fmt=lambda v: dec(v, 1) + " %"),
                                     dict(xs=xs, ys=ACC["gnn"] * 100, upto=e, col=GOLD, name="+ vecinos", val_fmt=lambda v: dec(v, 1) + " %")],
          "acierto en el examen", xmax=E_, ylim=(0, 100), yticks=[25, 50, 75, 100], logy=False, xlabel="época", yfmt=lambda v: f"{int(v)}")

def trap(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    rows = [("castilla", "decir siempre «Castilla y León»", 0.28, MUTED), (None, "solo el nombre", ACC["mlp"][-1], WHITE), (None, "nombre + vecinos", ACC["gnn"][-1], GOLD)]
    for i, (cue, lab, v, col) in enumerate(rows):
        a = s.a(t, cue) if cue else 1.0; y = 580 + i * 80
        if a <= 0: continue
        d.text((100, y - 22), lab, font=font(FS["monob"], 24), fill=col + (int(255 * a),), anchor="lm")
        d.rounded_rectangle((100, y, 100 + 760 * v, y + 30), radius=8, fill=col + (int(255 * a),)); label(d, 120 + 760 * v + 40, y + 15, f"{dec(v * 100, 1)} %", col + (int(255 * a),), 24, bold=True)
    E_ = len(ACC["gnn"]); xs = np.arange(1, E_ + 1); a_s = s.a(t, "memoriza")
    chart(lay, (90, 830, 990, 1170), [dict(xs=xs, ys=R["mlp"]["L"][:, 0], upto=1e9 if a_s > 0 else 0, col=BLUE, name="aprende", val_fmt=lambda v: dec(v, 2)),
                                     dict(xs=xs, ys=R["mlp"]["L"][:, 1], upto=1e9 if a_s > 0 else 0, col=RED, name="examen", val_fmt=lambda v: dec(v, 2))],
          "red del nombre", xmax=E_, ylim=(0.02, 5), yticks=[0.1, 1], xlabel="época", yfmt=lambda v: dec(v, 1) if v < 1 else "1")
    if s.a(t, "sube") > 0: chip(d, CX, 1000, "aprende cada vez mejor… y el examen empeora", col=RED, size=22, anchor="mm", bg=(40, 14, 18, int(235 * s.a(t, "sube"))))

def errors(lay, s, t):
    p = GNNF; err = TEset & (p != Y); cols = [RED if err[i] else (52, 64, 78) for i in range(N)]
    px, py, h = RN.draw_map(lay, (90, 540, 560), cols, r=2.0); d = ImageDraw.Draw(lay, "RGBA")
    if s.a(t, "fronteras") > 0: label(d, 370, 540 + h + 26, "rojo: pueblos en los que se equivoca", RED + (int(255 * s.a(t, "fronteras")),), 20, bold=True)
    a = s.a(t, "trevino")
    if a > 0:
        x, y = px[TRE], py[TRE]; d.ellipse((x - 18, y - 18, x + 18, y + 18), outline=GOLD + (int(255 * a),), width=5); d.line((x + 18, y, 680, 620), fill=GOLD + (int(255 * a),), width=3)
        ccx, ccy = 830, 760; nbs = NB[TRE]
        for k, v in enumerate(nbs):
            ang = 2 * math.pi * k / len(nbs) - math.pi / 2; vx, vy = ccx + 115 * math.cos(ang), ccy + 115 * math.sin(ang)
            d.line((ccx, ccy, vx, vy), fill=(110, 124, 140, int(255 * a)), width=2)
            d.ellipse((vx - 16, vy - 16, vx + 16, vy + 16), fill=col_of(Y[v]) + (int(255 * a),), outline=(8, 12, 18), width=2)
        d.ellipse((ccx - 24, ccy - 24, ccx + 24, ccy + 24), fill=col_of(Y[TRE]) + (int(255 * a),), outline=WHITE + (int(255 * a),), width=4)
        label(d, ccx, 610, "Condado de Treviño", GOLD + (int(255 * a),), 24, bold=True)
        cnt = collections.Counter(PROV[v] for v in nbs); top = cnt.most_common(1)[0]
        label(d, ccx, 905, f"es de Burgos · {top[1]} de sus {len(nbs)} vecinos: Álava", SOFT + (int(255 * max(a, s.a(t, "alava"))),), 20, bold=True)
    if s.a(t, "vecinos") > 0:
        a2 = s.a(t, "vecinos"); chip(d, 830, 960, f"la red dice: {COMM[GNNF[TRE]]}", col=RED + (int(255 * a2),), size=24, anchor="mm", bg=(40, 14, 18, int(240 * a2)))
        label(d, CX, 1100, "sus vecinos le dan la respuesta equivocada", GOLD + (int(255 * a2),), 26, bold=True)

INK = None
def next_(lay, s, t):
    global INK
    d = ImageDraw.Draw(lay, "RGBA")
    rows = [("moleculas", "moléculas, redes sociales, recomendaciones")]
    a = s.a(t, "moleculas"); d.text((CX, 590), rows[0][1], font=font(FS["body"], 34), fill=WHITE + (int(255 * a),), anchor="mm")
    d.text((CX, 680), "ÚLTIMO EPISODIO", font=font(FS["monob"], 30), fill=SOFT, anchor="mm")
    d.text((CX, 755), T_("Redes que sueñan"), font=font(FS["head"], 66), fill=GOLD, anchor="mm")
    if INK is None:
        im = Image.open("cuadros_pd/style_wave.jpg").convert("L").resize((560, 310), Image.LANCZOS); g = np.asarray(im) < 140
        INK = Image.fromarray(np.where(g[..., None], np.array([22, 26, 34], np.uint8), np.array([240, 234, 220], np.uint8)).astype(np.uint8))
    paste_img(lay, INK, CX - 280, 820, 560, 310, radius=14, alpha=s.a(t, "origenes")); d = ImageDraw.Draw(lay, "RGBA")
    a = s.a(t, "sonando"); d.line((876, 360, 912, 360), fill=GOLD + (int(255 * a),), width=4); d.polygon([(912, 352), (912, 368), (926, 360)], fill=GOLD + (int(255 * a),))

def end(tl_, t):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    d.text((W / 2, 720), "REDES NEURONALES", font=font(FS["head"], 84), fill=GOLD, anchor="mm")
    d.text((W / 2, 810), "Ep. 7 · Redes de grafos", font=font(FS["body"], 46), fill=WHITE, anchor="mm")
    d.rounded_rectangle((200, 900, 880, 1140), radius=24, fill=(16, 22, 32, 255), outline=(44, 58, 74, 255), width=2)
    d.text((W / 2, 960), "ÚLTIMO EPISODIO: REDES QUE SUEÑAN", font=font(FS["monob"], 26), fill=GOLD, anchor="mm")
    d.text((W / 2, 1020), "código completo en la descripción", font=font(FS["body"], 30), fill=SOFT, anchor="mm")
    d.text((W / 2, 1090), HANDLE, font=font(FS["head"], 52), fill=WHITE, anchor="mm")
    return img

SCENES = dict(hook=hook, problem=problem, graph=graph, messages=messages, formula=formula, code=code, demo=demo, trap=trap, errors=errors, next=next_, _end=end)
if __name__ == "__main__":
    cli(tl, SCENES, "ep07_gnn_piloto")
