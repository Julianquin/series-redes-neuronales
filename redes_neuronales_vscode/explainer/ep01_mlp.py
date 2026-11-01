"""Episodio 01 · Perceptrón multicapa.  Uso: python3 ep01_mlp.py info | prev f… | chunk 85 | encode"""
import os, pickle, math
import numpy as np
from PIL import Image, ImageDraw
from explainer import *

HERE = os.path.dirname(os.path.abspath(__file__))
tl = Timeline(os.path.join(HERE, "guion_ep01_mlp.yaml"))
P0 = PANEL; CX = (P0[0] + P0[2]) / 2

# ---------- datos reales ----------
RUN = pickle.load(open("work/run.pkl", "rb"))          # Ep. 1: MLP [2, 64, 64, 1] entrenado en espirales
XTR, YTR, L, SN = RUN["Xtr"], RUN["ytr"], RUN["L"], RUN["snaps"]; KEYS = sorted(SN)
HERO = Image.fromarray(np.load("work/hd_hero.npy"))     # La noche estrellada pintada por el MLP con Fourier
def predict(P, X):
    h = X
    for i, (Wt, b) in enumerate(P):
        z = h @ Wt + b; h = np.maximum(z, 0) if i < len(P) - 1 else 1 / (1 + np.exp(-z))
    return h[:, 0]
LIM = 6.6; G = 110; gx = np.linspace(-LIM, LIM, G); GX, GY = np.meshgrid(gx, -gx); GRID = np.c_[GX.ravel(), GY.ravel()]
_dm = {}
def decision(P, size, key):
    if key in _dm: return _dm[key]
    p = predict(P, GRID).reshape(G, G).astype(np.float32); pu = np.array(Image.fromarray(p, "F").resize((size, size), Image.BILINEAR))
    conf = np.abs(2 * pu - 1) ** 0.6; base = np.array((14, 20, 30), np.float32)
    tint = np.array(BLUE, np.float32) * (1 - pu[..., None]) + np.array(GOLD, np.float32) * pu[..., None]
    rgb = base + (tint - base) * (0.30 * conf[..., None]); gy_, gx_ = np.gradient(pu); gm = np.hypot(gx_, gy_)
    band = np.clip(1.6 - np.abs(pu - 0.5) / (gm + 1e-6) / 1.4, 0, 1) * (gm > 1e-4)
    rgb = rgb * (1 - band[..., None] * .9) + np.array(WHITE, np.float32) * band[..., None] * .9
    out = Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).convert("RGBA"); _dm.clear(); _dm[key] = out; return out
def scatter(d, x0, y0, s, alpha=1.0, r=5):
    for (x, y), c in zip(XTR, YTR):
        px, py = x0 + (x + LIM) / (2 * LIM) * s, y0 + (LIM - y) / (2 * LIM) * s
        d.ellipse((px - r, py - r, px + r, py + r), fill=(GOLD if c else BLUE) + (int(255 * alpha),), outline=(8, 12, 18, int(255 * alpha)), width=2)

def hl(s, t, name, base=WHITE, win=1.2):
    """Color de un término: dorado justo después de su cue, luego vuelve a blanco. Alfa: aparece en su cue."""
    a = s.a(t, name, 0.35); hot = 0 <= t - s.cue(name) < win
    return (GOLD if hot else base, a)

def label(d, x, y, txt, col=SOFT, size=26, anchor="mm", bold=False):
    d.text((x, y), txt, font=font(FS["monob" if bold else "mono"], size), fill=col, anchor=anchor)

# ---------- escenas ----------
def hook(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); z = 1 + 0.04 * t / s.dur; w, h = 880, 550
    cw, ch = HERO.width / z, HERO.height / z
    im = HERO.crop(((HERO.width - cw) / 2, (HERO.height - ch) / 2, (HERO.width + cw) / 2, (HERO.height + ch) / 2))
    paste_img(lay, im, CX - w / 2, 530, w, h, radius=18); d = ImageDraw.Draw(lay, "RGBA")
    a = s.a(t, "red")
    if a > 0:                                   # capa aparte: dibujar con transparencia sobre la imagen la «agujerearía»
        tmp = Image.new("RGBA", lay.size, (0, 0, 0, 0)); chip(ImageDraw.Draw(tmp), CX - w / 2 + 16, 546, "MLP · 3 capas ocultas · codificación de Fourier", col=WHITE, size=24, bg=(8, 12, 18, 220))
        lay.alpha_composite(faded(tmp, a)); d = ImageDraw.Draw(lay, "RGBA")
    pass

def problem(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); x0, y0, sz = CX - 280, 540, 560
    scatter(d, x0, y0, sz, alpha=s.a(t, "espiral", 0.6))
    a_l = s.a(t, "recta"); fail = s.a(t, "ninguna", 0.3); bend = s.a(t, "doblar", 0.6)
    if a_l > 0 and bend < 1:
        ang = math.radians(20 + 50 * math.sin((t - s.cue("recta")) * 1.3)); L_ = 420
        cx_, cy_ = x0 + sz / 2, y0 + sz / 2; col = RED if fail > 0.5 else WHITE
        d.line((cx_ - L_ * math.cos(ang), cy_ - L_ * math.sin(ang), cx_ + L_ * math.cos(ang), cy_ + L_ * math.sin(ang)), fill=col + (int(255 * a_l * (1 - bend)),), width=6)
    if fail > 0 and bend < 1: chip(d, CX, 1135, "ninguna recta los separa", col=RED + (int(255 * fail * (1 - bend)),), size=32, anchor="mm", bg=(40, 14, 18, int(240 * fail * (1 - bend))))
    if bend > 0:
        dm = faded(decision(SN[KEYS[-1]], sz, "final").copy(), bend * 0.95); lay.alpha_composite(dm, (int(x0), int(y0)))
        d = ImageDraw.Draw(lay, "RGBA"); scatter(d, x0, y0, sz, alpha=1.0)
        chip(d, CX, 1135, "hace falta una frontera curva", col=GOLD + (int(255 * bend),), size=32, anchor="mm", bg=(40, 32, 14, int(240 * bend)))

F_Z = Formula([("z", r"$z$"), ("eq", "$=$"), ("wx", r"$\mathbf{w}\cdot\mathbf{x}$"), ("plus", "$+$"), ("b", "$b$")], px=72)
F_A = Formula([("a", "$a$"), ("eq", "$=$"), ("relu", r"$\mathrm{ReLU}(z)$"), ("eq2", "$=$"), ("max", r"$\max(0,\ z)$")], px=64)
def neuron(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); a_in = s.a(t, "entradas"); a_w = s.a(t, "peso"); a_b = s.a(t, "sesgo"); a_act = s.a(t, "activacion")
    xs = [(170, 610, "x₁", "0,8"), (170, 760, "x₂", "−1,2"), (170, 910, "x₃", "0,5")]; sx, sy = 520, 760
    for i, (x, y, nm, v) in enumerate(xs):
        hot = 0 <= t - s.cue("entradas") < 1.2
        d.line((x + 40, y, sx - 64, sy), fill=(GOLD if 0 <= t - s.cue("peso") < 1.4 else (90, 104, 122)) + (int(255 * max(a_in, .3)),), width=3 + int(3 * a_w))
        d.ellipse((x - 40, y - 40, x + 40, y + 40), fill=(22, 30, 40, 255), outline=(GOLD if hot else BLUE) + (int(255 * max(a_in, .35)),), width=4)
        label(d, x, y - 4, nm, WHITE, 30, bold=True); label(d, x - 82, y, v, SOFT if a_in > 0 else (60, 70, 80), 24)
        if a_w > 0: label(d, (x + sx) / 2 - 10, (y + sy) / 2 - 22, f"w{'₁₂₃'[i]}", GOLD + (int(255 * a_w),), 26, bold=True)
    d.ellipse((sx - 64, sy - 64, sx + 64, sy + 64), fill=(22, 30, 40, 255), outline=WHITE, width=4); label(d, sx, sy - 4, "Σ", WHITE, 54, bold=True)
    if a_b > 0:
        d.rounded_rectangle((sx - 44, 590, sx + 44, 640), radius=12, fill=(40, 32, 14, int(255 * a_b)), outline=GOLD + (int(255 * a_b),), width=3)
        label(d, sx, 615, "+ b", GOLD + (int(255 * a_b),), 28, bold=True); d.line((sx, 640, sx, sy - 66), fill=GOLD + (int(255 * a_b),), width=3)
    if a_act > 0:
        bx0, by0 = 660, 690; d.line((sx + 66, sy, bx0, sy), fill=WHITE + (int(255 * a_act),), width=4)
        d.rounded_rectangle((bx0, by0, bx0 + 150, by0 + 140), radius=16, fill=(22, 30, 40, int(255 * a_act)), outline=WHITE + (int(255 * a_act),), width=3)
        a_r = s.a(t, "relu"); a_c = s.a(t, "cero")
        if a_r > 0:
            ox, oy = bx0 + 75, by0 + 100; d.line((bx0 + 12, oy, bx0 + 138, oy), fill=MUTED, width=2); d.line((ox, by0 + 14, ox, by0 + 128), fill=MUTED, width=2)
            d.line((bx0 + 14, oy, ox, oy), fill=(RED if a_c > 0 else GOLD) + (int(255 * a_r),), width=6)
            d.line((ox, oy, bx0 + 136, by0 + 22), fill=GOLD + (int(255 * a_r),), width=6)
            label(d, bx0 + 75, by0 + 162, "ReLU", GOLD + (int(255 * a_r),), 26, bold=True)
            if s.a(t, "doblez") > 0:
                pr = 10 + 6 * math.sin((t - s.cue("doblez")) * 6); d.ellipse((ox - pr, oy - pr, ox + pr, oy + pr), outline=GOLD, width=4)
        d.line((bx0 + 152, sy, 880, sy), fill=WHITE + (int(255 * a_act),), width=4)
        d.ellipse((880, sy - 40, 960, sy + 40), fill=(22, 30, 40, 255), outline=GOLD + (int(255 * a_act),), width=4); label(d, 920, sy - 4, "a", WHITE, 32, bold=True)
    stz = {"z": hl(s, t, "entradas"), "eq": (WHITE, s.a(t, "entradas")), "wx": hl(s, t, "peso"), "plus": (WHITE, s.a(t, "sesgo")), "b": hl(s, t, "sesgo")}
    F_Z.draw(lay, CX, 1030, stz)
    sta = {"a": hl(s, t, "relu"), "eq": (WHITE, s.a(t, "relu")), "relu": hl(s, t, "relu"), "eq2": (WHITE, s.a(t, "cero")), "max": hl(s, t, "cero")}
    F_A.draw(lay, CX, 1118, sta)

F_H = Formula([("h", r"$\mathbf{h}$"), ("eq", "$=$"), ("relu", r"$\mathrm{ReLU}(W_1\mathbf{x} + \mathbf{b}_1)$")], px=60)
F_Y = Formula([("y", r"$\hat{y}$"), ("eq", "$=$"), ("sig", r"$\sigma(W_2\mathbf{h} + b_2)$")], px=60)
NET = [2, 5, 5, 1]; NX = [180, 400, 640, 880]
def net_pos(top=580, bot=930):
    out = []
    for n, x in zip(NET, NX):
        ys = np.linspace(top, bot, n + 2)[1:-1] if n > 1 else [(top + bot) / 2]; out.append([(x, y) for y in ys])
    return out
def layers(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); pos = net_pos(); vis = [1.0, s.a(t, "capa"), s.a(t, "apilamos"), s.a(t, "apilamos")]
    for L_ in range(3):
        a = min(vis[L_], vis[L_ + 1])
        for p in pos[L_]:
            for q in pos[L_ + 1]: d.line((p, q), fill=(80, 96, 116, int(200 * a)), width=2)
    if s.a(t, "fluye") > 0:
        ph = ((t - s.cue("fluye")) * 0.9) % 1.0; seg = int(ph * 3); u = ph * 3 - seg
        for p in pos[seg]:
            for q in pos[seg + 1]:
                x, y = p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u; d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=GOLD + (220,))
    hot_o = 0 <= t - s.cue("ocultas") < 1.4; hot_s = 0 <= t - s.cue("salida") < 1.4
    for L_, col_ in enumerate(pos):
        for (x, y) in col_:
            oc = GOLD if (L_ in (1, 2) and hot_o) or (L_ == 3 and hot_s) else (BLUE if L_ == 0 else WHITE)
            d.ellipse((x - 26, y - 26, x + 26, y + 26), fill=(22, 30, 40, int(255 * vis[L_])), outline=oc + (int(255 * vis[L_]),), width=4)
    for x, txt, L_ in ((NX[0], "entrada", 0), ((NX[1] + NX[2]) / 2, "capas ocultas", 1), (NX[3], "salida", 3)):
        label(d, x, 975, txt, SOFT + (int(255 * vis[L_]),), 26)
    if s.a(t, "dobla") > 0: chip(d, CX, 545, "cada capa dobla el espacio un poco más", col=GOLD + (int(255 * s.a(t, "dobla")),), size=26, anchor="mm", bg=(40, 32, 14, int(240 * s.a(t, "dobla"))))
    F_H.draw(lay, CX, 1048, {"h": hl(s, t, "capa"), "eq": (WHITE, s.a(t, "capa")), "relu": hl(s, t, "capa")})
    F_Y.draw(lay, CX, 1125, {"y": hl(s, t, "salida"), "eq": (WHITE, s.a(t, "salida")), "sig": hl(s, t, "salida")})

F_L = Formula([("L", r"$\mathcal{L}$"), ("eq", "$=$"), ("m", r"$-\,[$"), ("t1", r"$y\,\log\hat{y}$"), ("plus", "$+$"),
               ("t2", r"$(1-y)\,\log(1-\hat{y})$"), ("c", r"$]$")], px=60)
def loss(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); a_al = s.a(t, "aleatorios")
    chip(d, CX, 545, "pesos al azar → predicciones al azar", col=SOFT + (int(255 * a_al),), size=26, anchor="mm", bg=(30, 40, 54, int(255 * a_al)))
    # Entre «aleatorios» (1,8 s) y «pérdida» (5,9 s) el panel se quedaba con el chip y nada más.
    # Se enseña la frontera real en la iteración 1 —ruido— que es lo que la frase describe y
    # prepara el s08, donde esa misma frontera acaba siguiendo la espiral. Capa aparte: dibujar
    # con alfa sobre lo ya pintado lo agujerearía.
    a_rnd = s.a(t, "aleatorios") * (1 - s.a(t, "perdida"))
    if a_rnd > 0.01:
        k0 = KEYS[0]; szr = 300; xr, yr = (W - szr) / 2, 600
        tmp = Image.new("RGBA", lay.size, (0, 0, 0, 0))
        tmp.alpha_composite(decision(SN[k0], szr, k0), (int(xr), yr))
        dr = ImageDraw.Draw(tmp, "RGBA"); scatter(dr, xr, yr, szr, r=3)
        dr.rectangle((xr, yr, xr + szr, yr + szr), outline=(44, 58, 74), width=2)
        label(dr, CX, yr + szr + 34, f"iteración {fmt(k0)}: la frontera es ruido", MUTED, 24)
        lay.alpha_composite(faded(tmp, a_rnd)); d = ImageDraw.Draw(lay, "RGBA")

    ap = s.a(t, "perdida"); hot = 0 <= t - s.cue("entropia") < 1.6
    F_L.draw(lay, CX, 640, {k: ((GOLD if hot and k in ("t1", "t2") else WHITE), ap) for k in ("L", "eq", "m", "t1", "plus", "t2", "c")}, max_w=900)
    a_c = s.a(t, "castiga", 0.8)
    if a_c > 0:
        x0, y0, x1, y1 = 200, 720, 900, 1100; d.line((x0, y1, x1, y1), fill=MUTED, width=2); d.line((x0, y0, x0, y1), fill=MUTED, width=2)
        label(d, (x0 + x1) / 2, y1 + 30, "probabilidad que da a la respuesta correcta (ŷ)", MUTED, 22); label(d, x0 - 20, y0 + 10, "pérdida", MUTED, 22, anchor="rm")
        X = np.linspace(0.02, 1, 120); Y = -np.log(X); sx = lambda v: x0 + v * (x1 - x0); sy = lambda v: y1 - min(v, 4) / 4 * (y1 - y0)
        n = max(2, int(len(X) * a_c)); d.line([(sx(a), sy(b)) for a, b in zip(X[::-1][:n], Y[::-1][:n])], fill=GOLD, width=6, joint="curve")
        if s.a(t, "seguro") > 0:
            u = ease((t - s.cue("seguro")) / 1.4); pv = 0.95 + (0.05 - 0.95) * u; lv = -math.log(pv)
            px, py = sx(pv), sy(lv); d.ellipse((px - 12, py - 12, px + 12, py + 12), fill=RED if pv < .3 else WHITE, outline=WHITE, width=3)
            chip(d, px + 24, py - 20, f"ŷ = {dec(pv, 2)} → pérdida {dec(lv, 2)}", col=WHITE, size=24, bg=(8, 12, 18, 230))

F_C = Formula([("lhs", r"$\frac{\partial\mathcal{L}}{\partial W_1}$"), ("eq", "$=$"), ("a1", r"$\frac{\partial\mathcal{L}}{\partial\hat{y}}$"), ("d1", r"$\cdot$"),
               ("a2", r"$\frac{\partial\hat{y}}{\partial\mathbf{h}}$"), ("d2", r"$\cdot$"), ("a3", r"$\frac{\partial\mathbf{h}}{\partial W_1}$")], px=66)
F_U = Formula([("w", "$W$"), ("ar", r"$\leftarrow$"), ("w2", "$W$"), ("mi", "$-$"), ("eta", r"$\eta$"), ("g", r"$\frac{\partial\mathcal{L}}{\partial W}$")], px=66)
def grad(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); pos = [[(x, y) for y in np.linspace(560, 680, n + 2)[1:-1]] if n > 1 else [(x, 620)] for n, x in zip([2, 4, 4, 1], [260, 440, 640, 820])]
    for L_ in range(3):
        for p in pos[L_]:
            for q in pos[L_ + 1]: d.line((p, q), fill=(70, 86, 104), width=2)
    if s.a(t, "atras") > 0:
        ph = ((t - s.cue("atras")) * 0.9) % 1.0; seg = 2 - int(ph * 3); u = ph * 3 - int(ph * 3)
        for p in pos[seg + 1]:
            for q in pos[seg]:
                x, y = p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u; d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=RED + (230,))
        chip(d, 860, 560, "← hacia atrás", col=RED, size=24, anchor="lm", bg=(40, 14, 18, 230))
    for col_ in pos:
        for (x, y) in col_: d.ellipse((x - 17, y - 17, x + 17, y + 17), fill=(22, 30, 40), outline=WHITE, width=3)
    tc = s.cue("cadena"); st = {"lhs": hl(s, t, "gradiente"), "eq": (WHITE, s.a(t, "gradiente"))}
    for k, (key, dk) in enumerate((("a1", "d1"), ("a2", "d2"), ("a3", None))):
        a = ease((t - tc - 0.35 * k) / 0.35); hot = 0 <= t - tc - 0.35 * k < 1.2
        st[key] = (GOLD if hot else WHITE, a)
        if dk: st[dk] = (WHITE, a)
    F_C.draw(lay, CX, 820, st)
    if s.a(t, "retro") > 0: chip(d, CX, 912, "retropropagación = regla de la cadena, capa a capa", col=SOFT + (int(255 * s.a(t, "retro")),), size=24, anchor="mm", bg=(30, 40, 54, int(255 * s.a(t, "retro"))))
    F_U.draw(lay, CX - 160, 1060, {"w": hl(s, t, "pasito"), "ar": (WHITE, s.a(t, "pasito")), "w2": (WHITE, s.a(t, "pasito")), "mi": (WHITE, s.a(t, "pasito")),
                                   "eta": hl(s, t, "pasito"), "g": hl(s, t, "contraria")})
    a_b = s.a(t, "pasito")
    if a_b > 0:
        bx0, bx1, by0, by1 = 640, 960, 990, 1150; X = np.linspace(-1, 1, 60); f = lambda v: by1 - 10 - (1 - v ** 2) * 0 - (v ** 2) * (by1 - by0 - 20)
        f = lambda v: by0 + 10 + (1 - v ** 2) * (by1 - by0 - 30)
        pts = [(bx0 + (v + 1) / 2 * (bx1 - bx0), by1 - 10 - (1 - (1 - v ** 2)) * (by1 - by0 - 30)) for v in X]
        d.line(pts, fill=MUTED + (int(255 * a_b),), width=4, joint="curve")
        k = int(np.clip((t - s.cue("pasito") - 0.5) / 0.4, 0, 7)); v = 0.9 * (0.6 ** k)
        bx = bx0 + (v + 1) / 2 * (bx1 - bx0); byy = by1 - 10 - (v ** 2) * (by1 - by0 - 30) - 14
        d.ellipse((bx - 13, byy - 13, bx + 13, byy + 13), fill=GOLD + (int(255 * a_b),))
        label(d, (bx0 + bx1) / 2, by1 + 18, "η: tamaño del pasito", MUTED + (int(255 * a_b),), 22)

CODE = """# 1. pasada hacia delante
z1 = X @ W1 + b1
h  = np.maximum(0, z1)        # ReLU
z2 = h @ W2 + b2
p  = 1 / (1 + np.exp(-z2))    # sigmoide
# 2. pérdida: entropía cruzada
L = -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
# 3. retropropagación
dz2 = (p - y) / len(y)
dW2, db2 = h.T @ dz2, dz2.sum(0)
dz1 = (dz2 @ W2.T) * (z1 > 0)
dW1, db1 = X.T @ dz1, dz1.sum(0)
# 4. actualización
params = [W1, b1, W2, b2]
grads  = [dW1, db1, dW2, db2]
for P, g in zip(params, grads):
    P -= lr * g"""
CB = CodeBlock(CODE, size=27, lh=1.36)
CODE_GROUPS = [("delante", range(0, 5)), ("perdida", range(5, 7)), ("retro", range(7, 12)), ("actualizacion", range(12, 17))]
def code(lay, s, t):
    act, foc = None, 0.0
    for name, rng in CODE_GROUPS:
        if t >= s.cue(name) - 0.1: act, foc = set(rng), ease((t - s.cue(CODE_GROUPS[0][0])) / 0.4)
    CB.draw(lay, 112, 532, act, foc)

def demo(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); t0, t1 = s.cue("frontera"), s.cue("espiral")
    u = np.clip((t - t0) / max(0.5, t1 - t0), 0, 1) ** 1.2; k = KEYS[int(round(u * (len(KEYS) - 1)))]
    # La curva de error vive en el piso 2 (piso2: curva). Aquí manda la frontera, así que ocupa
    # todo el panel en vez de media pantalla con la gráfica repetida al lado.
    sz = 520; x0, y0 = (W - sz) / 2, 480; lay.alpha_composite(decision(SN[k], sz, k), (int(x0), y0)); d = ImageDraw.Draw(lay, "RGBA")
    d.rectangle((x0, y0, x0 + sz, y0 + sz), outline=(44, 58, 74), width=2); scatter(d, x0, y0, sz, r=4)
    chip(d, x0 + 10, y0 + 10, f"iteración {fmt(k)}", col=WHITE, size=24)
    acc = L[k - 1, 3]; d = ImageDraw.Draw(lay, "RGBA")
    d.text((CX, 1080), T_(f"acierta el {dec(acc * 100, 0)} % de puntos nuevos"), font=fit(FS["head"], 46, [T_(f"acierta el {dec(acc * 100, 0)} % de puntos nuevos")], 880),
           fill=GOLD if acc > .9 else WHITE, anchor="mm")
    label(d, CX, 1140, "red real: 2 → 64 → 64 → 1, entrenada en NumPy", MUTED, 22)
from reel_common import T_

META = pickle.load(open("work/hd_meta.pkl", "rb")); IDXH, PSH = META["IDX"], META["ps"]; NJ = len(IDXH)
_hd = {}
def hd(kind, j):
    k = (kind, j)
    if k not in _hd:
        if len(_hd) > 6: _hd.clear()
        _hd[k] = Image.fromarray(np.load(f"work/hd/{kind}_{j:03d}.npy"))
    return _hd[k]
F_G = Formula([("g", r"$\gamma(\mathbf{p})$"), ("eq", "$=$"), ("v", r"$[\,\sin(2\pi B\mathbf{p}),\ \cos(2\pi B\mathbf{p})\,]$")], px=56)
F_F = Formula([("f", r"$f(x,\,y)$"), ("ar", r"$\rightarrow$"), ("c", r"$(r,\,g,\,b)$")], px=60)
def painting(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); b = s.a(t, "normal", 0.5)
    if b < 1:                                   # A: la tarea — posición -> color
        w, h = 880, 550; x0, y0 = CX - w / 2, 540
        paste_img(lay, HERO, x0, y0, w, h, radius=18, alpha=1 - b); d = ImageDraw.Draw(lay, "RGBA")
        a_p = s.a(t, "posicion") * (1 - b)
        if a_p > 0:
            tt = t - s.cue("posicion"); u = 0.5 + 0.36 * math.sin(1.7 * tt + 0.6); v = 0.5 + 0.30 * math.sin(2.6 * tt + 1.9)
            px, py = x0 + u * w, y0 + v * h; col = HERO.getpixel((min(HERO.width - 1, int(u * HERO.width)), min(HERO.height - 1, int(v * HERO.height))))
            for r_, al in ((32, 90), (18, 255)): d.ellipse((px - r_, py - r_, px + r_, py + r_), outline=WHITE + (int(al * a_p),), width=3)
            f = font(FS["monob"], 40); txt = f"({dec(u, 2)} ; {dec(v, 2)})"; tw = f.getlength(txt); xs = CX - (tw + 170) / 2
            d.text((xs, 1132), txt, font=f, fill=WHITE + (int(255 * a_p),), anchor="lm"); d.text((xs + tw + 30, 1132), "→", font=f, fill=GOLD + (int(255 * a_p),), anchor="lm")
            if s.a(t, "color") > 0: d.ellipse((xs + tw + 100, 1132 - 34, xs + tw + 168, 1132 + 34), fill=col[:3] + (int(255 * a_p),), outline=WHITE, width=4)
    if b > 0:                                   # B: red normal vs red con ondas (Fourier)
        w, h = 440, 275; y0 = 560; xl, xr = 90, 550
        paste_img(lay, hd("plain", NJ - 1), xl, y0, w, h, radius=14, alpha=b); d = ImageDraw.Draw(lay, "RGBA")
        chip(d, xl + 12, y0 + 12, "red normal · (x, y)", col=WHITE, size=22)
        chip(d, xl + w - 12, y0 + h - 44, f"{dec(PSH['plain'][IDXH[-1]][1])} dB", col=WHITE, size=26, anchor="rt")
        label(d, xl + w / 2, y0 + h + 30, "una mancha borrosa", SOFT + (int(255 * s.a(t, "borrosa")),), 26)
        a_o = s.a(t, "ondas", 0.5)
        if a_o < 1:
            d.rounded_rectangle((xr, y0, xr + w, y0 + h), radius=14, outline=(70, 86, 104, int(255 * b * (1 - a_o))), width=3)
            d.text((xr + w / 2, y0 + h / 2), "?", font=font(FS["head"], 90), fill=(70, 86, 104, int(255 * b * (1 - a_o))), anchor="mm")
        if a_o > 0:
            u = np.clip((t - s.cue("ondas")) / max(0.8, s.cue("pinceladas") - s.cue("ondas")), 0, 1); j = int(round(u ** 0.8 * (NJ - 1)))
            paste_img(lay, hd("fourier", j), xr, y0, w, h, radius=14, alpha=a_o, outline=GOLD); d = ImageDraw.Draw(lay, "RGBA")
            chip(d, xr + 12, y0 + 12, "red + ondas (Fourier)", col=GOLD, size=22)
            chip(d, xr + w - 12, y0 + h - 44, f"{dec(PSH['fourier'][IDXH[j]][1])} dB", col=GOLD, size=26, anchor="rt")
            label(d, xr + w / 2, y0 + h + 30, "hasta las pinceladas", SOFT + (int(255 * s.a(t, "pinceladas")),), 26)
        F_G.draw(lay, CX, 990, {"g": hl(s, t, "ondas"), "eq": (WHITE, s.a(t, "ondas")), "v": hl(s, t, "ondas")}, max_w=900)
        label(d, CX, 1060, "p = (x, y) · B = 96 frecuencias al azar · más dB = más parecido", MUTED + (int(255 * a_o),), 22)
        label(d, CX, 1112, "aprende este cuadro de memoria: 230 × 144 píxeles", MUTED + (int(255 * s.a(t, "pinceladas")),), 22)
    if b < 1 and s.a(t, "posicion") > 0:
        pass

PIX = np.asarray(HERO.resize((24, 15), Image.LANCZOS)); PERM = np.random.default_rng(3).permutation(24 * 15)
def when(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    rows = [("tabulares", "✓", GREEN, "datos tabulares: precios, fichas de clientes"), ("arboles", "≈", GOLD, "pero los árboles potenciados suelen ganarle"),
            ("limitacion", "✗", RED, "trata cada entrada por separado")]
    for i, (cue, sym, col, txt) in enumerate(rows):
        a = s.a(t, cue); y = 570 + i * 78
        if a <= 0: continue
        d.ellipse((96, y - 24, 144, y + 24), fill=col + (int(60 * a),), outline=col + (int(255 * a),), width=3)
        d.text((120, y), sym, font=font(FS["monob"], 30), fill=col + (int(255 * a),), anchor="mm")
        d.text((166, y), txt, font=font(FS["body"], 34), fill=WHITE + (int(255 * a),), anchor="lm")
    a_s = s.a(t, "separado")
    if a_s > 0:
        w, h = 400, 250; y0 = 830
        paste_img(lay, Image.fromarray(PIX).resize((w, h), Image.NEAREST), 100, y0, w, h, radius=12, alpha=a_s)
        u = ease((t - s.cue("separado") - 0.6) / 1.2); flat = PIX.reshape(-1, 3); mix = flat.copy()
        n_sw = int(u * len(flat)); mix[:n_sw] = flat[PERM[:n_sw]]
        paste_img(lay, Image.fromarray(mix.reshape(15, 24, 3)).resize((w, h), Image.NEAREST), 580, y0, w, h, radius=12, alpha=a_s); d = ImageDraw.Draw(lay, "RGBA")
        label(d, 300, y0 + h + 28, "la imagen", SOFT + (int(255 * a_s),), 24); label(d, 780, y0 + h + 28, "píxeles desordenados", SOFT + (int(255 * a_s),), 24)
        d.text((540, y0 + h / 2), "→", font=font(FS["head"], 50), fill=GOLD + (int(255 * a_s),), anchor="mm")
        if s.a(t, "vecinos") > 0:
            chip(d, CX, 790, "para un MLP, ambas son igual de fáciles", col=GOLD + (int(255 * s.a(t, "vecinos")),), size=26, anchor="mm", bg=(40, 32, 14, int(240 * s.a(t, "vecinos"))))

def next_(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    d.text((CX, 600), "PRÓXIMO EPISODIO", font=font(FS["monob"], 30), fill=SOFT, anchor="mm")
    d.text((CX, 680), T_("Redes convolucionales"), font=font(FS["head"], 62), fill=GOLD, anchor="mm")
    w, h = 480, 300; x0, y0 = CX - w / 2, 760; paste_img(lay, Image.fromarray(PIX).resize((w, h), Image.NEAREST), x0, y0, w, h, radius=12)
    d = ImageDraw.Draw(lay, "RGBA"); c = w / 24; ph = (t * 2.2) % 1; gx_ = int(ph * 21); gy_ = int(((t * 2.2) // 1) % 12)
    d.rectangle((x0 + gx_ * c, y0 + gy_ * c, x0 + (gx_ + 3) * c, y0 + (gy_ + 3) * c), outline=GOLD, width=5)
    label(d, CX, y0 + h + 34, "un filtro pequeño que recorre la imagen", SOFT, 26)
    a = s.a(t, "convolucionales")
    d.line((198, 360, 302, 360), fill=GOLD + (int(255 * a),), width=4); d.polygon([(302, 352), (302, 368), (316, 360)], fill=GOLD + (int(255 * a),))

def end(tl_, t):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    d.text((W / 2, 720), "REDES NEURONALES", font=font(FS["head"], 84), fill=GOLD, anchor="mm")
    d.text((W / 2, 810), "Ep. 1 · Perceptrón multicapa", font=font(FS["body"], 46), fill=WHITE, anchor="mm")
    d.rounded_rectangle((200, 900, 880, 1140), radius=24, fill=(16, 22, 32, 255), outline=(44, 58, 74, 255), width=2)
    d.text((W / 2, 960), "PRÓXIMO: REDES CONVOLUCIONALES", font=font(FS["monob"], 28), fill=GOLD, anchor="mm")
    d.text((W / 2, 1020), "código completo en la descripción", font=font(FS["body"], 30), fill=SOFT, anchor="mm")
    d.text((W / 2, 1090), HANDLE, font=font(FS["head"], 52), fill=WHITE, anchor="mm")
    return img

F_LIN = Formula([("p", r"$\hat{y}$"), ("eq", "$=$"), ("s", r"$\sigma(\mathbf{w}\cdot\mathbf{x} + b)$")], px=64)
def curve_loss(img, box, s, t):
    E_ = len(L); xs = np.arange(1, E_ + 1)[::5]; t0, t1 = s.cue("frontera"), s.cue("espiral")
    u = np.clip((t - t0) / max(0.5, t1 - t0), 0, 1) ** 1.2; k = KEYS[int(round(u * (len(KEYS) - 1)))]
    chart(img, box, [dict(xs=xs, ys=L[::5, 0], upto=k, col=BLUE, name="aprende", val_fmt=lambda v: dec(v, 3)), dict(xs=xs, ys=L[::5, 1], upto=k, col=GOLD, name="examen", val_fmt=lambda v: dec(v, 3))],
          "error durante el entrenamiento", xmax=E_, ylim=(0.008, 2), yticks=[0.01, 0.1, 1], yfmt=lambda v: dec(v, 2) if v < 1 else "1")
def curve_psnr(img, box, s, t):
    sp, sf = np.array(PSH["plain"]), np.array(PSH["fourier"]); its = np.array(META["keys"])
    a_o = t >= s.cue("ondas"); up = 1e9 if t >= s.cue("pinceladas") else (its[IDXH[int(np.clip((t - s.cue("ondas")) / max(0.8, s.cue("pinceladas") - s.cue("ondas")), 0, 1) ** 0.8 * (NJ - 1))]] if a_o else 0)
    chart(img, box, [dict(xs=sp[:, 0], ys=sp[:, 1], upto=1e9, col=WHITE, name="normal", val_fmt=lambda v: dec(v) + " dB"),
                     dict(xs=sf[:, 0], ys=sf[:, 1], upto=up, col=GOLD, name="Fourier", val_fmt=lambda v: dec(v) + " dB")],
          "calidad del cuadro (dB)", xmax=META["ITERS"], ylim=(10, 36), yticks=[15, 25, 35], logy=False, yfmt=lambda v: str(int(v)))

SCENES = dict(_extras=dict(code=CODE, code_title="mlp.py", formulas={"lineal": F_LIN}, curves={"perdida": curve_loss, "psnr": curve_psnr}), painting=painting, hook=hook, problem=problem, neuron=neuron, layers=layers, loss=loss, grad=grad, code=code, demo=demo, when=when, next=next_, _end=end)
if __name__ == "__main__":
    cli(tl, SCENES, "ep01_mlp_piloto")
