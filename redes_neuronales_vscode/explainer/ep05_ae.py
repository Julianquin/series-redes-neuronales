"""Episodio 05 · Autoencoders.  Uso: python3 ep05_ae.py info | prev f… | chunk 85 | encode"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw
from explainer import *
from reel_common import T_
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_ae_cuadros as TA
import reel_ae as RA                      # piezas del Reel 5b: mapa latente, paseo, reconstrucciones

HERE = os.path.dirname(os.path.abspath(__file__))
tl = Timeline(os.path.join(HERE, "guion_ep05_ae.yaml"))
CX = (PANEL[0] + PANEL[2]) / 2
RUN, KEYS, PF, LD = RA.RUN, RA.KEYS, RA.PF, RA.LD
patch, CCOL, NAMES = RA.patch, RA.CCOL, RA.NAMES

# vista suavizada del mapa latente por instantánea (para la escena de entrenamiento)
VIEW, cur = {}, None
for k in KEYS:
    Z = TA.encode(RUN["snaps"][k][2], RA.XS); lo, hi = np.percentile(Z, 1, 0), np.percentile(Z, 99, 0)
    b = np.r_[(lo + hi) / 2, (hi - lo).max() / 2 * 1.12]; cur = b if cur is None else 0.8 * cur + 0.2 * b; VIEW[k] = cur.copy()
# anomalías: error de reconstrucción (red de 32) de trozos normales frente a ruido puro
rng = np.random.default_rng(9); NOISE = rng.random((200, 768)).astype(np.float32)
err = lambda X: np.mean((TA.forward(PF[32], X)[-1] - X) ** 2, 1)
E_NORM, E_NOISE = err(TA.XVA), err(NOISE)

def label(d, x, y, txt, col=SOFT, size=26, anchor="mm", bold=False):
    d.text((x, y), txt, font=font(FS["monob" if bold else "mono"], size), fill=col, anchor=anchor)
def hl(s, t, name, base=WHITE, win=1.2):
    return (GOLD if 0 <= t - s.cue(name) < win else base, s.a(t, name, 0.35))
def chips_row(d, items, y, size=20):
    f = font(FS["monob"], size); ws = [f.getlength(txt) + 28 for _, txt, _ in items]; gap = 12; x = CX - (sum(ws) + gap * (len(ws) - 1)) / 2
    for (txt_a, txt, a), w in zip(items, ws):
        if a > 0: chip(d, x + w / 2, y, txt, col=GOLD + (int(255 * a),), size=size, anchor="mm", bg=(40, 32, 14, int(255 * a)))
        x += w + gap

# ---------- escenas ----------
def hook(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); sz = 300; y0 = 560
    paste_img(lay, patch(RA.HOOK_X, sz), 110, y0, sz, sz, radius=16); d = ImageDraw.Draw(lay, "RGBA")
    label(d, 260, y0 + sz + 28, "768 números", SOFT, 24)
    a_z = s.a(t, "dos")
    d.text((CX, y0 + sz / 2 - 30), "→", font=font(FS["head"], 46), fill=GOLD + (int(255 * a_z),), anchor="mm")
    d.text((CX, y0 + sz / 2 + 20), f"({dec(RA.HOOK_Z[0], 2)} ; {dec(RA.HOOK_Z[1], 2)})", font=font(FS["head"], 38), fill=GOLD + (int(255 * a_z),), anchor="mm")
    label(d, CX, y0 + sz / 2 + 66, "2 números", GOLD + (int(255 * a_z),), 22, bold=True)
    a_r = s.a(t, "reconstruye")
    if a_r > 0:
        paste_img(lay, patch(RA.HOOK_R, sz), 670, y0, sz, sz, radius=16, alpha=a_r, outline=GOLD); d = ImageDraw.Draw(lay, "RGBA")
        label(d, 820, y0 + sz + 28, "reconstruido", SOFT + (int(255 * a_r),), 24)
    a_i = s.a(t, "inventa")
    if a_i > 0:
        cx, cy, r = RA.BOUNDS[-1]; gr = np.array([[cx + u * r, cy + 0.2 * r] for u in np.linspace(-0.8, 0.8, 8)])
        for i, x in enumerate(TA.decode(PF[2], gr)): paste_img(lay, patch(x, 96, False), 110 + i * 109, 940, 96, 96, radius=8, alpha=a_i)
        d = ImageDraw.Draw(lay, "RGBA")
    chips_row(d, [(None, "CÓMO COMPRIME", s.a(t, "comprime")), (None, "FÓRMULAS", s.a(t, "formulas")), (None, "CÓDIGO", s.a(t, "codigo")),
                  (None, "QUÉ SE PIERDE", s.a(t, "pierde")), (None, "PINCELADAS NUEVAS", s.a(t, "inventa"))], 1112, 19)

SAMP = [c * 200 + 3 for c in range(3)]
def problem(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    for k, (cue, title, lab) in enumerate((("profesor", "CON PROFESOR", True), ("etiquetas", "SIN ETIQUETAS", False))):
        a = s.a(t, cue); x0 = 100 + k * 450
        if a <= 0: continue
        d.rounded_rectangle((x0, 560, x0 + 430, 960), radius=20, fill=(18, 26, 36, int(255 * a)), outline=(GREEN if lab else GOLD) + (int(255 * a),), width=3)
        label(d, x0 + 215, 600, title, (GREEN if lab else GOLD) + (int(255 * a),), 26, bold=True)
        for j, i in enumerate(SAMP):
            y = 650 + j * 98; paste_img(lay, patch(TA.XVA[i], 80), x0 + 40, y, 80, 80, radius=8, alpha=a); d = ImageDraw.Draw(lay, "RGBA")
            txt = f"→ {NAMES[TA.YVA[i]]}" if lab else "→ ?"
            d.text((x0 + 150, y + 40), txt, font=font(FS["monob"], 30), fill=(WHITE if lab else GOLD) + (int(255 * a),), anchor="lm")
    if s.a(t, "mirando") > 0: label(d, CX, 1040, "¿aprender solo mirando los datos?", GOLD + (int(255 * s.a(t, "mirando")),), 32, bold=True)

def idea(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); a_c = s.a(t, "copiar")
    paste_img(lay, patch(RA.HOOK_X, 120), 100, 700, 120, 120, radius=10, alpha=max(a_c, .3)); d = ImageDraw.Draw(lay, "RGBA")
    label(d, 160, 840, "entrada", SOFT, 22)
    bars = [768, 256, 64, 2, 64, 256, 768]; xs = np.linspace(300, 780, 7)
    for i, (k, x) in enumerate(zip(bars, xs)):
        hgt = 50 + 300 * (np.log(k) / np.log(768)); a = ease((t - s.cue("codificador") - i * 0.18) / 0.3) if i < 4 else ease((t - s.cue("decodificador") - (i - 3) * 0.18) / 0.3)
        a = max(a, 0.18); col = GOLD if k == 2 else BLUE; hot = k == 2 and s.a(t, "cuello") > 0
        d.rounded_rectangle((x - 24, 760 - hgt / 2, x + 24, 760 + hgt / 2), radius=16, fill=(GOLD if hot else col) + (int(220 * a),))
        label(d, x, 760 + hgt / 2 + 22, str(k), (GOLD if k == 2 else SOFT) + (int(255 * a),), 20, bold=True)
    if s.a(t, "decodificador") > 0:
        a = s.a(t, "decodificador"); paste_img(lay, patch(RA.HOOK_R, 120), 860, 700, 120, 120, radius=10, alpha=a); d = ImageDraw.Draw(lay, "RGBA")
        label(d, 920, 840, "copia", SOFT + (int(255 * a),), 22)
    label(d, 420, 1000, "codificador", BLUE + (int(255 * max(.3, s.a(t, "codificador"))),), 26, bold=True)
    label(d, 660, 1000, "decodificador", BLUE + (int(255 * max(.3, s.a(t, "decodificador"))),), 26, bold=True)
    if s.a(t, "cuello") > 0:
        ph = ((t - s.cue("cuello")) * 0.8) % 1; x = 300 + ph * 480; d.ellipse((x - 9, 751, x + 9, 769), fill=WHITE)
        label(d, 540, 560, "cuello de botella: 2 números", GOLD, 26, bold=True)
    if s.a(t, "esencial") > 0: chip(d, CX, 1090, "solo pasa lo esencial", col=GOLD, size=30, anchor="mm", bg=(40, 32, 14, int(255 * s.a(t, "esencial"))))

F_Z = Formula([("z", r"$z$"), ("eq", "$=$"), ("f", r"$f_\theta(x)$")], px=60)
F_X = Formula([("x", r"$\hat{x}$"), ("eq", "$=$"), ("g", r"$g_\phi(z)$")], px=60)
F_L = Formula([("L", r"$\mathcal{L}$"), ("eq", "$=$"), ("n", r"$\|\,x-\hat{x}\,\|^2$")], px=64)
def formulas(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); X = RA.HOOK_X; R = RA.HOOK_R; Er = np.abs(X - R).reshape(16, 16, 3).mean(-1); Er = Er / (Er.max() + 1e-9)
    items = [(110, X, "x", 1.0), (350, None, "z", s.a(t, "codificador")), (590, R, "x̂", s.a(t, "decodificador")), (830, None, "error", s.a(t, "perdida"))]
    for x0, img_, lab, a in items:
        if a <= 0: continue
        if lab == "z":
            d.rounded_rectangle((x0, 580, x0 + 140, 720), radius=14, fill=(40, 32, 14, int(255 * a)), outline=GOLD + (int(255 * a),), width=3)
            d.text((x0 + 70, 650), f"({dec(RA.HOOK_Z[0], 2)};\n{dec(RA.HOOK_Z[1], 2)})", font=font(FS["monob"], 24), fill=GOLD + (int(255 * a),), anchor="mm", align="center")
        elif lab == "error":
            em = Image.fromarray((np.stack([Er * 240, Er * 96, Er * 96], -1)).astype(np.uint8)).resize((140, 140), Image.NEAREST)
            paste_img(lay, em, x0, 580, 140, 140, radius=12, alpha=a); d = ImageDraw.Draw(lay, "RGBA")
        else:
            paste_img(lay, patch(img_, 140), x0, 580, 140, 140, radius=12, alpha=a); d = ImageDraw.Draw(lay, "RGBA")
        label(d, x0 + 70, 750, lab, WHITE + (int(255 * a),), 26, bold=True)
    for x in (270, 510, 750): d.text((x + 10, 650), "→", font=font(FS["head"], 40), fill=MUTED, anchor="mm")
    F_Z.draw(lay, 300, 850, {"z": hl(s, t, "codificador"), "eq": (WHITE, s.a(t, "codificador")), "f": hl(s, t, "codificador")})
    F_X.draw(lay, 780, 850, {"x": hl(s, t, "decodificador"), "eq": (WHITE, s.a(t, "decodificador")), "g": hl(s, t, "decodificador")})
    F_L.draw(lay, CX, 965, {"L": hl(s, t, "perdida"), "eq": (WHITE, s.a(t, "perdida")), "n": hl(s, t, "cuadratico", win=1.8)})
    label(d, 300, 905, "codificador", SOFT + (int(255 * s.a(t, "codificador")),), 20); label(d, 780, 905, "decodificador", SOFT + (int(255 * s.a(t, "decodificador")),), 20)
    if s.a(t, "respuesta") > 0: chip(d, CX, 1090, "la propia entrada es la respuesta: no hacen falta etiquetas", col=GOLD, size=24, anchor="mm", bg=(40, 32, 14, int(255 * s.a(t, "respuesta"))))

CODE = """def codificar(x):               # 768 -> 2
    h = relu(x @ W1 + b1)       # 256
    h = relu(h @ W2 + b2)       # 64
    return h @ W3 + b3          # 2, sin activación

def decodificar(z):             # 2 -> 768
    h = relu(z @ W4 + b4)       # 64
    h = relu(h @ W5 + b5)       # 256
    return sigmoid(h @ W6 + b6) # colores en [0, 1]

z = codificar(x)
x_rec = decodificar(z)
L = np.mean((x - x_rec) ** 2)   # error cuadrático"""
CB = CodeBlock(CODE, size=27, lh=1.42)
GROUPS = [("codificador", range(0, 4)), ("decodificador", range(5, 9)), ("perdida", range(10, 13))]
def code(lay, s, t):
    act, foc = None, 0.0
    for name, rng_ in GROUPS:
        if t >= s.cue(name) - 0.1: act, foc = set(rng_), ease((t - s.cue(GROUPS[0][0])) / 0.4)
    CB.draw(lay, 104, 560, act, foc)

def train(lay, s, t):
    t0 = s.cue("principio") - 0.4; t1 = s.cue("nunca") + 0.6
    u = np.clip((t - t0) / (t1 - t0), 0, 1) if t >= t0 else 0; k = KEYS[int(round(u ** 0.9 * (len(KEYS) - 1)))]; cx, cy, r = VIEW[k]
    RA.scatter(lay, (250, 540, 830, 900), TA.encode(RUN["snaps"][k][2], RA.XS), cx, cy, r, labels=u > 0.45, rad=3)
    d = ImageDraw.Draw(lay, "RGBA"); label(d, CX, 925, f"cada punto, un trozo · posición = sus 2 números · iteración {fmt(k)}", SOFT, 22)
    chart(lay, (90, 950, 990, 1170), [dict(xs=LD[2][:, 0], ys=LD[2][:, 1], upto=k, col=BLUE, name="2 núm.", val_fmt=lambda v: dec(v, 3)),
                                     dict(xs=LD[32][:, 0], ys=LD[32][:, 1], upto=k, col=GOLD, name="32 núm.", val_fmt=lambda v: dec(v, 3))],
          "error de reconstrucción", xmax=RA.ITERS, ylim=(0.008, 0.12), yticks=[0.01, 0.03, 0.1], yfmt=lambda v: dec(v, 2))

SEL5 = [c * 200 + 3 for c in range(5)]
def lost(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); X = TA.XVA[SEL5]
    rows = [("originales (nunca vistos)", X, WHITE, 1.0, None), ("con 2 números", TA.forward(PF[2], X)[-1], BLUE, s.a(t, "dos"), 2),
            ("con 32 números", TA.forward(PF[32], X)[-1], GOLD, s.a(t, "treinta"), 32)]
    ts, step = 150, 166; x0 = CX - (5 * step - 16) / 2
    for i, (lab, arr, col, a, k) in enumerate(rows):
        if a <= 0: continue
        y = 560 + i * 200; d.text((x0, y), T_(lab), font=font(FS["monob"], 26), fill=col + (int(255 * a),), anchor="lm")
        if k: d.text((x0 + 5 * step - 16, y), f"error {dec(LD[k][-1, 1], 3)}", font=font(FS["mono"], 24), fill=SOFT + (int(255 * a),), anchor="rm")
        for j in range(5): paste_img(lay, patch(arr[j], ts), x0 + j * step, y + 24, ts, ts, radius=12, alpha=a)
        d = ImageDraw.Draw(lay, "RGBA")
    if s.a(t, "color") > 0 and s.a(t, "treinta") <= 0: label(d, CX, 1160, "guarda el color y la luz… pero no la pincelada", BLUE, 24, bold=True)
    if s.a(t, "ancho") > 0: label(d, CX, 1160, "cuello más ancho = mejor copia (y menos compresión)", GOLD, 24, bold=True)

SEG = 2.4
def walk(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); t0 = s.cue("movernos")
    sp = min(len(RA.ROUTE) - 1.0001, max(0, (t - t0) / SEG)); a, u = int(sp), ease(sp - int(sp))
    z = RA.MED[RA.ROUTE[a]] + (RA.MED[RA.ROUTE[a + 1]] - RA.MED[RA.ROUTE[a]]) * u
    paste_img(lay, patch(TA.decode(PF[2], z[None])[0], 420), 110, 560, 420, 420, radius=16, outline=GOLD, owidth=5); d = ImageDraw.Draw(lay, "RGBA")
    chip(d, 320, 1010, f"de {NAMES[RA.ROUTE[a]]} a {NAMES[RA.ROUTE[a + 1]]}", col=WHITE, size=26, anchor="mm", bg=(30, 40, 54, 255))
    cx, cy, r = RA.BOUNDS[-1]; to = RA.scatter(lay, (580, 580, 980, 980), RA.ZF, cx, cy, r, alpha=120, rad=2); d = ImageDraw.Draw(lay, "RGBA")
    d.line([to(RA.MED[c]) for c in RA.ROUTE], fill=WHITE + (150,), width=3, joint="curve"); px, py = to(z); d.ellipse((px - 13, py - 13, px + 13, py + 13), outline=GOLD, width=5)
    if s.a(t, "mezclas") > 0: label(d, CX, 1090, "mezclas de color y luz: con 2 números no da para más", SOFT, 22)

def when(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    rows = [("comprimir", "✓", GREEN, "comprimir y quitar ruido"), ("anomalias", "✓", GREEN, "detectar anomalías"), ("limite", "✗", RED, "su espacio tiene huecos: genera mal")]
    for i, (cue, sym, col, txt) in enumerate(rows):
        a = s.a(t, cue); y = 580 + i * 76
        if a <= 0: continue
        d.ellipse((96, y - 24, 144, y + 24), fill=col + (int(60 * a),), outline=col + (int(255 * a),), width=3)
        d.text((120, y), sym, font=font(FS["monob"], 30), fill=col + (int(255 * a),), anchor="mm"); d.text((166, y), txt, font=font(FS["body"], 34), fill=WHITE + (int(255 * a),), anchor="lm")
    a = s.a(t, "raro")
    if a > 0:
        y0 = 860; paste_img(lay, patch(TA.XVA[3], 120), 130, y0, 120, 120, radius=10, alpha=a); paste_img(lay, patch(NOISE[0], 120), 130, y0 + 150, 120, 120, radius=10, alpha=a)
        d = ImageDraw.Draw(lay, "RGBA"); mx = float(np.median(E_NOISE)) * 1.15
        for k, (v, col, txt) in enumerate(((float(np.median(E_NORM)), GREEN, "trozo de un cuadro"), (float(np.median(E_NOISE)), RED, "ruido: algo raro"))):
            y = y0 + 60 + k * 150; w = 560 * min(1, v / mx)
            d.rounded_rectangle((290, y - 22, 290 + w, y + 22), radius=10, fill=col + (int(255 * a),))
            d.text((300 + w, y), f"  error {dec(v, 3)}", font=font(FS["monob"], 24), fill=col + (int(255 * a),), anchor="lm")
            label(d, 290, y - 40, txt, SOFT + (int(255 * a),), 22, anchor="lm")

def next_(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    d.text((CX, 590), "PRÓXIMO EPISODIO", font=font(FS["monob"], 30), fill=SOFT, anchor="mm")
    d.text((CX, 670), "GAN", font=font(FS["head"], 80), fill=GOLD, anchor="mm")
    for k, (x, title, sub, col) in enumerate(((110, "GENERADOR", "falsifica", GOLD), (570, "DISCRIMINADOR", "detecta", BLUE))):
        a = s.a(t, "falsifica") if k == 0 else ease((t - s.cue("falsifica") - 0.5) / 0.4)
        d.rounded_rectangle((x, 780, x + 400, 960), radius=22, fill=(18, 26, 36, int(255 * a)), outline=col + (int(255 * a),), width=4)
        d.text((x + 200, 845), title, font=font(FS["head"], 38), fill=col + (int(255 * a),), anchor="mm"); d.text((x + 200, 910), sub, font=font(FS["body"], 32), fill=SOFT + (int(255 * a),), anchor="mm")
    a = s.a(t, "gan"); d.line((640, 360, 676, 360), fill=GOLD + (int(255 * a),), width=4); d.polygon([(676, 352), (676, 368), (690, 360)], fill=GOLD + (int(255 * a),))

def end(tl_, t):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    d.text((W / 2, 720), "REDES NEURONALES", font=font(FS["head"], 84), fill=GOLD, anchor="mm")
    d.text((W / 2, 810), "Ep. 5 · Autoencoders", font=font(FS["body"], 46), fill=WHITE, anchor="mm")
    d.rounded_rectangle((200, 900, 880, 1140), radius=24, fill=(16, 22, 32, 255), outline=(44, 58, 74, 255), width=2)
    d.text((W / 2, 960), "PRÓXIMO: GAN", font=font(FS["monob"], 28), fill=GOLD, anchor="mm")
    d.text((W / 2, 1020), "código completo en la descripción", font=font(FS["body"], 30), fill=SOFT, anchor="mm")
    d.text((W / 2, 1090), HANDLE, font=font(FS["head"], 52), fill=WHITE, anchor="mm")
    return img

SCENES = dict(hook=hook, problem=problem, idea=idea, formulas=formulas, code=code, train=train, lost=lost, walk=walk, when=when, next=next_, _end=end)
if __name__ == "__main__":
    cli(tl, SCENES, "ep05_ae_piloto")
