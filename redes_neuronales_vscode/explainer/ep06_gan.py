"""Episodio 06 · GAN.  Uso: python3 ep06_gan.py info | prev f… | chunk 85 | encode"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from explainer import *
from reel_common import T_
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reel_gan as RG                      # piezas del Reel 6: mapa, sombreado del discriminador, datos de la GAN

HERE = os.path.dirname(os.path.abspath(__file__))
tl = Timeline(os.path.join(HERE, "guion_ep06_gan.yaml"))
CX = (PANEL[0] + PANEL[2]) / 2
KEYS, MET, RUN, ZF = RG.KEYS, RG.MET, RG.RUN, RG.ZF
ART = "cuadros_pd/jcjohnson_starry_night.jpg"
BLUR = Image.open(ART).convert("RGB").crop((300, 120, 560, 380)).resize((220, 220)).filter(ImageFilter.GaussianBlur(9))
COV = np.array([(k, MET[k][1]) for k in KEYS]); LAND = np.array([(k, MET[k][0]) for k in KEYS])
WORST = min(RG.COLLAPSE, key=lambda k: MET[k][1])                 # el peor colapso

def label(d, x, y, txt, col=SOFT, size=26, anchor="mm", bold=False):
    d.text((x, y), txt, font=font(FS["monob" if bold else "mono"], size), fill=col, anchor=anchor)
def hl(s, t, name, base=WHITE, win=1.2):
    return (GOLD if 0 <= t - s.cue(name) < win else base, s.a(t, name, 0.35))
def arrow(d, x0, y0, x1, y1, col, a=1.0, w=4):
    d.line((x0, y0, x1, y1), fill=col + (int(255 * a),), width=w); ang = math.atan2(y1 - y0, x1 - x0)
    d.polygon([(x1, y1), (x1 - 15 * math.cos(ang - .45), y1 - 15 * math.sin(ang - .45)), (x1 - 15 * math.cos(ang + .45), y1 - 15 * math.sin(ang + .45))], fill=col + (int(255 * a),))
def fakes(k): return RG.T.fwd(RUN["snaps"][k][0], ZF)[-1]

# ---------- escenas ----------
def hook(lay, s, t):
    RG.draw_map(lay, (160, 540, 760), fake=RG.FAKE_F, real_col=(40, 50, 62)); d = ImageDraw.Draw(lay, "RGBA")
    a = s.a(t, "existen"); chip(d, 180, 556, "2.500 pueblos inventados · en gris, los reales", col=WHITE + (int(255 * a),), size=22, bg=(8, 12, 18, int(220 * a)))
    items = [("compitiendo", "CÓMO COMPITE"), ("formulas", "FÓRMULAS"), ("codigo", "CÓDIGO"), ("atasca", "POR QUÉ SE ATASCA"), ("distinguir", "EL RETO")]
    f = font(FS["monob"], 19); ws = [f.getlength(x) + 28 for _, x in items]; x = CX - (sum(ws) + 12 * 4) / 2
    for (cue, txt), w in zip(items, ws):
        a = s.a(t, cue)
        if a > 0: chip(d, x + w / 2, 1140, txt, col=GOLD + (int(255 * a),), size=19, anchor="mm", bg=(40, 32, 14, int(255 * a)))
        x += w + 12

def problem(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); a = s.a(t, "borrosas")
    paste_img(lay, BLUR, 140, 600, 300, 300, radius=16, alpha=max(a, .4)); d = ImageDraw.Draw(lay, "RGBA")
    label(d, 290, 940, "autoencoder: copias borrosas", SOFT + (int(255 * max(a, .4)),), 24)
    if s.a(t, "inventar") > 0:
        aa = s.a(t, "inventar"); d.text((540, 750), "→", font=font(FS["head"], 60), fill=GOLD + (int(255 * aa),), anchor="mm")
    if s.a(t, "juzgue") > 0:
        aa = s.a(t, "juzgue"); d.ellipse((650, 620, 930, 900), fill=(18, 26, 36, int(255 * aa)), outline=BLUE + (int(255 * aa),), width=5)
        d.text((790, 730), "?", font=font(FS["head"], 120), fill=BLUE + (int(255 * aa),), anchor="mm")
        label(d, 790, 850, "¿es creíble?", BLUE + (int(255 * aa),), 28, bold=True); label(d, 790, 940, "un juez", SOFT + (int(255 * aa),), 24)

def idea(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    a_r = s.a(t, "ruido"); rng = np.random.default_rng(int(t * 3))
    for i in range(8):
        v = rng.normal(); ya, yb = sorted((650 - 40 * v, 650)); d.rectangle((110 + i * 20, ya, 124 + i * 20, yb), fill=SOFT + (int(255 * a_r),))
    label(d, 180, 720, "ruido z", SOFT + (int(255 * a_r),), 22)
    a_g = s.a(t, "generador")
    d.rounded_rectangle((300, 600, 450, 700), radius=18, fill=(40, 32, 14, int(255 * a_g)), outline=GOLD + (int(255 * a_g),), width=4)
    label(d, 375, 650, "G", GOLD + (int(255 * a_g),), 54, bold=True); label(d, 375, 728, "generador", GOLD + (int(255 * a_g),), 22)
    if a_r > 0: arrow(d, 270, 650, 296, 650, SOFT, a_r)
    if a_g > 0:
        arrow(d, 455, 650, 505, 650, GOLD, a_g); tmp = Image.new("RGBA", lay.size, (0, 0, 0, 0)); RG.draw_map(tmp, (520, 585, 220), fake=RG.FAKE_F, real=False, rf_=1.4); lay.alpha_composite(faded(tmp, a_g))
        d = ImageDraw.Draw(lay, "RGBA"); label(d, 630, 770, "falsos", GOLD + (int(255 * a_g),), 22, bold=True)
    a_re = s.a(t, "reales")
    if a_re > 0:
        tmp = Image.new("RGBA", lay.size, (0, 0, 0, 0)); RG.draw_map(tmp, (520, 830, 220), fake=None, real_col=(225, 230, 236), rr=1.0); lay.alpha_composite(faded(tmp, a_re))
        d = ImageDraw.Draw(lay, "RGBA"); label(d, 630, 1015, "reales", WHITE + (int(255 * a_re),), 22, bold=True)
    a_d = s.a(t, "discriminador")
    if a_d > 0:
        d.rounded_rectangle((820, 760, 970, 860), radius=18, fill=(14, 26, 44, int(255 * a_d)), outline=BLUE + (int(255 * a_d),), width=4)
        label(d, 895, 810, "D", BLUE + (int(255 * a_d),), 54, bold=True); label(d, 895, 888, "discriminador", BLUE + (int(255 * a_d),), 20)
        arrow(d, 750, 680, 815, 780, GOLD, a_d, 3); arrow(d, 750, 920, 815, 840, WHITE, a_re, 3); label(d, 895, 930, "¿real o falso?", SOFT + (int(255 * a_d),), 22)
    a_a = s.a(t, "aprende")
    if a_a > 0:
        ph = (t - s.cue("aprende")) * 1.2; pts = [(895 - u * 520, 1060 + 40 * math.sin(math.pi * u)) for u in np.linspace(0, 1, 30)]
        d.line(pts, fill=GOLD + (int(255 * a_a),), width=4); arrow(d, pts[-2][0], pts[-2][1], 375, 715, GOLD, a_a)
        k = int((ph % 1) * 29); d.ellipse((pts[k][0] - 8, pts[k][1] - 8, pts[k][0] + 8, pts[k][1] + 8), fill=WHITE)
        label(d, CX, 1130, "cada uno aprende de los errores del otro", GOLD + (int(255 * a_a),), 26, bold=True)

F_MM = Formula([("mm", r"$\min_G\,\max_D$"), ("r", r"$\ \mathbb{E}_x[\log D(x)]$"), ("p", "$+$"), ("f", r"$\mathbb{E}_z[\log(1-D(G(z)))]$")], px=48)
F_NS = Formula([("mx", r"$\max_G$"), ("g", r"$\ \mathbb{E}_z[\log D(G(z))]$")], px=52)
def formula(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    F_MM.draw(lay, CX, 620, {"mm": (GOLD if 0 <= t - s.cue("juego") < 1.5 or 0 <= t - s.cue("minimizarlo") < 1.5 else WHITE, s.a(t, "juego")),
                             "r": hl(s, t, "reales"), "p": (WHITE, s.a(t, "reales")), "f": hl(s, t, "falsos")}, max_w=920)
    a_m = s.a(t, "maximizar")
    if a_m > 0:
        for k, (txt, v, col) in enumerate((("D(pueblo real)", 0.92, GREEN), ("D(pueblo falso)", 0.08, RED))):
            y = 740 + k * 70; d.text((120, y), txt, font=font(FS["monob"], 26), fill=WHITE + (int(255 * a_m),), anchor="lm")
            d.rounded_rectangle((420, y - 18, 420 + 460 * v, y + 18), radius=8, fill=col + (int(255 * a_m),)); label(d, 430 + 460 * v + 40, y, dec(v, 2), col + (int(255 * a_m),), 24, bold=True)
        label(d, CX, 880, "el discriminador quiere: reales → 1, falsos → 0", BLUE + (int(255 * a_m),), 24, bold=True)
    if s.a(t, "minimizarlo") > 0: label(d, CX, 930, "el generador quiere lo contrario", GOLD + (int(255 * s.a(t, "minimizarlo")),), 24, bold=True)
    a_p = s.a(t, "practica")
    if a_p > 0:
        F_NS.draw(lay, CX, 1030, {"mx": (GOLD, a_p), "g": (GOLD, a_p)})
        label(d, CX, 1110, "versión «no saturante»: da gradientes útiles aunque D gane al principio", SOFT + (int(255 * a_p),), 21)

CODE = """for paso in range(8000):
    # 1. entrena al detective (discriminador)
    reales = muestra(pueblos, 512)
    falsos = G(np.random.randn(512, 8))
    p_r, p_f = D(reales), D(falsos)
    loss_D = -np.mean(np.log(p_r) + np.log(1 - p_f))
    actualizar(D, gradiente(loss_D))

    # 2. entrena al falsificador (generador)
    falsos = G(np.random.randn(512, 8))
    loss_G = -np.mean(np.log(D(falsos)))
    actualizar(G, gradiente(loss_G))  # cruza D"""
CB = CodeBlock(CODE, size=26, lh=1.42)
GROUPS = [("primero", range(1, 7)), ("despues", range(8, 12)), ("gradiente", {11})]
def code(lay, s, t):
    act, foc = None, 0.0
    for name, rng_ in GROUPS:
        if t >= s.cue(name) - 0.1: act, foc = set(rng_), ease((t - s.cue(GROUPS[0][0])) / 0.4)
    CB.draw(lay, 100, 560, act, foc)
    label(ImageDraw.Draw(lay, "RGBA"), CX, 1150, "resumido: en el script, la retropropagación está escrita a mano", MUTED, 20)

MBOX = (220, 520, 560)
_ds = {}
def dsh(k):
    if k not in _ds:
        if len(_ds) > 3: _ds.clear()
        _ds[k] = RG.dshade(RUN["snaps"][k][1], MBOX)
    return _ds[k]
def train(lay, s, t):
    t0 = s.cue("municipios"); u = np.clip((t - t0) / max(1.0, s.dur - 0.8 - t0), 0, 1) if t >= t0 else 0
    k = KEYS[int(round(u ** 1.15 * (len(KEYS) - 1)))]
    RG.draw_map(lay, MBOX, fake=fakes(k), dmap=dsh(k) if s.a(t, "fondo") > 0 else None, rf_=2.2); d = ImageDraw.Draw(lay, "RGBA")
    chip(d, 240, 536, f"iteración {fmt(k)}", col=WHITE, size=22); chip(d, 760, 536, f"{dec(MET[k][0] * 100, 0)} % en tierra", col=GOLD, size=22, anchor="rt")
    xs = np.arange(1, len(RG.L) + 1)[::10]
    chart(lay, (90, 960, 990, 1170), [dict(xs=xs, ys=RG.L[::10, 1], upto=k, col=GOLD, name="falsificador", val_fmt=lambda v: dec(v, 2)),
                                     dict(xs=xs, ys=RG.L[::10, 0], upto=k, col=BLUE, name="detective", val_fmt=lambda v: dec(v, 2))],
          "pérdida de cada red", xmax=RG.ITERS, ylim=(0.5, 1.4), yticks=[0.6, 0.9, 1.2], logy=False, yfmt=lambda v: dec(v, 1))
    if s.a(t, "fondo") > 0:
        a = s.a(t, "fondo"); chip(d, CX, 905, "fondo azul: «parece real» · rojo: «parece falso»", col=SOFT + (int(255 * a),), size=20, anchor="mm", bold=False, bg=(8, 12, 18, int(220 * a)))

def collapse(lay, s, t):
    i0 = KEYS.index(WORST); t_z, t_r = s.cue("zona"), s.cue("recupera")
    if t < t_z: k = KEYS[max(0, i0 - 8)]
    elif t < t_z + 1.5: k = KEYS[max(0, i0 - 8) + int((t - t_z) / 1.5 * 8)]
    elif t < t_r: k = WORST
    else: k = KEYS[-1]
    RG.draw_map(lay, MBOX, fake=fakes(k), rf_=2.2); d = ImageDraw.Draw(lay, "RGBA")
    chip(d, 240, 536, f"iteración {fmt(k)}", col=WHITE, size=22)
    bad = MET[k][1] < 0.4
    chip(d, 760, 536, f"cubre el {dec(MET[k][1] * 100, 0)} % del mapa", col=RED if bad else GOLD, size=22, anchor="rt")
    if s.a(t, "colapso") > 0 and k == WORST: chip(d, CX, 900, "colapso de modos: todos los puntos en una zona", col=RED, size=24, anchor="mm", bg=(40, 14, 18, 240))
    a_c = s.a(t, "caida")
    chart(lay, (90, 960, 990, 1170), [dict(xs=COV[:, 0], ys=COV[:, 1] * 100, upto=k if a_c <= 0 else 1e9, col=GOLD, name="cobertura", val_fmt=lambda v: dec(v, 0) + " %")],
          "% del mapa cubierto por los falsos", xmax=RG.ITERS, ylim=(10, 100), yticks=[25, 50, 75, 100], logy=False, yfmt=lambda v: f"{int(v)}")
    if a_c > 0:
        d = ImageDraw.Draw(lay, "RGBA"); x0, x1, y0, y1 = 182, 954, 1052, 1112
        for kk in RG.COLLAPSE:
            x = x0 + np.log10(max(kk, 1)) / np.log10(RG.ITERS) * (x1 - x0); y = y1 - (MET[kk][1] * 100 - 10) / 90 * (y1 - y0)
            d.ellipse((x - 9, y - 9, x + 9, y + 9), outline=RED + (int(255 * a_c),), width=4)

def challenge(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); rev = t >= s.cue("arriba")
    for i, (pts, lab, ok) in enumerate(((RG.FAKE_F, "A", False), (RG.REAL_S, "B", True))):
        x0 = 90 + i * 480; col = (GREEN if ok else GOLD) if rev else WHITE
        h = RG.draw_map(lay, (x0, 600, 420), fake=pts, real=False, fake_col=(236, 236, 228), rf_=1.5); d = ImageDraw.Draw(lay, "RGBA")
        d.ellipse((x0 + 170, 560 - 32 + 400, x0 + 250, 560 + 48 + 400), fill=(22, 30, 40), outline=col, width=4)
        label(d, x0 + 210, 968, lab, col, 44, bold=True)
        if rev: chip(d, x0 + 210, 1050, "REAL" if ok else "FALSO · la red", col=col, size=26, anchor="mm", bg=(8, 12, 18, 235))
    if not rev:
        k = min(int((t - s.cue("distingues")) // 1.0), 2) if t >= s.cue("distingues") else 0
        if t >= s.cue("distingues"):
            p = ((t - s.cue("distingues")) % 1.0); d.ellipse((CX - 46, 1070, CX + 46, 1162), outline=(40, 52, 66), width=8)
            d.arc((CX - 46, 1070, CX + 46, 1162), start=-90, end=-90 + 360 * p, fill=GOLD, width=8); label(d, CX, 1116, str(3 - k), WHITE, 44, bold=True)
    if s.a(t, "mar") > 0: label(d, CX, 1120, "se sale al mar y a Portugal: no sabe dónde acaba España", GOLD, 24, bold=True)

def when(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    rows = [("reinas", "✓", GREEN, "caras que no existen, superresolución"), ("inestables", "✗", RED, "inestables y difíciles de entrenar"),
            ("difusion", "→", GOLD, "hoy dominan los modelos de difusión")]
    for i, (cue, sym, col, txt) in enumerate(rows):
        a = s.a(t, cue); y = 600 + i * 90
        if a <= 0: continue
        d.ellipse((96, y - 26, 148, y + 26), fill=col + (int(60 * a),), outline=col + (int(255 * a),), width=3)
        d.text((122, y), sym, font=font(FS["monob"], 30), fill=col + (int(255 * a),), anchor="mm"); d.text((170, y), txt, font=font(FS["body"], 34), fill=WHITE + (int(255 * a),), anchor="lm")

def next_(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    d.text((CX, 590), "PRÓXIMO EPISODIO", font=font(FS["monob"], 30), fill=SOFT, anchor="mm")
    d.text((CX, 670), T_("Redes de grafos"), font=font(FS["head"], 70), fill=GOLD, anchor="mm")
    rng = np.random.default_rng(2); P_ = rng.uniform([200, 760], [880, 1080], (14, 2)); E_ = [(i, j) for i in range(14) for j in range(i + 1, 14) if np.hypot(*(P_[i] - P_[j])) < 210]
    for i, j in E_: d.line((tuple(P_[i]), tuple(P_[j])), fill=(110, 124, 140), width=3)
    ph = (t * 0.8) % 1
    for i, j in E_[:6]:
        x, y = P_[i] + (P_[j] - P_[i]) * ph; d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=GOLD)
    for x, y in P_: d.ellipse((x - 16, y - 16, x + 16, y + 16), fill=(26, 34, 44), outline=GOLD, width=4)
    label(d, CX, 1130, "cuando los datos son conexiones", SOFT, 28)
    a = s.a(t, "grafos"); d.line((758, 360, 794, 360), fill=GOLD + (int(255 * a),), width=4); d.polygon([(794, 352), (794, 368), (808, 360)], fill=GOLD + (int(255 * a),))

def end(tl_, t):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    d.text((W / 2, 720), "REDES NEURONALES", font=font(FS["head"], 84), fill=GOLD, anchor="mm")
    d.text((W / 2, 810), "Ep. 6 · GAN", font=font(FS["body"], 46), fill=WHITE, anchor="mm")
    d.rounded_rectangle((200, 900, 880, 1140), radius=24, fill=(16, 22, 32, 255), outline=(44, 58, 74, 255), width=2)
    d.text((W / 2, 960), "PRÓXIMO: REDES DE GRAFOS", font=font(FS["monob"], 28), fill=GOLD, anchor="mm")
    d.text((W / 2, 1020), "código completo en la descripción", font=font(FS["body"], 30), fill=SOFT, anchor="mm")
    d.text((W / 2, 1090), HANDLE, font=font(FS["head"], 52), fill=WHITE, anchor="mm")
    return img

SCENES = dict(hook=hook, problem=problem, idea=idea, formula=formula, code=code, train=train, collapse=collapse, challenge=challenge, when=when, next=next_, _end=end)
if __name__ == "__main__":
    cli(tl, SCENES, "ep06_gan_piloto")
