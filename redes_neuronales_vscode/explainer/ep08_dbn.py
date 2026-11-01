"""Episodio 08 · Redes de creencias profundas (DBN) · cierre de temporada.  Uso: python3 ep08_dbn.py info | prev f… | chunk 85 | encode"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw
from explainer import *
from reel_common import T_
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reel_ola as RO                     # piezas del Reel 8b: grabado, sueños, restauración
T = RO.T

HERE = os.path.dirname(os.path.abspath(__file__))
tl = Timeline(os.path.join(HERE, "guion_ep08_dbn.yaml"))
CX = (PANEL[0] + PANEL[2]) / 2
RUN, S1, P = RO.RUN, RO.S1, T.PS
W1, BV, CH = RUN["W1"], RUN["b1"], RUN["c1"]          # pesos, sesgo visible, sesgo oculto
sig = T.sig
ink = RO.ink

# energía libre real: F(v) = -a·v - Σ log(1 + e^{c + vW})
def free_energy(V): return -(V @ BV) - np.log1p(np.exp(np.clip(V @ W1 + CH, -30, 30))).sum(1)
rng = np.random.default_rng(3); XT = T.XTE[:200]
NOISE = (rng.random(XT.shape) < XT.mean()).astype(np.float32)
F_REAL, F_NOISE = float(np.median(free_energy(XT))), float(np.median(free_energy(NOISE)))
EXP = XT[np.argmax(XT.std(1))]                          # trocito de ejemplo, con tinta
# cadena de sueño (Gibbs) a partir del trocito real, con el modelo final
def chain(v, n=6, seed=1):
    r = np.random.default_rng(seed); out = [v.copy()]; hs = []
    for _ in range(n):
        ph = sig(v @ W1 + CH); h = (r.random(ph.shape) < ph).astype(np.float32); hs.append(ph)
        pv = sig(h @ W1.T + BV); v = (r.random(pv.shape) < pv).astype(np.float32); out.append(pv)
    return out, hs
CHAIN, HS = chain(EXP[None])

def label(d, x, y, txt, col=SOFT, size=26, anchor="mm", bold=False):
    d.text((x, y), txt, font=font(FS["monob" if bold else "mono"], size), fill=col, anchor=anchor)
def hl(s, t, name, base=WHITE, win=1.2):
    return (GOLD if 0 <= t - s.cue(name) < win else base, s.a(t, name, 0.35))
def signed(w, size):
    v = np.clip(w / (np.abs(w).max() + 1e-9), -1, 1).reshape(P, P)[..., None]; base = np.array((18, 26, 35), np.float32)
    rgb = np.where(v > 0, base + (np.array(GOLD) - base) * v, base + (np.array(BLUE) - base) * (-v))
    return Image.fromarray(rgb.astype(np.uint8)).resize((size, size), Image.NEAREST)

# ---------- escenas ----------
def hook(lay, s, t):
    gap = 40; ch = 560; cw = int(ch * RO.RW / RO.RH); x0 = (W - 2 * cw - gap) // 2; y = 540
    for k, (arr, lab, col) in enumerate(((RO.DMG, "antes", WHITE), (RO.HIST[-1], "después", GOLD))):
        x = x0 + k * (cw + gap); paste_img(lay, ink(arr, cw, ch), x, y, cw, ch, radius=14, outline=GOLD if k else None)
        chip(ImageDraw.Draw(lay, "RGBA"), x + cw / 2, y + 30, T_(lab), col=col, size=26, anchor="mm")
    d = ImageDraw.Draw(lay, "RGBA")
    items = [("sonando", "SOÑAR"), ("formulas", "FÓRMULAS"), ("codigo", "CÓDIGO"), ("apilan", "APILAR"), ("resucitaron", "HISTORIA")]
    f = font(FS["monob"], 20); ws = [f.getlength(x) + 28 for _, x in items]; x = CX - (sum(ws) + 12 * 4) / 2
    for (cue, txt), w in zip(items, ws):
        a = s.a(t, cue)
        if a > 0: chip(d, x + w / 2, 1140, txt, col=GOLD + (int(255 * a),), size=20, anchor="mm", bg=(40, 32, 14, int(255 * a)))
        x += w + 12

def rbm(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); a_v, a_o, a_s, a_d = s.a(t, "visible"), s.a(t, "oculta"), s.a(t, "sentidos"), s.a(t, "dentro")
    vis = [(170 + i * 105, 1000) for i in range(8)]; hid = [(220 + i * 128, 660) for i in range(6)]
    for p in vis:
        for q in hid: d.line((p, q), fill=(110, 124, 140, int(160 * min(a_v, a_o))), width=2)
    if a_s > 0:
        ph = ((t - s.cue("sentidos")) * 0.8) % 1; up = ph < 0.5; u = ph * 2 if up else (ph - 0.5) * 2
        for p in vis[::2]:
            for q in hid[::2]:
                a_, b_ = (p, q) if up else (q, p); x, y = a_[0] + (b_[0] - a_[0]) * u, a_[1] + (b_[1] - a_[1]) * u
                d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=GOLD + (int(220 * a_s),))
    for k, (x, y) in enumerate(vis):
        d.rectangle((x - 26, y - 26, x + 26, y + 26), fill=((22, 26, 34) if k % 3 else (240, 234, 220)) + (int(255 * max(a_v, .3)),), outline=WHITE + (int(255 * max(a_v, .3)),), width=2)
    for (x, y) in hid: d.ellipse((x - 28, y - 28, x + 28, y + 28), fill=(26, 34, 44, int(255 * max(a_o, .3))), outline=GOLD + (int(255 * max(a_o, .3)),), width=4)
    label(d, CX, 1068, "visible: los píxeles (400)", WHITE + (int(255 * max(a_v, .3)),), 24, bold=True)
    label(d, CX, 598, "oculta: detectores de patrones (300)", GOLD + (int(255 * max(a_o, .3)),), 24, bold=True)
    if a_d > 0:
        x0, x1 = vis[2][0], vis[3][0]; d.arc((x0, 1030, x1, 1110), 0, 180, fill=RED + (int(255 * a_d),), width=4)
        d.line(((x0 + x1) / 2 - 16, 1094, (x0 + x1) / 2 + 16, 1126), fill=RED + (int(255 * a_d),), width=5); d.line(((x0 + x1) / 2 - 16, 1126, (x0 + x1) / 2 + 16, 1094), fill=RED + (int(255 * a_d),), width=5)
        label(d, 760, 1130, "nada dentro de una capa", RED + (int(255 * a_d),), 22, bold=True)

F_E = Formula([("E", r"$E(v,h)$"), ("eq", "$=$"), ("a", r"$-a^{\top}v$"), ("b", r"$-\,b^{\top}h$"), ("w", r"$-\,v^{\top}W h$")], px=58)
F_P = Formula([("p", r"$p(v,h)$"), ("pr", r"$\propto$"), ("e", r"$e^{-E(v,h)}$")], px=54)
def energy(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    F_E.draw(lay, CX, 590, {"E": hl(s, t, "energia"), "eq": (WHITE, s.a(t, "energia")), "a": (WHITE, s.a(t, "energia")), "b": (WHITE, s.a(t, "energia")), "w": hl(s, t, "energia", win=1.8)})
    F_P.draw(lay, CX, 680, {"p": (WHITE, s.a(t, "energia")), "pr": (WHITE, s.a(t, "energia")), "e": hl(s, t, "baja")})
    label(d, CX, 735, "menos energía = más probable", SOFT + (int(255 * s.a(t, "energia")),), 22)
    lo, hi = F_REAL, F_NOISE; mn = min(lo, hi) * 1.15; rng_ = max(hi, lo) - mn
    for k, (cue, arr, v, col, txt) in enumerate((("baja", EXP, lo, GREEN, "trozo del grabado"), ("alta", NOISE[0], hi, RED, "ruido"))):
        a = s.a(t, cue); y = 800 + k * 170
        if a <= 0: continue
        paste_img(lay, ink(arr.reshape(P, P), 130, 130), 110, y, 130, 130, radius=10, alpha=a); d = ImageDraw.Draw(lay, "RGBA")
        w_ = 580 * (v - mn) / (rng_ + 1e-9); d.rounded_rectangle((280, y + 50, 280 + w_, y + 90), radius=8, fill=col + (int(255 * a),))
        label(d, 280, y + 28, f"{txt} · energía {dec(v, 0)}", col + (int(255 * a),), 22, anchor="lm", bold=True)

def dream(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA"); n_vis = int(np.clip((t - s.cue("reconstruye")) * 2.2 + 1, 1, len(CHAIN))) if t >= s.cue("reconstruye") else (1 if t >= s.cue("real") else 0)
    sz, gap = 130, 26; x0 = CX - (4 * sz + 3 * gap) / 2
    for k in range(min(n_vis, 4)):
        idx = k if n_vis <= 4 else n_vis - 4 + k; x = x0 + k * (sz + gap)
        paste_img(lay, ink(CHAIN[idx].reshape(P, P), sz, sz), x, 600, sz, sz, radius=10, outline=GOLD if idx == 0 else None); d = ImageDraw.Draw(lay, "RGBA")
        label(d, x + sz / 2, 760, "real" if idx == 0 else f"sueño {idx}", GOLD if idx == 0 else SOFT, 22, bold=idx == 0)
        if k < 3 and idx + 1 < n_vis: d.text((x + sz + gap / 2, 665), "→", font=font(FS["head"], 30), fill=MUTED, anchor="mm")
    if n_vis >= 2:
        h = HS[min(n_vis - 2, len(HS) - 1)][0][:120]; bw = 7
        for j, v in enumerate(h): d.rectangle((CX - 60 * bw / 2 + (j % 60) * bw, 820 + (j // 60) * 16, CX - 60 * bw / 2 + (j % 60) * bw + bw - 1, 834 + (j // 60) * 16), fill=tuple(int(18 + (g - 18) * v) for g in GOLD))
        label(d, CX, 872, "neuronas ocultas activadas (120 de 300)", SOFT, 20)
    a_c = s.a(t, "compara")
    if a_c > 0:
        chip(d, 330, 960, "lo que ve: datos", col=GREEN + (int(255 * a_c),), size=24, anchor="mm", bg=(14, 30, 22, int(240 * a_c)))
        chip(d, 750, 960, "lo que sueña: su modelo", col=GOLD + (int(255 * a_c),), size=24, anchor="mm", bg=(40, 32, 14, int(240 * a_c)))
    if s.a(t, "ajusta") > 0: label(d, CX, 1060, "ajusta los pesos para que el sueño se parezca a los datos", GOLD + (int(255 * s.a(t, "ajusta")),), 24, bold=True)

F_CD = Formula([("d", r"$\Delta W$"), ("eq", "$=$"), ("eta", r"$\eta\,($"), ("data", r"$\langle v\,h^{\top}\rangle_{\mathrm{datos}}$"), ("mi", "$-$"), ("dr", r"$\langle v\,h^{\top}\rangle_{\mathrm{sueño}}$"), ("c", "$)$")], px=56)
VS = XT[:64]; PH_D = sig(VS @ W1 + CH); VD = np.array([chain(v[None], 1, seed=i)[0][-1][0] for i, v in enumerate(VS)]); PH_S = sig(VD @ W1 + CH)
C_D, C_S = (VS.T @ PH_D / len(VS))[:12, :12], (VD.T @ PH_S / len(VS))[:12, :12]
def heat(M, size, signed_=False):
    m = np.abs(M).max() + 1e-9; v = M / m
    rgb = np.where(v[..., None] > 0, np.array(GOLD) * v[..., None], np.array(BLUE) * (-v[..., None])) if signed_ else np.array(GOLD) * v[..., None]
    return Image.fromarray(np.clip(rgb + 14, 0, 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
def formula(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    F_CD.draw(lay, CX, 600, {"d": hl(s, t, "cambio"), "eq": (WHITE, s.a(t, "cambio")), "eta": (WHITE, s.a(t, "cambio")), "c": (WHITE, s.a(t, "cambio")),
                             "data": hl(s, t, "reales", base=GREEN), "mi": (WHITE, s.a(t, "menos")), "dr": hl(s, t, "suenos", base=GOLD)}, max_w=920)
    for k, (cue, M, lab, col, sg_) in enumerate((("reales", C_D, "con datos", GREEN, False), ("suenos", C_S, "en sueños", GOLD, False), ("suenos", C_D - C_S, "= cambio", WHITE, True))):
        a = s.a(t, cue) if k < 2 else ease((t - s.cue("suenos") - 0.6) / 0.4)
        if a <= 0: continue
        x = 130 + k * 300; paste_img(lay, heat(M, 220, sg_), x, 720, 220, 220, radius=10, alpha=a); d = ImageDraw.Draw(lay, "RGBA")
        label(d, x + 110, 965, lab, col + (int(255 * a),), 24, bold=True)
        if k < 2: d.text((x + 260, 830), "−" if k == 0 else "→", font=font(FS["head"], 46), fill=MUTED + (int(255 * a),), anchor="mm")
    label(d, CX, 1040, "correlación píxel-neurona (12 × 12 de 400 × 300)", MUTED, 20)

CODE = """def paso_rbm(v0, W, a, b, lr):
    ph0 = sigmoid(v0 @ W + b)            # lo que ve
    h0  = ph0 > np.random.rand(*ph0.shape)  # muestreo
    pv1 = sigmoid(h0 @ W.T + a)          # sueña píxeles
    ph1 = sigmoid(pv1 @ W + b)
    W += lr * (v0.T @ ph0 - pv1.T @ ph1) / len(v0)
    a += lr * (v0 - pv1).mean(0)
    b += lr * (ph0 - ph1).mean(0)"""
CB = CodeBlock(CODE, size=24, lh=1.6)
GROUPS = [("probabilidades", {1}), ("muestreo", {2}), ("reconstruccion", {3, 4}), ("diferencia", {5, 6, 7})]
def code(lay, s, t):
    act, foc = None, 0.0
    for name, rng_ in GROUPS:
        if t >= s.cue(name) - 0.1: act, foc = set(rng_), ease((t - s.cue(GROUPS[0][0])) / 0.4)
    CB.draw(lay, 96, 600, act, foc)
    label(ImageDraw.Draw(lay, "RGBA"), CX, 1120, "versión CD-1; el script usa cadenas de sueño persistentes (PCD)", MUTED, 20)

def stack(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    boxes = [("píxeles · 400", MUTED, 1080, 1.0), ("capa 1 · 300", BLUE, 880, max(.3, s.a(t, "entrenada"))), ("capa 2 · 120", GOLD, 680, s.a(t, "otra"))]
    for i, (lab, col, y, a) in enumerate(boxes):
        if a <= 0: continue
        d.rounded_rectangle((110, y - 46, 470, y + 46), radius=18, fill=(18, 26, 36, int(255 * a)), outline=col + (int(255 * a),), width=4)
        label(d, 290, y, T_(lab), col + (int(255 * a),), 28, bold=True)
        if i > 0: d.line((290, y + 48, 290, y + 152), fill=col + (int(255 * a),), width=5)
    for k, (cue, lab, y0) in enumerate((("entrenada", "patrones (capa 1)", 910), ("patrones", "patrones de patrones (capa 2)", 650))):
        a = s.a(t, cue)
        if a <= 0: continue
        filt = [W1[:, j] for j in range(8)] if k == 0 else [W1 @ RUN["S2"][-1][1][:, j] for j in range(8)]
        for j, w in enumerate(filt): paste_img(lay, signed(w, 98), 510 + (j % 4) * 120, y0 - 100 + (j // 4) * 110, 98, 98, radius=6, alpha=a)
        d = ImageDraw.Draw(lay, "RGBA"); label(d, 750, y0 + 122, lab, (BLUE if k == 0 else GOLD) + (int(255 * a),), 22, bold=True)
    if s.a(t, "profunda") > 0: chip(d, CX, 524, "red de creencias profunda (DBN)", col=GOLD, size=28, anchor="mm", bg=(40, 32, 14, int(240 * s.a(t, "profunda"))))

STEPS, CYC = 30, 40
DREAMS = {}
def learn(lay, s, t):
    n = int(s.dur * FPS) + 2
    if not DREAMS:
        dr = RO.Dreamer()
        for f in range(n):
            u = min(1.0, f / (n - 20)); si = int(round(u ** 1.5 * (len(S1) - 1))); _, Wt, b, c = S1[si]; ph = f % CYC
            if ph == 0: dr.reset()
            elif ph <= STEPS: dr.step(Wt, b, c)
            DREAMS[f] = ((dr.pv if ph > 0 else dr.v).copy(), S1[si][0])
    f = min(int(t * FPS), n - 1); dv, stp = DREAMS[f]; d = ImageDraw.Draw(lay, "RGBA")
    ts_, step = 170, 186; x0 = CX - (4 * step - 16) / 2
    for j in range(8):
        r_, c_ = divmod(j, 4); paste_img(lay, ink(dv[j].reshape(P, P), ts_, ts_), x0 + c_ * step, 560 + r_ * step, ts_, ts_, radius=12)
    d = ImageDraw.Draw(lay, "RGBA"); label(d, CX, 945, f"sueños desde ruido · paso {fmt(stp)} de {fmt(len(RO.E1))}", SOFT, 22)
    xs = np.arange(1, len(RO.E1s) + 1)[::25]; ys = RO.E1s[::25]; lo, hi = float(RO.E1s[100:].min()), float(RO.E1s.max())
    chart(lay, (90, 970, 990, 1170), [dict(xs=xs, ys=ys, upto=stp, col=GOLD, name="error")], "error de reconstrucción", xmax=len(RO.E1s),
          ylim=(lo * 0.9, hi * 1.05), yticks=[0.1, 0.15, 0.2], logy=False, xlabel="paso", yfmt=lambda v: dec(v, 2))

def restore(lay, s, t):
    on = t >= s.cue("rompemos"); k = int(np.clip((t - s.cue("rompemos") - 0.6) // 0.9 + 1, 0, 6)) if on else 0; cur = RO.DMG if k == 0 else RO.HIST[k - 1]
    gap = 24; ch = 430; cw = int(ch * RO.RW / RO.RH); x0 = 100; y = 560
    for kk, (arr, lab, col) in enumerate(((RO.DMG if on else RO.REG, "dañado" if on else "nunca vista", WHITE), (cur if on else RO.REG, f"pasada {k}/6", GOLD))):
        x = x0 + kk * (cw + gap); paste_img(lay, ink(arr, cw, ch), x, y, cw, ch, radius=12, outline=GOLD if kk else None)
        chip(ImageDraw.Draw(lay, "RGBA"), x + cw / 2, y + 26, T_(lab), col=col, size=20, anchor="mm")
    xs = np.arange(0, 7); ys = np.array(RO.ACC) * 100
    chart(lay, (x0 + 2 * cw + gap + 30, 560, 990, 990), [dict(xs=xs, ys=ys, upto=k, col=GOLD, name="acierto", val_fmt=lambda v: dec(v) + " %")],
          "", xmax=6, ylim=(45, 100), yticks=[50, 75, 100], logx=False, logy=False, xlabel="pasada", yfmt=lambda v: f"{int(v)}")
    d = ImageDraw.Draw(lay, "RGBA"); d.text((CX, 1070), T_(f"recupera el {dec(RO.ACC[k] * 100)} %"), font=font(FS["head"], 60), fill=GOLD if k >= 5 else WHITE, anchor="mm")
    label(d, CX, 1135, "en una zona del grabado que nunca vio", SOFT, 22)

HIST = [(1986, "retropropagación", "MLP"), (1997, "LSTM", "RNN"), (1998, "LeNet", "CNN"), (2006, "redes de creencias profundas", "DBN"),
        (2012, "AlexNet", "CNN"), (2014, "GAN", "GAN"), (2017, "Transformer", "Transf."), (2020, "difusión", "")]
def history(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    for i, (yr, name, ep) in enumerate(HIST):
        a = ease((t - 0.2 - i * 0.35) / 0.3); y = 560 + i * 72; hot = yr == 2006 and s.a(t, "dosmil") > 0
        if a <= 0: continue
        col = GOLD if hot else WHITE
        d.text((150, y), str(yr), font=font(FS["head"], 40), fill=col + (int(255 * a),), anchor="lm")
        d.text((330, y), name, font=font(FS["body"], 34), fill=col + (int(255 * a),), anchor="lm")
        if ep: chip(d, 980, y, ep, col=(GOLD if hot else SOFT) + (int(255 * a),), size=22, anchor="rm", bg=(30, 40, 54, int(255 * a)))
        if hot: d.rounded_rectangle((120, y - 32, 1000, y + 32), radius=14, outline=GOLD, width=3)
    if s.a(t, "apenas") > 0: label(d, CX, 1145, "hoy apenas se usan · pero abrieron camino", SOFT + (int(255 * s.a(t, "apenas")),), 24, bold=True)

SERIE8 = [("MLP", "Perceptrón"), ("CNN", "Convolucional"), ("RNN", "Recurrente"), ("Transf.", "Transformer"), ("AE", "Autoencoder"), ("GAN", "Adversaria"), ("GNN", "Grafos"), ("DBN", "Creencias")]
def season(lay, s, t):
    d = ImageDraw.Draw(lay, "RGBA")
    for i, (sig_, name) in enumerate(SERIE8):
        a = ease((t - i * 0.15) / 0.3); r_, c_ = divmod(i, 4); x = 110 + c_ * 225; y = 600 + r_ * 170
        d.rounded_rectangle((x, y, x + 205, y + 145), radius=18, fill=(40, 32, 14, int(255 * a)), outline=GOLD + (int(255 * a),), width=3)
        d.text((x + 102, y + 52), sig_, font=font(FS["head"], 40), fill=GOLD + (int(255 * a),), anchor="mm"); d.text((x + 102, y + 102), name, font=font(FS["body"], 24), fill=SOFT + (int(255 * a),), anchor="mm")
    if s.a(t, "cuentame") > 0: label(d, CX, 1000, "¿qué arquitectura quieres ver?", GOLD, 34, bold=True)

def end(tl_, t):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    d.text((W / 2, 720), "REDES NEURONALES", font=font(FS["head"], 84), fill=GOLD, anchor="mm")
    d.text((W / 2, 810), "Fin de la temporada 1", font=font(FS["body"], 46), fill=WHITE, anchor="mm")
    d.rounded_rectangle((200, 900, 880, 1140), radius=24, fill=(16, 22, 32, 255), outline=(44, 58, 74, 255), width=2)
    d.text((W / 2, 960), "COMENTA QUÉ QUIERES VER", font=font(FS["monob"], 28), fill=GOLD, anchor="mm")
    d.text((W / 2, 1020), "código completo en la descripción", font=font(FS["body"], 30), fill=SOFT, anchor="mm")
    d.text((W / 2, 1090), HANDLE, font=font(FS["head"], 52), fill=WHITE, anchor="mm")
    return img

SCENES = dict(hook=hook, rbm=rbm, energy=energy, dream=dream, formula=formula, code=code, stack=stack, learn=learn, restore=restore, history=history, season=season, _end=end)
if __name__ == "__main__":
    cli(tl, SCENES, "ep08_dbn_piloto")
