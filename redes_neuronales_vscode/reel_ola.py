"""Ep. 8b · DBN · «Una red restaura La gran ola».  Uso: python3 reel_ola.py prev f… | chunk 85 | encode"""
import numpy as np, pickle
from PIL import Image, ImageDraw
from reel_common import *
import train_dbn_ola as T
import repair_ola as RP

RUN = pickle.load(open("work/run_dbn_ola_v2.pkl", "rb")); S1, E1 = RUN["S1"], RUN["E1"]; P = T.PS; sig = T.sig
E1s = E1.copy()
for i in range(1, len(E1s)): E1s[i] = 0.98 * E1s[i - 1] + 0.02 * E1[i]
REG, _ = RP.region(); DMG, MASK = RP.damage(REG); HIST = RP.repair(RUN["W1"], RUN["b1"], RUN["c1"], DMG, MASK, hist=True)
ACC = [float((DMG == REG)[MASK].mean())] + [float(((e > .5) == (REG > .5))[MASK].mean()) for e in HIST]
PAPER, INKC = (240, 234, 220), (22, 26, 34)
def ink(a, w, h):
    a = (np.asarray(a) > 0.5)[..., None]
    return Image.fromarray(np.where(a, np.array(INKC, np.uint8), np.array(PAPER, np.uint8)).astype(np.uint8)).resize((int(w), int(h)), Image.NEAREST)
RH, RW = REG.shape; COLH = 760; COLW = int(COLH * RW / RH)
PRINT = None

def two_cols(img, left, right, labels, y=470, rcol=GOLD):
    gap = 40; x0 = (W - 2 * COLW - gap) // 2
    for k, (arr, lab) in enumerate(((left, labels[0]), (right, labels[1]))):
        x = x0 + k * (COLW + gap); paste_img(img, ink(arr, COLW, COLH), x, y, COLW, COLH, radius=16, outline=rcol if k == 1 else None)
        chip(ImageDraw.Draw(img, "RGBA"), x + COLW / 2, y + 34, T_(lab), col=rcol if k == 1 else WHITE, size=28, anchor="mm")

def s_hook(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["40 % de píxeles rotos.", "Una red los ha arreglado"], 320, size=74)
    two_cols(img, DMG, HIST[-1], ("antes", "después"))
    d = ImageDraw.Draw(img, "RGBA"); small(d, "La gran ola · Hokusai, ~1831 · en blanco y negro", 1275, col=SOFT, size=30)
    body(d, ["Y nunca había visto esta parte del grabado"], 1360, size=40, col=GOLD)
    return img

def s_what(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["Un trozo de", "La gran ola"], 320, size=86)
    pw = 960; ph = int(pw * T.HT / T.WID); px, py = 60, 560
    paste_img(img, ink(T.INK, pw, ph), px, py, pw, ph, radius=16); d = ImageDraw.Draw(img, "RGBA")
    cut = px + pw * T.CUT / T.WID; y0r, x0r = 20, T.CUT + 2; s = pw / T.WID
    d = shade(img, (cut, py, px + pw, py + ph), (8, 12, 18, 100))
    a = 0.6 + 0.4 * np.sin(rf / 4); d.rounded_rectangle((px + x0r * s, py + y0r * s, px + pw - 2, py + (y0r + RH) * s), radius=8, outline=GOLD + (int(255 * a),), width=6)
    chip(d, px + 20, py + 20, "APRENDE CON ESTO", col=WHITE, size=26); chip(d, px + pw - 20, py + 20, "ESTO, NUNCA", col=GOLD, size=26, anchor="rt")
    body(d, ["La red aprende con trocitos de 20×20 píxeles", "de la parte izquierda del grabado"], 1180, size=40)
    return img

STEPS, CYC = 30, 40
class Dreamer:
    def __init__(s): s.r = np.random.default_rng(11); s.reset()
    def reset(s): s.v = (s.r.random((8, P * P)) < T.X.mean()).astype(np.float32); s.pv = s.v.copy()
    def step(s, Wt, b, c):
        h = (s.r.random((8, Wt.shape[1])) < sig(s.v @ Wt + c)).astype(np.float32); s.pv = sig(h @ Wt.T + b)
        s.v = (s.r.random(s.pv.shape) < s.pv).astype(np.float32)
DREAMS = {}
def precompute_dreams(n):                      # estado de los sueños por fotograma (determinista)
    dr = Dreamer()
    for rf in range(n):
        si = snap_learn(rf, n); _, Wt, b, c = S1[si]; ph = rf % CYC
        if ph == 0: dr.reset()
        elif ph <= STEPS: dr.step(Wt, b, c)
        DREAMS[rf] = (dr.pv if ph > 0 else dr.v).copy()
def snap_learn(rf, n): return int(round((min(rf, n - 20) / (n - 20)) ** 1.5 * (len(S1) - 1)))

def s_learn(rf, n):
    if not DREAMS: precompute_dreams(n)
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["Aprende", "soñando"], 288, size=86)
    small(d, "parte de ruido y lo convierte en grabado", 452, col=SOFT, size=32)
    ts, step = 190, 205; x0 = (W - (4 * step - 15)) // 2
    for j in range(8):
        r_, c_ = divmod(j, 4); paste_img(img, ink(DREAMS[rf][j].reshape(P, P), ts, ts), x0 + c_ * step, 500 + r_ * step, ts, ts, radius=14)
    d = ImageDraw.Draw(img, "RGBA")
    si = snap_learn(rf, n); stp = S1[si][0]
    small(d, f"paso {fmt(stp)} de {fmt(len(E1))} · capa 1 (300 neuronas)", 945, col=SOFT, size=30)
    xs = np.arange(1, len(E1s) + 1)[::25]; ys = E1s[::25]; lo, hi = float(E1s[100:].min()), float(E1s.max())
    chart(img, (110, 990, 970, 1400), [dict(xs=xs, ys=ys, upto=stp, col=GOLD, name="error")],
          "error al reconstruir lo que ve", xmax=len(E1s), ylim=(lo * 0.9, hi * 1.05), yticks=[0.1, 0.15, 0.2], logy=False, xlabel="paso", yfmt=lambda v: dec(v, 2))
    return img

def s_stack(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["Y apila otra capa", "encima"], 320, size=82)
    boxes = [("píxeles · 400", MUTED, 1180), ("capa 1 · 300 neuronas", BLUE, 900), ("capa 2 · 120 neuronas", GOLD, 620)]
    for i, (lab, col, y) in enumerate(boxes):
        a = ease((rf - i * 14) / 10)
        if a <= 0: continue
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
        ld.rounded_rectangle((190, y - 70, 890, y + 70), radius=22, fill=(18, 26, 36, 255), outline=col + (255,), width=5)
        ld.text((W / 2, y), T_(lab), font=font(FS["head"], 42), fill=col, anchor="mm")
        if i > 0:
            ld.line((W / 2, y + 74, W / 2, y + 206), fill=col, width=6)
            ld.polygon([(W / 2 - 14, y + 92), (W / 2 + 14, y + 92), (W / 2, y + 74)], fill=col); ld.polygon([(W / 2 - 14, y + 188), (W / 2 + 14, y + 188), (W / 2, y + 206)], fill=col)
        img.alpha_composite(faded(lay, a))
    d = ImageDraw.Draw(img, "RGBA")
    body(d, ["Cada capa aprende sobre la anterior:", "eso es una Deep Belief Network"], 1360, size=40)
    return img

def s_restore(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    k = int(np.clip((rf - 12) // 11 + 1, 0, 6)); cur = DMG if k == 0 else HIST[k - 1]
    headline(d, ["Ahora,", "a restaurar"], 290, size=82)
    gap = 40; ch = 540; cw = int(ch * RW / RH); x0 = (W - 2 * cw - gap) // 2; y = 440
    for kk, (arr, lab, col) in enumerate(((DMG, "dañado", WHITE), (cur, f"pasada {k}/6", GOLD))):
        x = x0 + kk * (cw + gap); paste_img(img, ink(arr, cw, ch), x, y, cw, ch, radius=14, outline=GOLD if kk else None)
        chip(ImageDraw.Draw(img, "RGBA"), x + cw / 2, y + 30, T_(lab), col=col, size=26, anchor="mm")
    d = ImageDraw.Draw(img, "RGBA")
    d.text((W / 2, 1040), T_(f"recupera el {dec(ACC[k] * 100)} %"), font=font(FS["head"], 72), fill=GOLD if k >= 5 else WHITE, anchor="mm")
    xs = np.arange(0, 7); ys = np.array(ACC) * 100
    chart(img, (110, 1095, 970, 1450), [dict(xs=xs, ys=ys, upto=k, col=GOLD, name="acierto", val_fmt=lambda v: dec(v) + " %")],
          "píxeles dañados que acierta", xmax=6, ylim=(45, 100), yticks=[50, 75, 100], logx=False, logy=False, xlabel="pasada", yfmt=lambda v: f"{int(v)} %")
    return img

def s_end(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["¿Qué obra", "restauramos?"], 330)
    pw = 960; ph = int(pw * T.HT / T.WID); paste_img(img, ink(T.INK, pw, ph), 60, 520, pw, ph, radius=16, outline=GOLD)
    d = ImageDraw.Draw(img, "RGBA")
    body(d, ["Comenta una obra de dominio público"], 1140)
    d.text((W / 2, 1230), T_("Redes Neuronales · Ep. 8"), font=font(FS["monob"], 32), fill=GOLD, anchor="mm")
    d.text((W / 2, 1320), HANDLE, font=font(FS["head"], 56), fill=WHITE, anchor="mm")
    return img

TL = Timeline([("hook", 75), ("what", 75), ("learn", 165), ("stack", 60), ("restore", 120), ("end", 75)],
              dict(hook=s_hook, what=s_what, learn=s_learn, stack=s_stack, restore=s_restore, end=s_end))

def make_audio():
    A = Audio(TL.N); A.pad((98, 146.8, 196, 233.1))
    for nm in ("what", "learn", "stack", "restore", "end"): A.at(TL.start(nm) - 6, A.whoosh(), 0.2)
    A.at(0, A.tone(55, 0.6, 9), 0.5)
    br = A.whoosh(1.0)
    for f0 in range(0, 165, CYC): A.at(TL.start("learn") + f0, br, 0.18)
    for i in range(3): A.at(TL.start("stack") + i * 14, A.tone(330 * 1.5 ** i, 0.5, 7), 0.12)
    for k in range(1, 7): A.at(TL.start("restore") + 12 + (k - 1) * 11, A.tone(392 * 2 ** ((k - 1) / 6), 0.5, 7), 0.13)
    A.chime(TL.start("end")); return A

if __name__ == "__main__":
    cli(TL, "ep8b_reel_ola", make_audio)
