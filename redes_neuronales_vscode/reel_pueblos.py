"""Ep. 3b · LSTM · «Uno de estos pueblos no existe».  Uso: python3 reel_pueblos.py prev f… | chunk 85 | encode"""
import numpy as np, pickle, json
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reel_common import *

RUN = pickle.load(open("work/run_pueblos.pkl", "rb")); SEL = json.load(open("data/pueblos_sel.json"))
L, CK = RUN["L"], RUN["CK"]; ITERS = CK[-1]
Ls = np.convolve(np.pad(L, 12, mode="edge"), np.ones(25) / 25, "valid")
EVO, ROUNDS = SEL["evo"], SEL["rounds"]
REALS = [x["n"] for x in RUN["recs"]]; rr = np.random.default_rng(8); STREAM = [n for n in (REALS[i] for i in rr.choice(len(REALS), 200, replace=False)) if len(n) <= 16][:70]
CAPS = ["ruido puro", "empieza a separar palabras", "sílabas casi reales", "ya suena a pueblo", "un pueblo que no existe"]
PAPER, INK = (244, 244, 238), (14, 16, 20)
SIGN_FONT = (FS["monob"][0], {"Weight": 800})

SS = 2
def sign(w, h, full, shown=None, edge=None, glow=None, caret=False):
    shown = full if shown is None else shown; pad = 50
    lay = Image.new("RGBA", ((w + 2 * pad) * SS, (h + 2 * pad) * SS), (0, 0, 0, 0))
    if glow:
        gl = Image.new("RGBA", lay.size, (0, 0, 0, 0)); ImageDraw.Draw(gl).rounded_rectangle((pad * SS, pad * SS, (pad + w) * SS, (pad + h) * SS), radius=26 * SS, fill=glow + (190,))
        lay.alpha_composite(gl.filter(ImageFilter.GaussianBlur(18 * SS)))
    d = ImageDraw.Draw(lay); box = (pad * SS, pad * SS, (pad + w) * SS, (pad + h) * SS)
    d.rounded_rectangle(box, radius=24 * SS, fill=PAPER, outline=edge or (30, 34, 40), width=(6 if edge else 3) * SS)
    ins = 13 * SS; d.rounded_rectangle((box[0] + ins, box[1] + ins, box[2] - ins, box[3] - ins), radius=14 * SS, outline=INK, width=6 * SS)
    size = 112
    while size > 30 and font(SIGN_FONT, size).getlength(full) > (w - 100): size -= 4
    fs = font(SIGN_FONT, size * SS); tw = fs.getlength(full); x0 = box[0] + (box[2] - box[0] - tw) / 2; cy = (box[1] + box[3]) / 2
    d.text((x0, cy), shown, font=fs, fill=INK, anchor="lm")
    if caret:
        cx = x0 + fs.getlength(shown) + 6 * SS; d.rectangle((cx, cy - size * SS * .36, cx + 8 * SS, cy + size * SS * .36), fill=GOLD)
    return lay.resize((lay.width // SS, lay.height // SS), Image.LANCZOS)
def put(img, lay, cx, cy, scale=1.0, alpha=1.0):
    if scale != 1: lay = lay.resize((int(lay.width * scale), int(lay.height * scale)), Image.LANCZOS)
    if alpha < 1: lay = faded(lay.copy(), alpha)
    img.alpha_composite(lay, (int(cx - lay.width / 2), int(cy - lay.height / 2)))

def check(d, x, y, s, col, w=7): d.line([(x - s * .5, y), (x - s * .12, y + s * .42), (x + s * .55, y - s * .42)], fill=col, width=w, joint="curve")
def cross(d, x, y, s, col, w=7): d.line([(x - s * .42, y - s * .42), (x + s * .42, y + s * .42)], fill=col, width=w); d.line([(x - s * .42, y + s * .42), (x + s * .42, y - s * .42)], fill=col, width=w)
def ring(d, cx, cy, r, p, num):
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(40, 52, 66), width=10)
    if p > 0: d.arc((cx - r, cy - r, cx + r, cy + r), start=-90, end=-90 + 360 * p, fill=GOLD, width=10)
    d.text((cx, cy), str(num), font=font(FS["head"], 70), fill=WHITE, anchor="mm")

SLOT_Y = [600, 820, 1040]; LET = "ABC"
def round_view(img, R_, rf, reveal_at=None, count_from=None):
    """Dibuja 3 carteles; reveal_at: fotograma relativo en que empieza la revelación (None = sin revelar)."""
    items = R_["items"]; inv = next(i for i, it in enumerate(items) if not it["real"])
    order = [i for i in range(3) if i != inv] + [inv]; rev = {i: (reveal_at + 8 * k + (6 if i == inv else 0)) if reveal_at is not None else 1e9 for k, i in enumerate(order)}
    for i, it in enumerate(items):
        cy = SLOT_Y[i]; ap = ease((rf + 6 - i * 4) / 10); tr = rf - rev[i]; isr = tr >= 0
        glow = (GREEN if it["real"] else GOLD) if isr else None
        sc = 1 + (0.08 * np.exp(-tr / 5) * np.cos(tr * 0.9) if isr and not it["real"] else 0)
        put(img, sign(720, 150, it["name"], edge=glow, glow=glow), 600, cy + (1 - ap) * 40, scale=sc, alpha=ap)
        d = ImageDraw.Draw(img, "RGBA")
        d.ellipse((118, cy - 44, 206, cy + 44), fill=(22, 30, 40), outline=glow or (70, 86, 104), width=4)
        d.text((162, cy), LET[i], font=font(FS["head"], 50), fill=glow or WHITE, anchor="mm")
        if isr:
            a = ease(tr / 6); lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
            if it["real"]: check(ld, 270, cy + 102, 28, GREEN); ld.text((300, cy + 102), f"REAL · {it['prov']}", font=font(FS["monob"], 30), fill=GREEN, anchor="lm")
            else: cross(ld, 270, cy + 102, 28, GOLD); ld.text((300, cy + 102), "INVENTADO POR LA RED", font=font(FS["monob"], 30), fill=GOLD, anchor="lm")
            img.alpha_composite(faded(lay, a))
    return inv, rev[inv]

def countdown(img, rf, start, y=1290):
    d = ImageDraw.Draw(img, "RGBA"); k = min(int((rf - start) // 20), 2); p = ((rf - start) % 20) / 20 if rf < start + 60 else 1
    ring(d, W / 2, y, 64, p, 3 - k)

def s_hook(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["Uno de estos pueblos", "no existe"], 320, size=80)
    chip(d, W / 2, 470, ROUNDS[0]["comm"], col=WHITE, size=30, anchor="mm", bg=(30, 40, 54, 255))
    round_view(img, ROUNDS[0], rf)
    if rf >= 24: countdown(img, rf, 24)
    d = ImageDraw.Draw(img, "RGBA"); body(d, ["Lo ha inventado una red neuronal"], 1420, size=40, col=GOLD)
    return img

def s_reveal1(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["Uno de estos pueblos", "no existe"], 320, size=80)
    chip(d, W / 2, 470, ROUNDS[0]["comm"], col=WHITE, size=30, anchor="mm", bg=(30, 40, 54, 255))
    inv, t_inv = round_view(img, ROUNDS[0], rf + 20, reveal_at=20)
    if rf + 20 > t_inv + 6:
        a = ease((rf + 20 - t_inv - 6) / 8); lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay); it = ROUNDS[0]["items"][inv]
        ld.text((W / 2, 1300), T_(f"{LET[inv]} era el falso"), font=font(FS["head"], 66), fill=GOLD, anchor="mm")
        body(ld, [f"se parece a {it['near']} ({it['prov']}), que sí existe"], 1385, size=34)
        img.alpha_composite(faded(lay, a))
    return img

def s_how(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, [f"Leyó los {fmt(SEL['nnames'])} pueblos", "de España"], 320, size=78)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay); f = font(FS["body"], 40); off = rf * 3.2
    for k, nm in enumerate(STREAM):
        col_ = k % 2; row = k // 2; x = [310, 770][col_]; y = 1300 - row * 58 + off + col_ * 29
        if 470 < y < 1290: ld.text((x, y), nm, font=f, fill=SOFT, anchor="mm")
    m = Image.new("L", (1, H), 0)
    for yy_ in range(H): m.putpixel((0, yy_), int(255 * np.clip(min((yy_ - 470) / 160, (1290 - yy_) / 160), 0, 1)))
    lay.putalpha(Image.composite(lay.getchannel("A"), Image.new("L", (W, H), 0), m.resize((W, H))))
    img.alpha_composite(lay); d = ImageDraw.Draw(img, "RGBA")
    body(d, ["y aprendió a adivinar la siguiente letra", "según la comunidad autónoma"], 1360, size=40, col=WHITE)
    return img

SEG = 30
def s_learn(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    ci = min(4, rf // SEG); lf = rf - ci * SEG if ci < 4 else rf - 4 * SEG; prev = CK[ci - 1] if ci else 1
    upto = prev + (CK[ci] - prev) * ease(lf / 12); full = EVO[ci]; nshow = int(np.clip((lf - 3) * len(full) / 16, 0, len(full)))
    headline(d, ["Así aprende", "a inventar"], 300, size=82)
    chip(d, W / 2, 462, SEL["comm1"], col=WHITE, size=30, anchor="mm", bg=(30, 40, 54, 255))
    done = ci == 4 and nshow >= len(full)
    put(img, sign(860, 230, full, full[:nshow], caret=(nshow < len(full)) or (rf // 8) % 2 == 0, edge=GOLD if done else None, glow=GOLD if done else None), 540, 672)
    d = ImageDraw.Draw(img, "RGBA")
    d.text((W / 2, 838), T_(CAPS[ci]), font=font(FS["head"], 44), fill=GOLD if ci == 4 else WHITE, anchor="mm")
    small(d, f"iteración {fmt(CK[ci])} de {fmt(ITERS)}", 892, col=SOFT, size=30)
    if done and lf > 22:
        a = ease((lf - 22) / 8); lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
        check(ld, 250, 960, 30, GREEN); ld.text((285, 960), f"NO EXISTE · se parece a {SEL['near_name']} ({SEL['near_prov']})", font=font(FS["monob"], 28), fill=GREEN, anchor="lm")
        img.alpha_composite(faded(lay, a))
    xs = np.arange(1, len(Ls) + 1)
    chart(img, (110, 1040, 970, 1440), [dict(xs=xs[::5], ys=Ls[::5], upto=upto, col=GOLD, name="error")],
          "error al adivinar la siguiente letra", xmax=ITERS, ylim=(1.2, 4.4), yticks=[1.5, 2.5, 3.5], logy=False,
          yfmt=lambda v: dec(v, 1))
    return img

def s_round2(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["Otra ronda:", ROUNDS[1]["comm"]], 320, size=82)
    inv, t_inv = round_view(img, ROUNDS[1], rf, reveal_at=84)
    if rf < 84:
        if rf >= 20: countdown(img, rf, 20)
        d = ImageDraw.Draw(img, "RGBA"); body(d, ["¿Cuál es el inventado?"], 1420, size=40, col=GOLD)
    elif rf > t_inv + 6:
        a = ease((rf - t_inv - 6) / 8); lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay); it = ROUNDS[1]["items"][inv]
        ld.text((W / 2, 1300), T_(f"{LET[inv]} era el falso"), font=font(FS["head"], 66), fill=GOLD, anchor="mm")
        body(ld, [f"se parece a {it['near']} ({it['prov']}), que sí existe"], 1385, size=34)
        img.alpha_composite(faded(lay, a))
    return img

def s_end(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["Comenta tu comunidad", "y te invento un pueblo"], 330, size=74)
    put(img, sign(760, 200, "¿Dónde estoy?", edge=GOLD, glow=GOLD), 540, 720, scale=1 + 0.03 * np.sin(rf / 8))
    d = ImageDraw.Draw(img, "RGBA")
    small(d, f"{fmt(SEL['nnames'])} pueblos · {SEL['ncomm']} comunidades · red LSTM de {fmt(SEL['nparams'])} parámetros", 960, col=SOFT, size=28)
    d.text((W / 2, 1120), T_("Redes Neuronales · Ep. 3"), font=font(FS["monob"], 32), fill=GOLD, anchor="mm")
    d.text((W / 2, 1210), HANDLE, font=font(FS["head"], 56), fill=WHITE, anchor="mm")
    return img

TL = Timeline([("hook", 90), ("reveal1", 60), ("how", 60), ("learn", 150), ("round2", 120), ("end", 75)],
              dict(hook=s_hook, reveal1=s_reveal1, how=s_how, learn=s_learn, round2=s_round2, end=s_end))

def make_audio():
    A = Audio(TL.N); A.pad((98, 146.8, 196, 246.9))
    for nm in ("reveal1", "how", "learn", "round2", "end"): A.at(TL.start(nm) - 6, A.whoosh(), 0.2)
    A.at(0, A.tone(55, 0.6, 9), 0.5)
    tick = A.tone(880, 0.18, 30)
    for k in range(3): A.at(24 + 20 * k, tick, 0.15); A.at(TL.start("round2") + 20 + 20 * k, tick, 0.15)
    def reveals(s0, at0, items):
        inv = next(i for i, it in enumerate(items) if not it["real"]); order = [i for i in range(3) if i != inv] + [inv]
        for k, i in enumerate(order):
            fr = s0 + at0 + 8 * k + (6 if i == inv else 0); f0 = 523.3 if i == inv else 660
            A.at(fr, A.tone(f0, 0.7, 5) + (0.8 * A.tone(f0 * 1.26, 0.7, 5) if i == inv else 0.6 * A.tone(f0 * 1.5, 0.7, 5)), 0.18 if i == inv else 0.11)
    reveals(TL.start("reveal1") - 20, 20, ROUNDS[0]["items"]); reveals(TL.start("round2"), 84, ROUNDS[1]["items"])
    rng = np.random.default_rng(0); m = int(0.012 * A.sr); cl = rng.normal(0, 1, m) * np.exp(-np.arange(m) / (0.002 * A.sr))
    s0 = TL.start("learn")
    for ci in range(5):
        for k in range(len(EVO[ci])): A.at(s0 + ci * SEG + 3 + k * 16 / len(EVO[ci]), cl, 0.12)
    A.chime(TL.start("end")); return A

if __name__ == "__main__":
    cli(TL, "ep3b_reel_pueblos", make_audio)
